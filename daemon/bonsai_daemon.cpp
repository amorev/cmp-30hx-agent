// Bonsai-2 LLM daemon: один llama_context, ручное управление KV,
// инкрементальный prefill по партиалам ASR, снапшот базового префикса.
//
// Протокол: JSON-строка на команду, JSON-строка (или поток строк) на ответ.
// Транспорт: TCP, 127.0.0.1:18521 по умолчанию. Многоклиентный (poll).
//
// Ключевая идея инкрементального prefill:
//   промпт партиала = база + история + "<|im_start|>user\n" + текст
// то есть обрывается СРАЗУ после текста пользователя. Пока ASR удлиняет текст,
// каждый следующий промпт продолжает предыдущий токен-в-токен и весь префикс
// берётся из KV. Хвост (закрытие реплики + маркер ассистента + пустой
// think-блок) досчитывается один раз — в момент старта генерации.

#include "llama.h"
#include "nlohmann/json.hpp"

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;
using clk  = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// Конфигурация
// ---------------------------------------------------------------------------
struct Config {
    std::string model;
    int   n_ctx       = 32768;
    int   n_batch     = 256;
    int   n_threads   = 4;
    int   n_gpu_layers = 99;
    int   port        = 18521;
    std::string host  = "127.0.0.1";
    bool  kv_q8       = true;
    bool  warmup      = true;
    std::string system_prompt;
    // сэмплирование
    float temp     = 0.7f;
    float top_p    = 0.8f;
    int   top_k    = 20;
    float min_p    = 0.0f;
    float presence_penalty = 1.5f;
    float repeat_penalty   = 1.0f;
    int   penalty_last_n   = 64;
    int   max_gen_tokens   = 256;
};

static std::string env_str(const char * n, const std::string & def) {
    const char * v = std::getenv(n);
    return (v && *v) ? std::string(v) : def;
}
static int env_int(const char * n, int def) {
    const char * v = std::getenv(n);
    return (v && *v) ? std::atoi(v) : def;
}
static float env_float(const char * n, float def) {
    const char * v = std::getenv(n);
    return (v && *v) ? (float) std::atof(v) : def;
}
static bool env_bool(const char * n, bool def) {
    const char * v = std::getenv(n);
    if (!v || !*v) return def;
    std::string s(v);
    return !(s == "0" || s == "false" || s == "no" || s == "off");
}

static Config load_config() {
    Config c;
    c.model        = env_str("BONSAI_MODEL", "");
    c.n_ctx        = env_int("BONSAI_CTX", 32768);
    c.n_batch      = env_int("BONSAI_BATCH", 256);
    c.n_threads    = env_int("BONSAI_THREADS", 4);
    c.n_gpu_layers = env_int("BONSAI_GPU_LAYERS", 99);
    c.port         = env_int("BONSAI_PORT", 18521);
    c.host         = env_str("BONSAI_HOST", "127.0.0.1");
    c.kv_q8        = env_bool("BONSAI_KV_Q8", true);
    c.warmup       = env_bool("BONSAI_WARMUP", true);
    c.temp         = env_float("BONSAI_TEMP", 0.7f);
    c.top_p        = env_float("BONSAI_TOP_P", 0.8f);
    c.top_k        = env_int("BONSAI_TOP_K", 20);
    c.min_p        = env_float("BONSAI_MIN_P", 0.0f);
    c.presence_penalty = env_float("BONSAI_PRESENCE_PENALTY", 1.5f);
    c.repeat_penalty   = env_float("BONSAI_REPEAT_PENALTY", 1.0f);
    c.penalty_last_n   = env_int("BONSAI_PENALTY_LAST_N", 64);
    c.max_gen_tokens   = env_int("BONSAI_MAX_GEN_TOKENS", 256);

    {
        const std::string part1 =
            "Ты — голосовой ассистент. Отвечай коротко, по-русски, без markdown. "
            "Твои ответы озвучиваются вслух, поэтому пиши так, как говорят: без "
            "списков, таблиц, ссылок и форматирования. Обычно достаточно одного-двух "
            "предложений.";
        const std::string part2 =
            "Если для ответа нужны свежие данные из интернета (курс валют или "
            "криптовалют, новости, погода, цены), текущее время или вычисление — "
            "ОБЯЗАТЕЛЬНО вызови подходящую функцию, а не отвечай по памяти. Не "
            "выдумывай числа: если данных нет, сначала вызови функцию.";
        const std::string part3 =
            "Когда ты получаешь блок с результатом функции, это ГОТОВЫЕ данные: "
            "сразу отвечай по ним пользователю одной-двумя фразами, не "
            "спрашивай уточнений. Используй только то, что есть в результате, "
            "ничего не добавляй от себя.";
        const std::string part4 =
            "Если в предоставленных данных НЕТ ответа на новый вопрос "
            "пользователя (например, спросили про владельца, а в результате "
            "было только описание компании) — вызови функцию ещё раз с "
            "уточнённым запросом. Никогда не отвечай «у меня нет информации», "
            "если можешь вызвать функцию.";
        c.system_prompt = env_str("BONSAI_SYSTEM",
            part1 + std::string(2, '\n') + part2 + std::string(2, '\n') + part3);
    }
    return c;
}

// ---------------------------------------------------------------------------
// ChatML (Qwen-семейство). Рендерим сами, чтобы промпт ТОКЕН-В-ТОКЕН совпадал
// с тем, что лежит в KV, и чтобы промпт партиала можно было оборвать на тексте
// пользователя.
// ---------------------------------------------------------------------------
static const char * IM_START  = "<|im_start|>";
static const char * IM_END    = "<|im_end|>";

// Пустой think-блок = подавление reasoning.
// Тонкость, проверенная на месте: открывающий маркер токенизируется как
// обычный текст, а литерал закрывающего маркера токенизатор выбрасывает
// (подача такого литерала текстом даёт 0 токенов). Поэтому закрывающий
// маркер вставляется токеном напрямую (Daemon::close_think_token).
static const char * THINK_OPEN = " thinking\n\n";
// Литерал закрывающего маркера: U+FE0F (VS16) + "</thi" + "nk>" = 8 байт.
static const char * THINK_CLOSE = "</think>";  // служебный токен 248069

struct Msg { std::string role, content; };

static std::string asst_head() {
    return std::string(IM_START) + "assistant\n" + THINK_OPEN;
}

static std::string render_msg(const Msg & m) {
    std::string s;
    if (m.role == "user") {
        s += IM_START + std::string("user\n") + m.content + IM_END + "\n";
    } else if (m.role == "assistant") {
        s += asst_head() + m.content + IM_END + "\n";
    } else if (m.role == "tool") {
        s += IM_START + std::string("user\n<tool_response>\n") + m.content +
             "\n</tool_response>" + IM_END + "\n";
    } else {
        s += IM_START + m.role + "\n" + m.content + IM_END + "\n";
    }
    return s;
}


// Блок описаний инструментов. ВАЖНО: сначала системный текст, потом этот блок.
static std::string render_tools(const json & tools) {
    if (!tools.is_array() || tools.empty()) return "";
    std::ostringstream o;
    o << "\n\n# Tools\n\nYou have access to the following functions:\n\n<tools>";
    for (const auto & t : tools) o << "\n" << t.dump();
    o << "\n</tools>";
    o << "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
         "<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n"
         "</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n"
         "that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n"
         "<IMPORTANT>\nReminder:\n"
         "- Function calls MUST follow the specified format: an inner <function=...></function> block "
         "must be nested within <tool_call></tool_call> XML tags\n"
         "- Required parameters MUST be specified\n"
         "- You may provide optional reasoning for your function call in natural language BEFORE the "
         "function call, but NOT after\n"
         "- If there is no function call available, answer the question like normal with your current "
         "knowledge and do not tell the user about function calls\n</IMPORTANT>";
    return o.str();
}

// ---------------------------------------------------------------------------
// Разбор tool_call-разметки из текста ответа.
// ---------------------------------------------------------------------------
struct ToolCall {
    std::string name;
    json args = json::object();
};

static std::string trim_str(const std::string & s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

static std::vector<ToolCall> parse_tool_calls(const std::string & text) {
    std::vector<ToolCall> out;
    const std::string OPEN = "<tool_call>";
    const std::string FN   = "<function=";
    const std::string PAR  = "<parameter=";
    size_t p = 0;
    while (true) {
        size_t a = text.find(OPEN, p);
        if (a == std::string::npos) break;
        size_t b = text.find("</tool_call>", a);
        if (b == std::string::npos) break;
        std::string body = text.substr(a + OPEN.size(), b - a - OPEN.size());
        p = b + 12;

        ToolCall tc;
        size_t f1 = body.find(FN);
        if (f1 == std::string::npos) continue;
        size_t f2 = body.find('>', f1);
        if (f2 == std::string::npos) continue;
        tc.name = trim_str(body.substr(f1 + FN.size(), f2 - f1 - FN.size()));

        size_t q = f2;
        while (true) {
            size_t p1 = body.find(PAR, q);
            if (p1 == std::string::npos) break;
            size_t p2 = body.find('>', p1);
            if (p2 == std::string::npos) break;
            std::string pname = trim_str(body.substr(p1 + PAR.size(), p2 - p1 - PAR.size()));
            size_t p3 = body.find("</parameter>", p2);
            if (p3 == std::string::npos) break;
            std::string val = body.substr(p2 + 1, p3 - p2 - 1);
            // срезаем ровно один ведущий и один хвостовой перевод строки
            if (!val.empty() && val.front() == '\n') val.erase(val.begin());
            if (!val.empty() && val.back() == '\n') val.pop_back();
            tc.args[pname] = val;
            q = p3 + 12;
        }
        if (!tc.name.empty()) out.push_back(tc);
    }
    return out;
}

// ---------------------------------------------------------------------------
// Фильтр потока: не отдавать клиенту разметку вызова и reasoning.
// Возвращает «безопасную» часть накопленного текста.
// ---------------------------------------------------------------------------
static std::string safe_prefix_of(const std::string & raw, bool finished) {
    std::string s = raw;

    // 1) убрать think-блок, если модель всё же начала рассуждать
    size_t nb = s.find_first_not_of(" \t\r\n");
    if (nb != std::string::npos && s.compare(nb, 7, " thinking") == 0) {
        size_t close = s.find("</think>", nb);
        if (close != std::string::npos) {
            s = s.substr(close + 8);
            size_t st = s.find_first_not_of(" \t\r\n");
            if (st != std::string::npos) s = s.substr(st);
        } else if (!finished) {
            return "";  // think-блок ещё открыт — ничего не отдаём
        } else {
            s = "";
        }
    }

    // 2) отрезать разметку вызова инструмента
    size_t tc = s.find("<tool_call>");
    if (tc != std::string::npos) return s.substr(0, tc);

    // 3) удержать хвост, который может оказаться началом "<tool_call>"
    if (!finished) {
        static const std::string MARK = "<tool_call>";
        size_t maxk = std::min<size_t>(s.size(), MARK.size() - 1);
        for (size_t k = maxk; k > 0; --k) {
            if (s.compare(s.size() - k, k, MARK, 0, k) == 0) {
                return s.substr(0, s.size() - k);
            }
        }
    }
    return s;
}

// ---------------------------------------------------------------------------
// Состояние модели
// ---------------------------------------------------------------------------
struct Daemon {
    Config cfg;
    llama_model   * model = nullptr;
    llama_context * ctx   = nullptr;
    const llama_vocab * vocab = nullptr;

    // Токены, реально лежащие в KV (источник истины).
    std::vector<llama_token> kv_tokens;

    // Снапшот базового префикса (система + инструменты).
    std::vector<uint8_t> base_state;
    size_t base_tokens = 0;
    std::string base_text;
    json   tools = json::array();

    // Токен закрывающего think-маркера: литерал "" токенизатор выбрасывает,
    // поэтому вставляем его напрямую по id.
    llama_token close_think_token = -1;

    // Диагностика
    long long n_prefill      = 0;
    long long n_generate     = 0;
    long long n_reset        = 0;
    long long n_snapshot_fallback = 0;

    bool init(const Config & c);
    void shutdown();

    std::string build_base() const {
        std::string s = IM_START + std::string("system\n") + cfg.system_prompt +
                        render_tools(tools) + IM_END + "\n";
        return s;
    }
    // Промпт собирается СЕГМЕНТАМИ: закрывающий think-маркер в историю
    // ассистентских реплик тоже попадает токеном, иначе промпт разойдётся с KV
    // на втором ходу и кэш будет считаться заново.
    using Seg = std::pair<bool, std::string>;   // (is_close_marker, text)

    void push_history(std::vector<Seg> & segs, const json & history) const {
        if (!history.is_array()) return;
        for (const auto & m : history) {
            Msg mm;
            mm.role    = m.value("role", "user");
            mm.content = m.value("content", "");
            if (mm.role == "assistant") {
                segs.push_back({false, asst_head()});
                segs.push_back({true, ""});
                segs.push_back({false, "\n\n" + mm.content + IM_END + "\n"});
            } else {
                segs.push_back({false, render_msg(mm)});
            }
        }
    }

    std::string build_prefix(const std::string & text, const json & history) const {
        std::vector<Seg> segs;
        segs.push_back({false, base_text});
        push_history(segs, history);
        segs.push_back({false, IM_START + std::string("user\n") + text});
        std::string s;
        for (auto & x : segs) s += x.second;
        return s;
    }

    std::string build_generate(const std::string & text, const json & history) const {
        // Продолжение промпта партиала: закрытие реплики + маркер ответа +
        // открывающий think-маркер. Закрывающий маркер вставляется токеном.
        return build_prefix(text, history) + IM_END + "\n" + asst_head();
    }

    std::vector<llama_token> tokenize_segments(const std::string & text,
                                               const json & history,
                                               bool with_generate_tail,
                                               bool add_user = true) const {
        std::vector<Seg> segs;
        segs.push_back({false, base_text});
        push_history(segs, history);
        // продолжение после инструментов: без новой реплики
        if (add_user) {
            segs.push_back({false, IM_START + std::string("user\n") + text});
        }
        if (with_generate_tail) {
            segs.push_back({false, std::string(IM_END) + "\n" + asst_head()});
            segs.push_back({true, ""});       // закрывающий think-маркер
            segs.push_back({false, "\n\n"});
        }
        std::vector<llama_token> out;
        for (auto & x : segs) {
            if (x.first) {
                out.push_back(close_think_token);
            } else {
                auto t = tokenize(x.second);
                out.insert(out.end(), t.begin(), t.end());
            }
        }
        return out;
    }

    std::vector<llama_token> tokenize(const std::string & s) const {
        std::vector<llama_token> toks(s.size() + 64);
        // parse_special ОБЯЗАТЕЛЕН: без него ChatML-разметка рассыпается в текст.
        int n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(),
                               toks.data(), (int32_t) toks.size(),
                               /*add_special*/ false, /*parse_special*/ true);
        if (n < 0) {
            toks.resize(-n);
            n = llama_tokenize(vocab, s.c_str(), (int32_t) s.size(),
                               toks.data(), (int32_t) toks.size(), false, true);
        }
        toks.resize(n < 0 ? 0 : n);
        return toks;
    }

    size_t common_prefix(const std::vector<llama_token> & a) const {
        size_t n = std::min(a.size(), kv_tokens.size());
        size_t i = 0;
        while (i < n && a[i] == kv_tokens[i]) ++i;
        return i;
    }

    bool restore_base();
    bool make_snapshot();
    bool rollback(size_t n, const std::vector<llama_token> & target);
    bool decode_range(const std::vector<llama_token> & toks, size_t from, size_t to, bool want_logits);
    bool prefill(const std::string & text, const json & history, json & out);
    void generate(int fd, const std::string & text, const json & history,
                  int max_tokens, bool is_continue = false);
    json stat() const;
    bool set_tools(const json & t, json & out);
    bool reset(json & out);
};

bool Daemon::init(const Config & c) {
    cfg = c;
    llama_backend_init();

    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = cfg.n_gpu_layers;
    // Вся модель на одной карте: без этого демон раскидывает слои по всем
    // видимым устройствам и вытесняет TTS (грабли №8).
    mparams.split_mode = LLAMA_SPLIT_MODE_NONE;
    mparams.main_gpu   = 0;
    mparams.load_mode  = LLAMA_LOAD_MODE_MMAP;

    model = llama_model_load_from_file(cfg.model.c_str(), mparams);
    if (!model) { std::cerr << "[fatal] не удалось загрузить модель " << cfg.model << "\n"; return false; }
    vocab = llama_model_get_vocab(model);

    auto cparams = llama_context_default_params();
    cparams.n_ctx        = (uint32_t) cfg.n_ctx;
    cparams.n_batch      = (uint32_t) cfg.n_batch;
    cparams.n_ubatch     = (uint32_t) cfg.n_batch;
    cparams.n_seq_max    = 1;
    cparams.n_rs_seq     = 0;   // снапшоты через n_rs_seq не используем (VRAM)
    // n_outputs_max = 1: иначе logits-буфер на весь батч (сотни МиБ) и OOM.
    cparams.n_outputs_max = 1;
    cparams.n_threads      = cfg.n_threads;
    cparams.n_threads_batch = cfg.n_threads;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cparams.type_k = cfg.kv_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
    cparams.type_v = cfg.kv_q8 ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;

    ctx = llama_init_from_model(model, cparams);
    if (!ctx) { std::cerr << "[fatal] llama_init_from_model failed\n"; return false; }

    // Закрывающий think-маркер: литерал токенизатор выбрасывает (проверено:
    // подача его текстом даёт 0 токенов), поэтому вставляем маркер по id.
    // Ищем служебный токен, чей текст — ровно "\n\n" (в Qwen3 это ).
    close_think_token = -1;
    {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        const std::string want = THINK_CLOSE;
        for (llama_token id = n_vocab - 400; id < n_vocab; ++id) {
            char buf[64];
            int np = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
            if (np > 0 && std::string(buf, (size_t) np) == want) {
                close_think_token = id;
                break;
            }
        }
    }
    if (close_think_token < 0) {
        std::cerr << "[fatal] не найден токен закрывающего think-маркера\n";
        return false;
    }
    std::cerr << "[think] закрывающий маркер = токен " << close_think_token << "\n";


    base_text = build_base();
    if (!make_snapshot()) { std::cerr << "[fatal] не удалось посчитать базовый префикс\n"; return false; }

    if (cfg.warmup) {
        // Прогрев decode-пути, чтобы первый ответ не платил за инициализацию ядер.
        auto t0 = clk::now();
        json o;
        prefill("прогрев", json::array(), o);
        double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        std::cerr << "[warmup] prefill " << ms << " ms, kv=" << kv_tokens.size() << "\n";
        restore_base();
        kv_tokens.resize(base_tokens);
    }
    return true;
}

bool Daemon::decode_range(const std::vector<llama_token> & toks, size_t from, size_t to, bool want_logits) {
    const size_t total = to - from;
    if (total == 0) return true;
    const size_t step = (size_t) std::max(1, cfg.n_batch);

    llama_batch batch = llama_batch_init((int32_t) step, 0, 1);
    bool ok = true;
    for (size_t off = from; off < to; off += step) {
        const size_t n = std::min(step, to - off);
        const bool last_chunk = (off + n >= to);
        batch.n_tokens = (int32_t) n;
        for (size_t i = 0; i < n; ++i) {
            batch.token[i]    = toks[off + i];
            batch.pos[i]      = (llama_pos) (off + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            // Логиты нужны только на последнем куске и только для последнего
            // токена: при n_outputs_max=1 единственный выход имеет индекс 0.
            batch.logits[i]   = (want_logits && last_chunk && i == n - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, batch) != 0) { ok = false; break; }
    }
    llama_batch_free(batch);
    return ok;
}

bool Daemon::make_snapshot() {
    // Восстановиться в чистое состояние и посчитать базу заново.
    llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
    kv_tokens.clear();

    auto toks = tokenize(base_text);
    if (toks.size() > (size_t) cfg.n_ctx - 256) {
        std::cerr << "[fatal] базовый префикс не влезает в контекст\n";
        return false;
    }
    auto t0 = clk::now();
    if (!decode_range(toks, 0, toks.size(), false)) return false;
    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();

    kv_tokens = toks;
    base_tokens = toks.size();

    size_t sz = llama_state_seq_get_size(ctx, 0);
    base_state.resize(sz);
    size_t got = llama_state_seq_get_data(ctx, base_state.data(), base_state.size(), 0);
    if (got != sz) {
        std::cerr << "[warn] snapshot size mismatch " << got << " vs " << sz << "\n";
        base_state.resize(got);
    }
    std::cerr << "[base] tokens=" << base_tokens << " decode=" << ms
              << " ms snapshot=" << (base_state.size() / 1024) << " KiB\n";
    return true;
}

bool Daemon::restore_base() {
    if (base_state.empty()) return false;
    size_t n = llama_state_seq_set_data(ctx, base_state.data(), base_state.size(), 0);
    if (n == 0) {
        std::cerr << "[warn] llama_state_seq_set_data failed, fallback на пересчёт базы\n";
        return make_snapshot();
    }
    kv_tokens.resize(base_tokens);
    return true;
}

bool Daemon::rollback(size_t n, const std::vector<llama_token> & target) {
    if (n == kv_tokens.size()) return true;

    if (n >= base_tokens) {
        // Частичный откат. Для линейных (recurrent) слоёв гибридной модели
        // может вернуть false — тогда аварийный путь ниже.
        if (llama_memory_seq_rm(llama_get_memory(ctx), 0, (llama_pos) n, -1)) {
            kv_tokens.resize(n);
            return true;
        }
    }
    // Аварийный путь: снапшот базы + пересчёт хвоста.
    ++n_snapshot_fallback;
    if (!restore_base()) return false;
    kv_tokens.resize(base_tokens);
    if (target.size() >= n && n > base_tokens) {
        if (!decode_range(target, base_tokens, n, false)) return false;
        kv_tokens.assign(target.begin(), target.begin() + n);
    }
    return true;
}

bool Daemon::prefill(const std::string & text, const json & history, json & out) {
    auto t0 = clk::now();

    auto toks = tokenize_segments(text, history, /*with_generate_tail*/ false);

    if (toks.size() > (size_t) cfg.n_ctx - 64) {
        out = {{"ok", false}, {"error", "prompt_too_long"}, {"prompt_tokens", toks.size()},
               {"ctx", cfg.n_ctx}};
        return false;
    }

    const size_t common = common_prefix(toks);
    const size_t cached = common;

    if (!rollback(common, toks)) { out = {{"ok", false}, {"error", "rollback_failed"}}; return false; }
    if (!decode_range(toks, common, toks.size(), /*want_logits*/ true)) {
        out = {{"ok", false}, {"error", "decode_failed"}};
        return false;
    }
    kv_tokens = toks;

    double ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    ++n_prefill;

    out = {
        {"ok", true},
        {"cached_tokens", cached},
        {"new_tokens", toks.size() - cached},
        {"prompt_tokens", toks.size()},
        {"base_tokens", base_tokens},
        {"ms", (long long) (ms * 1000) / 1000.0},
        {"ms_f", ms},
        {"kv_tokens", kv_tokens.size()},
        {"pos_max", (long long) kv_tokens.size() - 1},
        {"fallbacks", n_snapshot_fallback},
    };
    return true;
}

static void send_line(int fd, const std::string & s) {
    std::string line = s + "\n";
    size_t off = 0;
    while (off < line.size()) {
        ssize_t w = ::send(fd, line.data() + off, line.size() - off, MSG_NOSIGNAL);
        if (w <= 0) return;
        off += (size_t) w;
    }
}

void Daemon::generate(int fd, const std::string & text, const json & history,
                      int max_tokens, bool is_continue) {
    auto t0 = clk::now();

    auto toks = tokenize_segments(text, history, /*with_generate_tail*/ true,
                                  /*add_user*/ !is_continue);

    if (toks.size() > (size_t) cfg.n_ctx - (size_t) max_tokens - 16) {
        send_line(fd, json{{"type", "error"}, {"error", "prompt_too_long"},
                           {"prompt_tokens", toks.size()}, {"ctx", cfg.n_ctx}}.dump());
        return;
    }

    const size_t common = common_prefix(toks);
    const size_t cached = common;
    // Диагностика: сколько токенов стоил хвост ассистента
    // (то есть насколько prefill партиала сэкономил генерации).
    const size_t tail_tokens = toks.size() - common;

    if (!rollback(common, toks)) {
        send_line(fd, json{{"type", "error"}, {"error", "rollback_failed"}}.dump());
        return;
    }
    if (common >= toks.size()) {
        // Партиал уже оставил KV ровно на конце промпта: новых токенов нет, но
        // логиты для первого сэмпла нужны. Снимаем последнюю позицию из KV и
        // перекодируем её с логитами (иначе она декодировалась бы дважды).
        if (toks.empty() || common < 2) {
            send_line(fd, json{{"type", "error"}, {"error", "empty_prompt"}}.dump());
            return;
        }
        if (!rollback(common - 1, toks)) {
            send_line(fd, json{{"type", "error"}, {"error", "rollback_failed"}}.dump());
            return;
        }
        if (!decode_range(toks, common - 1, toks.size(), /*want_logits*/ true)) {
            send_line(fd, json{{"type", "error"}, {"error", "decode_failed"}}.dump());
            return;
        }
        kv_tokens = toks;
    } else if (!decode_range(toks, common, toks.size(), /*want_logits*/ true)) {
        send_line(fd, json{{"type", "error"}, {"error", "decode_failed"}}.dump());
        return;
    }
    kv_tokens = toks;

    auto t_prefill_done = clk::now();
    const double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_done - t0).count();

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = true;
    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    if (cfg.penalty_last_n > 0 && (cfg.presence_penalty != 1.0f || cfg.repeat_penalty != 1.0f)) {
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(
            llama_vocab_n_tokens(vocab), cfg.penalty_last_n,
            cfg.repeat_penalty, 0.0f, cfg.presence_penalty));
    }
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(cfg.top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(cfg.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(cfg.min_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(cfg.temp));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    std::string raw;
    size_t sent = 0;
    int n_gen = 0;
    bool stop_eog = false;
    double first_token_ms = -1.0;
    const int limit = std::max(1, max_tokens);

    llama_batch one = llama_batch_init(1, 0, 1);

    // Сгенерированный токен обязан попасть в учёт KV; при остановке по EOG
    // сам маркер конца реплики тоже декодируется и учитывается (грабли №5).
    auto decode_one = [&](llama_token id, bool want_logits) -> bool {
        const size_t pos = kv_tokens.size();
        one.n_tokens = 1;
        one.token[0] = id;
        one.pos[0] = (llama_pos) pos;
        one.n_seq_id[0] = 1;
        one.seq_id[0][0] = 0;
        one.logits[0] = want_logits ? 1 : 0;
        if (llama_decode(ctx, one) != 0) return false;
        {
        }
        kv_tokens.push_back(id);
        return true;
    };

    while (n_gen < limit) {
        if (llama_get_logits_ith(ctx, -1) == nullptr) {
            std::cerr << "DBG no logits at gen step " << n_gen
                      << std::endl;
            send_line(fd, json{{"type", "error"}, {"error", "no_logits"}}.dump());
            break;
        }
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) {
            stop_eog = true;
            // Диагностика: немедленный EOG на первом же шаге означает, что
            // промпт «выглядит законченным» для модели (обычно — обрыв на
            // границе реплики). Печатаем, только если не сгенерировано ничего.
            if (n_gen == 0) {
                std::cerr << "[eog-immediate] prompt=" << toks.size()
                          << " common=" << common << " cached=" << cached
                          << " kv_before=" << (kv_tokens.size() - 1)
                          << " last_tokens:";
                for (size_t i = (toks.size() > 6 ? toks.size() - 6 : 0); i < toks.size(); ++i)
                    std::cerr << " " << toks[i];
                std::cerr << std::endl;
            }
            if (n_gen == 0) {
                // Немедленный EOG на первом шаге: промпт «выглядит
                // законченным». Это не ответ — KV уже сдвинут на маркер,
                // поэтому откатываемся на длину промпта и сообщаем клиенту,
                // чтобы он мог повторить ход на чистом состоянии.
                stop_eog = false;
                rollback(toks.size(), toks);
                break;
            }
            decode_one(id, false);   // маркер конца реплики тоже уходит в KV
            break;
        }
        llama_sampler_accept(smpl, id);

        char buf[512];
        int np = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, /*special*/ false);
        if (np > 0) raw.append(buf, np);

        if (first_token_ms < 0.0 && !raw.empty()) {
            first_token_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
        }

        // инкрементальная выдача отфильтрованного текста
        std::string safe = safe_prefix_of(raw, false);
        if (safe.size() > sent) {
            send_line(fd, json{{"type", "token"}, {"text", safe.substr(sent)}}.dump());
            sent = safe.size();
        }

        // Сгенерированный токен обязан попасть в учёт KV (иначе позиции разъедутся).
        if (!decode_one(id, true)) break;
        ++n_gen;
    }
    llama_batch_free(one);

    std::string safe = safe_prefix_of(raw, true);
    if (safe.size() > sent) {
        send_line(fd, json{{"type", "token"}, {"text", safe.substr(sent)}}.dump());
        sent = safe.size();
    }

    const auto calls = parse_tool_calls(raw);
    json jc = json::array();
    for (const auto & c : calls) jc.push_back({{"name", c.name}, {"arguments", c.args}});

    const double total_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
    const double gen_ms = total_ms - first_token_ms;
    const double tok_s = (n_gen > 1 && gen_ms > 1.0) ? (n_gen - 1) * 1000.0 / gen_ms : 0.0;

    ++n_generate;
    send_line(fd, json{
        {"type", "done"},
        {"text", safe},
        {"raw", raw},
        {"finish", calls.empty() ? (stop_eog ? "stop" : "length") : "tool_calls"},
        {"tool_calls", jc},
        {"tokens", n_gen},
        {"first_token_ms", (long long) first_token_ms},
        {"prefill_ms", (long long) prefill_ms},
        {"cached_tokens", cached},
        {"new_tokens", tail_tokens},
        {"prompt_tokens", toks.size()},
        {"tok_s", tok_s},
        {"total_ms", (long long) total_ms},
        {"kv_tokens", kv_tokens.size()},
    }.dump());
    llama_sampler_free(smpl);
}

json Daemon::stat() const {
    return json{
        {"ok", true},
        {"type", "stat"},
        {"kv_tokens", kv_tokens.size()},
        {"base_tokens", base_tokens},
        {"pos_max", (long long) kv_tokens.size() - 1},
        {"ctx", cfg.n_ctx},
        {"kv_type", cfg.kv_q8 ? "q8_0" : "f16"},
        {"flash_attn", true},
        {"tools", (int) (tools.is_array() ? tools.size() : 0)},
        {"tool_names", [&] {
            json a = json::array();
            if (tools.is_array()) for (auto & t : tools) {
                if (t.contains("function") && t["function"].contains("name"))
                    a.push_back(t["function"]["name"]);
            }
            return a;
        }()},
        {"n_prefill", n_prefill},
        {"n_generate", n_generate},
        {"n_reset", n_reset},
        {"snapshot_fallbacks", n_snapshot_fallback},
        {"snapshot_kib", base_state.size() / 1024},
    };
}

bool Daemon::set_tools(const json & t, json & out) {
    tools = t.is_array() ? t : json::array();
    base_text = build_base();
    for (auto & m : kv_tokens) (void) m;
    if (!make_snapshot()) { out = {{"ok", false}, {"error", "snapshot_failed"}}; return false; }
    out = {{"ok", true}, {"tools", (int) tools.size()}, {"base_tokens", base_tokens},
           {"snapshot_kib", base_state.size() / 1024}};
    return true;
}

bool Daemon::reset(json & out) {
    ++n_reset;
    if (!restore_base()) { out = {{"ok", false}, {"error", "restore_failed"}}; return false; }
    kv_tokens.resize(base_tokens);
    out = {{"ok", true}, {"cached_tokens", base_tokens}, {"kv_tokens", kv_tokens.size()},
           {"pos_max", (long long) kv_tokens.size() - 1}};
    return true;
}

void Daemon::shutdown() {
    if (ctx)   { llama_free(ctx); ctx = nullptr; }
    if (model) { llama_model_free(model); model = nullptr; }
    llama_backend_free();
}

// ---------------------------------------------------------------------------
// Сетевой слой: многоклиентный неблокирующий poll.
// ---------------------------------------------------------------------------
struct Client {
    int fd = -1;
    std::string in;
};

static bool set_nonblock(int fd) {
    int fl = fcntl(fd, F_GETFL, 0);
    return fcntl(fd, F_SETFL, fl | O_NONBLOCK) == 0;
}

int main(int argc, char ** argv) {
    std::signal(SIGPIPE, SIG_IGN);

    Config cfg = load_config();
    if (cfg.model.empty()) { std::cerr << "[fatal] BONSAI_MODEL не задан\n"; return 2; }

    Daemon d;
    if (!d.init(cfg)) return 3;

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) { perror("socket"); return 4; }
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t) cfg.port);
    if (inet_pton(AF_INET, cfg.host.c_str(), &addr.sin_addr) != 1) {
        std::cerr << "[fatal] плохой bind-адрес " << cfg.host << "\n";
        return 4;
    }
    if (bind(srv, (sockaddr *) &addr, sizeof(addr)) < 0) { perror("bind"); return 4; }
    if (listen(srv, 16) < 0) { perror("listen"); return 4; }
    set_nonblock(srv);

    std::cerr << "[ready] слушаю " << cfg.host << ":" << cfg.port
              << " model=" << cfg.model << " ctx=" << cfg.n_ctx
              << " batch=" << cfg.n_batch << " kv=" << (cfg.kv_q8 ? "q8_0" : "f16") << "\n";

    std::vector<Client> clients;

    while (true) {
        std::vector<pollfd> pfds;
        pfds.push_back({srv, POLLIN, 0});
        for (auto & c : clients) pfds.push_back({c.fd, POLLIN, 0});

        int pr = poll(pfds.data(), pfds.size(), 1000);
        if (pr < 0) { if (errno == EINTR) continue; perror("poll"); break; }

        if (pfds[0].revents & POLLIN) {
            while (true) {
                int cfd = accept(srv, nullptr, nullptr);
                if (cfd < 0) break;
                setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
                set_nonblock(cfd);
                clients.push_back(Client{cfd, {}});
                std::cerr << "[conn] fd=" << cfd << " clients=" << clients.size() << "\n";
            }
            continue;  // пересобрать pfds
        }

        for (size_t i = 0; i < clients.size(); ++i) {
            if (!(pfds[i + 1].revents & (POLLIN | POLLHUP | POLLERR))) continue;
            Client & cl = clients[i];

            char buf[8192];
            ssize_t r = recv(cl.fd, buf, sizeof(buf), 0);
            if (r == 0) {
                close(cl.fd);
                std::cerr << "[disc] fd=" << cl.fd << "\n";
                clients.erase(clients.begin() + (long) i);
                --i;
                continue;
            }
            if (r < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
                close(cl.fd);
                clients.erase(clients.begin() + (long) i);
                --i;
                continue;
            }
            cl.in.append(buf, (size_t) r);

            // обработать все полные строки
            while (true) {
                size_t nl = cl.in.find('\n');
                if (nl == std::string::npos) break;
                std::string line = cl.in.substr(0, nl);
                cl.in.erase(0, nl + 1);
                if (line.empty()) continue;

                json req;
                try {
                    req = json::parse(line);
                } catch (const std::exception & e) {
                    send_line(cl.fd, json{{"ok", false}, {"error", "bad_json"},
                                          {"detail", e.what()}}.dump());
                    continue;
                }

                const std::string cmd = req.value("cmd", "");
                try {
                    if (cmd == "ping") {
                        send_line(cl.fd, json{{"ok", true}, {"pong", true},
                                              {"pid", getpid()}}.dump());
                    } else if (cmd == "stat") {
                        send_line(cl.fd, d.stat().dump());
                    } else if (cmd == "reset") {
                        json o;
                        d.reset(o);
                        send_line(cl.fd, o.dump());
                    } else if (cmd == "set_tools") {
                        json o;
                        d.set_tools(req.contains("tools") ? req["tools"] : json::array(), o);
                        send_line(cl.fd, o.dump());
                    } else if (cmd == "prefill") {
                        json o;
                        d.prefill(req.value("text", ""),
                                  req.contains("history") ? req["history"] : json::array(), o);
                        send_line(cl.fd, o.dump());
                    } else if (cmd == "generate") {
                        d.generate(cl.fd,
                                   req.value("text", ""),
                                   req.contains("history") ? req["history"] : json::array(),
                                   req.value("max_tokens", cfg.max_gen_tokens),
                                   req.value("continue", false));
                    } else {
                        send_line(cl.fd, json{{"ok", false}, {"error", "unknown_cmd"},
                                              {"cmd", cmd}}.dump());
                    }
                } catch (const std::exception & e) {
                    send_line(cl.fd, json{{"ok", false}, {"error", "exception"},
                                          {"detail", e.what()}}.dump());
                }
            }
        }
    }

    for (auto & c : clients) close(c.fd);
    close(srv);
    d.shutdown();
    return 0;
}
