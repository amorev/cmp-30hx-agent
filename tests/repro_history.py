#!/usr/bin/env python3
"""Репро: несколько ходов подряд с историей (проверка пустых ответов)."""
import json
import socket
import sys

PORT = 18521


def req(o):
    s = socket.create_connection(("127.0.0.1", PORT), timeout=300)
    s.sendall((json.dumps(o) + "\n").encode())
    f = s.makefile("r", encoding="utf-8")
    line = f.readline()
    s.close()
    return json.loads(line)


def gen(o):
    s = socket.create_connection(("127.0.0.1", PORT), timeout=600)
    s.sendall((json.dumps(o) + "\n").encode())
    f = s.makefile("r", encoding="utf-8")
    toks = []
    while True:
        line = f.readline()
        if not line:
            break
        m = json.loads(line)
        if m.get("type") == "token":
            toks.append(m["text"])
        elif m.get("type") == "done":
            s.close()
            return "".join(toks), m
        elif m.get("type") == "error":
            s.close()
            return "".join(toks), m
    s.close()
    return "".join(toks), {}


def main():
    req({"cmd": "reset"})
    hist = []
    qs = ["Привет, как дела?", "Который час?",
          "какой сейчас курс биткоина", "Расскажи про нейросети"]
    bad = 0
    for i, q in enumerate(qs, 1):
        t, d = gen({"cmd": "generate", "text": q, "history": hist, "max_tokens": 80})
        print("%d. q=%-28r ans=%r tokens=%s cached=%s new=%s finish=%s" % (
            i, q[:28], t[:50], d.get("tokens"), d.get("cached_tokens"),
            d.get("new_tokens"), d.get("finish")))
        if d.get("tokens") == 0:
            print("   !! ПУСТОЙ ОТВЕТ")
            bad += 1
        hist.append({"role": "user", "content": q})
        hist.append({"role": "assistant", "content": "\n\n" + t})
    print("пустых ответов:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
