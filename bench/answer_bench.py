import json, time, urllib.request, pathlib, statistics, sys
KEY = pathlib.Path.home().joinpath("keys/openrouter").read_text().strip()
PROMPT = """Live class transcript (speech recognition, noisy, Serbian/English mixed). The teacher just called on the student Bogdan.

TRANSCRIPT:
{t}

Reply in exactly this format, in the language the teacher used for the question:
QUESTION: <the question he must answer, cleaned up, one line>
ANSWER: <what to say out loud, 1-3 short sentences>
POINTERS: <2-3 short bullet keywords he can expand on>"""
CASES = [
 "dakle entropija je mera neuređenosti sistema, i drugi zakon kaže da u izolovanom sistemu ona... dobro. Bog dane, ajde ti, zašto entropija u izolovanom sistemu ne može da opada?",
 "so we covered merge sort and quick sort today. Bogdan, what's the worst case time complexity of quicksort and when does it happen?",
 "u prošlom času smo radili Francusku revoluciju. Bogdane, koje godine je pao Bastilj i zašto je to bitno?",
 "okay so mitochondria, ATP, all of that. Bogdan can you explain what the electron transport chain actually does, in your own words",
 "imamo funkciju f od x jednako x na kvadrat puta e na x. Bogdane, koliki je izvod ove funkcije?",
 "dobro, sad engleski. Bogdan, what is the difference between present perfect and past simple, give me an example",
]
def stream(model, prompt):
    body = json.dumps({"model": model, "stream": True, "max_tokens": 1000, "reasoning": ({"effort": "low"} if "sonnet-5.5" in model else {"enabled": False}),
        "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", body,
        {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "User-Agent": "classwatch"})
    t0 = time.time(); ttft = None; out = []
    with urllib.request.urlopen(req, timeout=60) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]": continue
            d = json.loads(line[6:])
            if "error" in d: out.append(f"[ERROR {d['error']}]")
            ch = (d.get("choices") or [{}])[0]
            dl = ch.get("delta", {})
            if dl.get("refusal"): out.append(f"[REFUSAL {dl['refusal']}]")
            if ch.get("finish_reason") not in (None, "stop"): out.append(f"[finish={ch['finish_reason']}]")
            c = dl.get("content")
            if c:
                if ttft is None: ttft = time.time() - t0
                out.append(c)
    return (ttft if ttft is not None else float('nan')), time.time() - t0, "".join(out)
for model in sys.argv[1:]:
    tt, tot = [], []
    for i, c in enumerate(CASES):
        a, b, txt = stream(model, PROMPT.format(t=c)); tt.append(a); tot.append(b)
        print(f"--- {model} case {i}: ttft {a:.2f}s total {b:.2f}s\n{txt}", flush=True)
    print(f"=== {model}: TTFT median {statistics.median(tt):.2f}s max {max(tt):.2f}s | full answer median {statistics.median(tot):.2f}s max {max(tot):.2f}s", flush=True)
