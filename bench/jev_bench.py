import json, time, urllib.request, urllib.error, pathlib, statistics
KEY = pathlib.Path.home().joinpath("keys/openrouter").read_text().strip()
MODEL = "typesafe/jev-1.13-20260917"
Q = {"c": {"type": "choice",
    "instructions": "This is the last ~30 seconds of a live online class transcript (Serbian and English mixed, produced by noisy speech recognition, so names may be misspelled or split, e.g. 'Bog dane', 'Bogdane', 'Bogdan', 'Stamenović'). Is the teacher, in the most recent lines, calling on or directly addressing the student Bogdan right now and expecting him to respond?",
    "criteria": {"called": "the teacher is addressing Bogdan now, asking him a question, to answer, to speak, or checking whether he is there",
                 "not_called": "Bogdan is not being addressed now: someone else is, he is only mentioned, or the name is a misheard ordinary word"}}}
def jev(state):
    body = json.dumps({"model": MODEL, "state": state, "questions": Q}).encode()
    req = urllib.request.Request("https://openrouter.ai/api/alpha/decisions", body,
        {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "User-Agent": "classwatch"})
    t0 = time.time()
    try:
        r = urllib.request.urlopen(req, timeout=20)
    except urllib.error.HTTPError as e:
        raise SystemExit(e.read().decode())
    with r:
        return json.loads(r.read()), time.time() - t0
CASES = [
 (1, "i to je bio drugi zakon termodinamike. Dobro, ko će mi reći šta je entropija? Bog dane, ajde ti."),
 (1, "OK so the next question. Bogdan, can you tell me what the time complexity of quicksort is?"),
 (1, "Dobro. Bogdane, jesi tu? Bogdane? Čuješ li me?"),
 (1, "Stamenoviću, šta ti misliš o ovome, jel ovo tačno?"),
 (1, "a sad bih volela da čujem nekog ko se nije javljao. Bogdan Stamenović. Izvoli."),
 (1, "so what does the derivative tell us here. bok dan, you want to try this one?"),
 (1, "Marko je rekao rekurzija. Dobro. A ti Bogdane, šta kažeš, da li se slažeš sa Markom?"),
 (1, "Let's go down the list. Ana, done. Bogdan, your turn, what did you get for problem three?"),
 (1, "Ne vidim nikog da diže ruku. Bogdane, ajde, šta je odgovor pod b?"),
 (1, "bogda ne, može ti? Koliko je ovde x?"),
 (1, "Okay Bogdan, unmute yourself please and read the next paragraph."),
 (1, "i onda imamo ovaj integral. Bogdan, kako bi ti rešio ovo?"),
 (0, "Dobro, ko će mi reći šta je entropija? Marko, ajde ti."),
 (0, "As Bogdan pointed out last week, the proof works by induction. Now, Ana, what's the base case?"),
 (0, "Bože moj, opet niko nije uradio domaći. Ajmo dalje, strana četrdeset dva."),
 (0, "Bogdan je poslao zadatak juče, hvala. Jovana, možeš ti da nam pročitaš rešenje?"),
 (0, "so the god of the gaps argument, as Bacon said, the bog standard version is weak."),
 (0, "Danas radimo matrice. Otvorite sveske. Prvi zadatak je na tabli."),
 (0, "Bogdana, hvala ti. A sada Nikola, šta si ti dobio?"),
 (0, "The deadline is Friday. Bogdan and Ana already submitted, the rest of you please hurry."),
 (0, "Okay, Stefan, can you share your screen and show us your solution?"),
 (0, "ovo je bog dan, rekao bi moj deda, kad sve ide kako treba. Elem, idemo na sledeću temu."),
 (0, "Thanks Bogdan, that was a great answer. Now let's move on to chapter five, everyone open the book."),
 (0, "Ko je ovde? Petar, Lazar, Milica. Dobro, počinjemo."),
]
lat, rows = [], []
for y, t in CASES:
    r, dt = jev(t); lat.append(dt)
    p = r["answers"]["c"]["probabilities"]["called"]
    rows.append((y, p, t))
    print(f"{'HIT ' if (p>=0.5)==bool(y) else 'MISS'} y={y} p={p:.3f} {dt:.2f}s  {t[:70]}", flush=True)
pos = [p for y,p,_ in rows if y]; neg = [p for y,p,_ in rows if not y]
auroc = sum((a>b)+0.5*(a==b) for a in pos for b in neg)/(len(pos)*len(neg))
acc = sum((p>=0.5)==bool(y) for y,p,_ in rows)/len(rows)
print(f"AUROC {auroc:.3f}  acc@0.5 {acc:.3f}  latency median {statistics.median(lat):.2f}s max {max(lat):.2f}s")
print("sample response:", json.dumps(r)[:600])
