import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

with open("cr_raw.json", encoding="utf-8") as f:
    data = json.load(f)

for qid in ["cr_08", "cr_09", "cr_19", "cr_20", "cr_21", "cr_22"]:
    r = data["clinical_review"][qid]
    q = r["question"]
    level = r["safety_level"]
    topic = r["matched_topic"]
    ans = r["answer"][:400]
    print(f"=== {qid} ===")
    print(f"Q: {q}")
    print(f"Level: {level}  Topic: {topic}")
    print(f"Answer: {ans}")
    print()

# Also check AT-005 and AT-017_t2
for qid in ["AT-005", "AT-017_t2", "AT-019"]:
    r = data["acceptance_tests"][qid]
    q = r["question"]
    level = r["safety_level"]
    topic = r["matched_topic"]
    ans = r["answer"][:300]
    print(f"=== {qid} ===")
    print(f"Q: {q}")
    print(f"Level: {level}  Topic: {topic}")
    print(f"Answer: {ans}")
    print()
