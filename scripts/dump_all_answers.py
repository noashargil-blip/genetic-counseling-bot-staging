import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

with open("cr_raw.json", encoding="utf-8") as f:
    data = json.load(f)

cr = data["clinical_review"]
at = data["acceptance_tests"]

print("=== CLINICAL REVIEW ===")
for qid, r in cr.items():
    print(f"\n--- {qid} ---")
    print(f"Q: {r['question']}")
    print(f"Level: {r['safety_level']}  Topic: {r['matched_topic']}")
    print(f"A: {r['answer'][:600]}")

print("\n=== ACCEPTANCE TESTS ===")
for qid, r in at.items():
    print(f"\n--- {qid} ---")
    print(f"Q: {r['question']}")
    print(f"Level: {r['safety_level']}  Topic: {r['matched_topic']}")
    print(f"A: {r['answer'][:300]}")
