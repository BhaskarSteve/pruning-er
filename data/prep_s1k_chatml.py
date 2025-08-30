# prep_s1k_chatml.py
import json, os
from datasets import load_dataset

OUT = "data/s1k-chatml.jsonl"
os.makedirs(os.path.dirname(OUT), exist_ok=True)

ds = load_dataset("simplescaling/s1K", split="train")
with open(OUT, "w", encoding="utf-8") as f:
    for ex in ds:
        q = ex["question"].strip()
        ans = str(ex["attempt"]).strip()
        # trace = str(ex["gemini_thinking_trajectory"]).strip()
        # gemini trace is a list length 1 per dataset card
        trace_list = ex.get("thinking_trajectories") or []
        trace = (trace_list[0] if trace_list else "").strip()

        # messages: user → think → answer
        obj = {
            "messages": [
                {"role": "user", "content": q},
                {"role": "think", "content": trace},
                {"role": "answer", "content": ans}
            ]
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        # obj = {
        #     "messages": [
        #         {"role": "user", "content": q}
        #     ]
        # }
        # f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Wrote {OUT}")
