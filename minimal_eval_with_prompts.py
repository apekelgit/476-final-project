from api_client import call_model_chat_completions
from prompts import (
    Question,
    model_foundation,
    parse_user_prompt,
    extract_final_answer,
)

import re
import time

tests = [
    {
        "id": "math_inequality",
        "type": "numeric",
        "prompt": "Solve for the smallest integer n such that 3n + 5 > 26. Answer with just the integer.",
        "expected": "8",
    },
    {
        "id": "commonsense_ice",
        "type": "text",
        "prompt": (
            "You place an ice cube in a glass of water and mark the water level. "
            "After the ice melts, does the water level rise, fall, or stay the same? "
            "Answer with exactly one of: 'rise', 'fall', 'stay the same'."
        ),
        "expected": "stay the same",
    },
    {
        "id": "logic_race",
        "type": "text",
        "prompt": (
            "In a race, you pass the person in second place. What position are you now in? "
            "Answer with a single word like 'first', 'second', 'third'."
        ),
        "expected": "second",
    },
]

def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    synonyms = {
        "unchanged": "stay the same",
        "no change": "stay the same",
        "same": "stay the same",
        "second place": "second",
        "2nd": "second",
        "first place": "first",
        "third place": "third",
    }
    return synonyms.get(s, s)


def extract_number(s: str):
    if not s:
        return None
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    return m.group(0) if m else None


def grade(expected: str, got: str, kind: str) -> bool:
    if kind == "numeric":
        exp_num = extract_number(expected)
        got_num = extract_number(got)
        return (exp_num is not None) and (got_num == exp_num)
    else:
        return normalize_text(got) == normalize_text(expected)

def evaluate_tests(tests):
    rows = []

    for t in tests:
        q = Question(
            questionID=t["id"],
            domain=t["id"], 
            input=t["prompt"],
            answer=t["expected"],
        )

        user_prompt = parse_user_prompt(q, inferenceAlg="direct")

        resp = call_model_chat_completions(
            prompt=user_prompt,
            system=model_foundation,
            temperature=0.0,
        )

        raw_text = (resp.get("text") or "").strip()
        parsed = extract_final_answer(raw_text)

        is_correct = grade(t["expected"], parsed, t["type"])

        rows.append(
            {
                "id": t["id"],
                "expected": t["expected"],
                "got": parsed,
                "correct": is_correct,
                "status": resp.get("status"),
                "error": resp.get("error"),
                "raw_text": raw_text,
            }
        )

        time.sleep(0.2)

    correct = sum(1 for x in rows if x["correct"])
    print(f"Score: {correct}/{len(rows)} correct")

    for x in rows:
        mark = "✅" if x["correct"] else "❌"
        print(f"{mark} {x['id']}: expected={x['expected']!r}, got={x['got']!r} (HTTP {x['status']})")
        if x["error"]:
            print("  error:", x["error"])

    return rows


if __name__ == "__main__":
    evaluate_tests(tests)
