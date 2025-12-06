from api_client import call_model_chat_completions
from prompts import (Question, model_foundation, parse_user_prompt, extract_final_answer)
from strategies import solve_self_consistency_cot
from agent import solve_auto

import re
import time
import json

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
    
def infer_kind(expected: str, domain: str, question_text: str) -> str:
    if expected is None:
        return "text"
    expected = expected.strip()
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", expected):
        return "numeric"
    return "text"



def load_dev_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for i, ex in enumerate(data):
        qid = str(i)
        dom = ex.get("domain", "unknown")
        inp = ex.get("input", "")
        expected = ex.get("output", "")

        q = Question(
            questionID=qid,
            domain=dom,
            input=inp,
            answer=expected,
        )
        questions.append(q)

    return questions

def evaluate_dev_data(path: str, limit: int | None = None, debug_n: int = 3):
    questions = load_dev_data(path)

    if limit is not None:
        questions = questions[:limit]

    total = len(questions)
    correct = 0

    for i, q in enumerate(questions, start=1):
        debug_mode = i <= debug_n

        pred, raw_text, num_calls = solve_auto(
            q,
            num_samples=3 if debug_mode else 5,
            sc_temperature=0.7,
            debug=debug_mode,
        )

        kind = infer_kind(q.answer, q.domain, q.input)
        is_correct = grade(q.answer, pred, kind)

        if is_correct:
            correct += 1

        if debug_mode:
            print("_______________________________________", flush=True)
            print(f"DEBUG Example {i}/{total}", flush=True)
            print("ID:       ", q.questionID, flush=True)
            print("Domain:   ", q.domain, flush=True)
            print("Question: ", q.input, flush=True)
            print("Expected: ", repr(q.answer), flush=True)
            print("Kind:     ", kind, flush=True)
            print("Pred:     ", repr(pred), flush=True)
            print("LLM calls used:", num_calls, flush=True)
            print("\nRaw text:", flush=True)
            print(raw_text[:800], flush=True)
            print("_______________________________________", flush=True)

        if i % 25 == 0 or i == total:
            print(f"Processed {i}/{total} questions...", flush=True)

    print(f"\naccuracy: {correct}/{total}")


if __name__ == "__main__":
    evaluate_dev_data("cse476_final_project_dev_data.json", limit=5)