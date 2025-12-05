from api_client import call_model_chat_completions
from prompts import (Question, model_foundation, parse_user_prompt, extract_final_answer)

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

def evaluate_dev_data(path: str, limit: int | None = None):
    questions = load_dev_data(path)

    if limit is not None:
        questions = questions[:limit]

    total = len(questions)
    correct = 0
    rows = []

    for i, q in enumerate(questions, start=1):
        user_prompt = parse_user_prompt(q, inferenceAlg="cot")

        resp = call_model_chat_completions(
            prompt=user_prompt,
            system=model_foundation,
            temperature=0.0,
        )

        raw_text = (resp.get("text") or "").strip()
        pred = extract_final_answer(raw_text)

        kind = infer_kind(q.answer, q.domain, q.input)
        is_correct = grade(q.answer, pred, kind)

        if is_correct:
            correct += 1

        row = {
            "id": q.questionID,
            "domain": q.domain,
            "input": q.input,
            "expected": q.answer,
            "prediction": pred,
            "correct": is_correct,
        }
        rows.append(row)

        if i % 50 == 0 or i == total:
            print(f"Processed {i}/{total} questions...")

    accuracy = correct / max(total, 1)
    print(f"\naccuracy: {correct}/{total}")

if __name__ == "__main__":
    evaluate_dev_data("cse476_final_project_dev_data.json", limit=50)