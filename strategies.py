from api_client import call_model_chat_completions
from prompts import (Question, model_foundation, parse_user_prompt, extract_final_answer, extract_final_answer_strict)
from collections import Counter
import re

def clean_candidate(ans: str) -> str:
    ans = re.sub(r"^FINAL ANSWER\s*:\s*", "", ans, flags=re.IGNORECASE).strip()
    ans = ans.strip("$ ").strip()
    return ans

def is_plausible_math_answer(ans: str) -> bool:
    if not ans:
        return False
    ans = ans.strip()

    if re.fullmatch(r"[-+]?\d+(\.\d+)?", ans):
        return True

    if re.fullmatch(r"[-+]?\d+/\d+", ans):
        return True

    compact = ans.replace(" ", "")
    if re.fullmatch(r"\\frac\{[-+]?\d+\}\{\d+\}", compact):
        return True

    return False



class StrategyResult:
    def __init__(self, answer: str, raw_text: str, num_calls: int):
        self.answer = answer
        self.raw_text = raw_text
        self.num_calls = num_calls


class CoTStrategy:
    name = "CoT"

    def __init__(self, temperature: float = 0.0, max_tokens: int = 512):
        self.temperature = temperature
        self.max_tokens = max_tokens

    def solve(self, question: Question) -> StrategyResult:
        user_prompt = parse_user_prompt(question, inferenceAlg="CoT")

        resp = call_model_chat_completions(
            prompt=user_prompt,
            system=model_foundation,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        raw_text = (resp.get("text") or "").strip()
        answer = extract_final_answer(raw_text)

        return StrategyResult(
            answer=answer,
            raw_text=raw_text,
            num_calls=1
        )

def solve_cot_once(
    question: Question,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> tuple[str, str]:
    
    user_prompt = parse_user_prompt(question, inferenceAlg="CoT")

    resp = call_model_chat_completions(
        prompt=user_prompt,
        system=model_foundation,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    raw_text = (resp.get("text") or "").strip()
    answer = clean_candidate(extract_final_answer_strict(raw_text))
    if not answer:
        fallback = clean_candidate(extract_final_answer(raw_text))
        if (question.domain or "").lower() == "math":
            if is_plausible_math_answer(fallback):
                answer = fallback
        else:
            answer = fallback

    return answer, raw_text

def solve_self_consistency_cot(
    question: Question,
    num_samples: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 512,
    debug: bool = False,
) -> tuple[str, list[str]]:

    answers = []
    raw_outputs = []

    for _ in range(num_samples):
        ans, raw = solve_cot_once(
            question=question,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        raw_outputs.append(raw)

        if (question.domain or "").lower() == "math":
            if is_plausible_math_answer(ans):
                answers.append(ans)
        else:
            if ans:
                answers.append(ans)

    if debug:
        print("Parsed answers:", answers, flush=True)

    if not answers:
        if debug:
            print("No strict FINAL ANSWER candidates found.", flush=True)
        return "", raw_outputs

    counts = Counter(answers)
    voted_answer, top_count = max(counts.items(), key=lambda kv: kv[1])

    low_consensus = (
        top_count == 1 or
        (num_samples == 3 and top_count == 2)
    )

    if low_consensus:
        det_ans, det_raw = solve_cot_once(
            question=question,
            temperature=0.0,
            max_tokens=max_tokens,
        )
        raw_outputs.append(det_raw)
        if det_ans:
            counts[det_ans] += 1
            voted_answer, _ = max(counts.items(), key=lambda kv: kv[1])

    if debug:
        print("Vote counts:", dict(counts), flush=True)
        print("Voted answer:", voted_answer, flush=True)

    return voted_answer, raw_outputs

class SelfConsistencyCoTStrategy:
    name = "SelfConsistencyCoT"

    def __init__(
        self,
        num_samples: int = 5,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        self.num_samples = num_samples
        self.temperature = temperature
        self.max_tokens = max_tokens

    def solve(self, question: Question) -> StrategyResult:
        voted, raws = solve_self_consistency_cot(
            question=question,
            num_samples=self.num_samples,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        combined_raw = "\n\n".join(raws)

        return StrategyResult(
            answer=voted,
            raw_text=combined_raw,
            num_calls=self.num_samples,
        )
    
def build_self_critique_prompt(question: Question, candidate: str) -> str:
    return "\n".join([
        f"DOMAIN: {question.domain}",
        "QUESTION:",
        question.input,
        f"CANDIDATE ANSWER: {candidate}",
        "INSTRUCTIONS: Check the candidate answer. "
        "If it is wrong, compute the correct answer. "
        "Respond with EXACTLY one line:",
        "FINAL ANSWER: <your best answer>",
        "FINAL ANSWER:",
    ])


def solve_self_critique(
    question: Question,
    candidate: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
):
    prompt = build_self_critique_prompt(question, candidate)

    resp = call_model_chat_completions(
        prompt=prompt,
        system=model_foundation,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    raw = (resp.get("text") or "").strip()
    revised = extract_final_answer(raw)
    return revised, raw
