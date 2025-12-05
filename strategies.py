from api_client import call_model_chat_completions
from prompts import (Question, model_foundation, parse_user_prompt, extract_final_answer)
from collections import Counter


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
    answer = extract_final_answer(raw_text)
    return answer, raw_text


def solve_self_consistency_cot(
    question: Question,
    num_samples: int = 5,
    temperature: float = 0.7,
    max_tokens: int = 512,
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
        if ans:
            answers.append(ans)

    if not answers:
        return "", raw_outputs

    counts = Counter(answers)
    voted_answer, _ = max(counts.items(), key=lambda kv: kv[1])

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