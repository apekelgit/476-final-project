from api_client import call_model_chat_completions
from prompts import (Question, model_foundation, parse_user_prompt, extract_final_answer)


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
        )

        raw_text = (resp.get("text") or "").strip()
        answer = extract_final_answer(raw_text)

        return StrategyResult(
            answer=answer,
            raw_text=raw_text,
            num_calls=1
        )
