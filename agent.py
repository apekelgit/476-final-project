from prompts import (Question, model_foundation, parse_user_prompt, extract_final_answer, DEFAULT_INFERENCE_PIPELINE)
from api_client import call_model_chat_completions
from strategies import solve_self_consistency_cot, solve_cot_once


def solve_auto(
    question: Question,
    pipeline=None,
    num_samples: int = 5,
    sc_temperature: float = 0.7,
):

    pipe = pipeline or DEFAULT_INFERENCE_PIPELINE
    pipe = [p.lower() for p in pipe]

    if "self_consistency" in pipe:
        ans, raws = solve_self_consistency_cot(
            question=question,
            num_samples=num_samples,
            temperature=sc_temperature,
        )
        if ans:
            return ans, "\n\n".join(raws), num_samples

    if "cot" in pipe:
        ans, raw = solve_cot_once(question, temperature=0.0)
        return ans, raw, 1

    user_prompt = parse_user_prompt(question, inferenceAlg="")
    resp = call_model_chat_completions(
        prompt=user_prompt,
        system=model_foundation,
        temperature=0.0,
        max_tokens=256,
    )
    raw = (resp.get("text") or "").strip()
    ans = extract_final_answer(raw)
    return ans, raw, 1
