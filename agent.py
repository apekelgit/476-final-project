from prompts import (Question, model_foundation, parse_user_prompt, extract_final_answer, DEFAULT_INFERENCE_PIPELINE)
from api_client import call_model_chat_completions
from strategies import solve_self_consistency_cot, solve_cot_once, solve_self_critique


def solve_auto(
    question: Question,
    pipeline=None,
    num_samples: int = 1,
    sc_temperature: float = 0.7,
    debug: bool = False,
):

    pipe = pipeline or DEFAULT_INFERENCE_PIPELINE
    pipe = [p.lower() for p in pipe]

    domain = (question.domain or "").lower()
    temp = sc_temperature
    if domain == "math":
        temp = min(sc_temperature, 0.4)

    if "self_consistency" in pipe:
        ans, raws = solve_self_consistency_cot(
            question=question,
            num_samples=num_samples,
            temperature=temp,
            debug=debug,
        )
        if ans:
            if "self_critique" in pipe:
                revised, crit_raw = solve_self_critique(question, ans)
                if revised:
                    return revised, "\n\n".join(raws + [crit_raw]), len(raws) + 1

            return ans, "\n\n".join(raws), len(raws)

    if "cot" in pipe:
        ans, raw = solve_cot_once(question, temperature=0.0)
        return ans, raw, 1

    user_prompt = parse_user_prompt(question, inferenceAlg="")
    resp = call_model_chat_completions(
        prompt=user_prompt,
        system=model_foundation,
        temperature=0.4,
        max_tokens=256,
    )
    raw = (resp.get("text") or "").strip()
    ans = extract_final_answer(raw)
    return ans, raw, 1

