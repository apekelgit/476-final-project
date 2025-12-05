import re

model_foundation = (
    "You are a careful reasoning assistant. You must follow the user instructions exactly."
)


class Question:
    def __init__(self, questionID: str, domain: str, input: str, answer: str):
        self.questionID = questionID
        self.domain = domain
        self.input = input
        self.answer = answer


def parse_user_prompt(question: Question, inferenceAlg: str) -> str:
    lines = []
    lines.append(f"DOMAIN: {question.domain}")
    lines.append("QUESTION:")
    lines.append(question.input)

    if inferenceAlg:
        lines.append(f"INFERENCE ALGORITHM: {inferenceAlg}")

    alg = (inferenceAlg or "").lower()

    if alg == "cot":
        lines.append(
            "INSTRUCTIONS: Think through the problem step-by-step. "
            "After solving, output EXACTLY one final line in the format:\n"
            "FINAL ANSWER: <your answer>\n"
            "Do NOT include anything after that line."
        )
    else:
        lines.append(
            "INSTRUCTIONS: Provide ONLY the final answer. "
            "No explanation. Use the format:\n"
            "FINAL ANSWER: <your answer>"
        )

    lines.append("FINAL ANSWER:")

    return "\n".join(lines)


def extract_final_answer(model_response: str) -> str:
    if not model_response:
        return ""

    text = model_response.strip()

    matches = re.findall(r"FINAL ANSWER\s*:\s*(.*)", text, flags=re.IGNORECASE)
    if matches:
        candidate = matches[-1].strip()
        candidate = candidate.strip("$ ").strip()
        return candidate

    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if nums:
        return nums[-1]

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    last = lines[-1]
    last = re.sub(r"^FINAL ANSWER\s*:\s*", "", last, flags=re.IGNORECASE)
    last = last.strip("$ ").strip()
    return last
