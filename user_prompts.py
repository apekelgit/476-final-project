import re

model_foundation = ("You are a careful reasoning assistant. You must follow the user instructions exactly.")

class Question:
    questionID: str
    domain: str
    input: str
    answer: str

def parse_user_prompt(question: Question, inferenceAlg: str) -> str:
    prompt = []
    prompt.append(f"DOMAIN: {question.domain}\n")
    prompt.append(f"QUESTION:")
    prompt.append(f"{question.input}\n")

    if inferenceAlg:
        prompt.append(f"INFERENCE ALGORITHM: {inferenceAlg}\n")
    
    prompt.append("INSTRUCTIONS: Provide ONLY the final answer to the question above. Do NOT include any explanations or reasoning steps.\n")
    prompt.append("FINAL ANSWER:")

    return "\n".join(prompt)