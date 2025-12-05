import re

model_foundation = ("You are a careful reasoning assistant. You must follow the user instructions exactly.")

class Question:
    def __init__(self, questionID: str, domain: str, input: str, answer: str):
        self.questionID = questionID
        self.domain = domain
        self.input = input
        self.answer = answer

def parse_user_prompt(question: Question, inferenceAlg: str) -> str:
    prompt = []
    prompt.append(f"DOMAIN: {question.domain}\n")
    prompt.append(f"QUESTION:")
    prompt.append(f"{question.input}\n")

    if inferenceAlg:
        prompt.append(f"INFERENCE ALGORITHM: {inferenceAlg}\n")

    if inferenceAlg == "CoT":
        prompt.append("INSTRUCTIONS: Think through the problem step-by-step before providing the final answer. You may include intermediate reasoning."
                      "After solving the problem, ouput one line that starts with 'FINAL ANSWER:' followed by the final answer. DO NOT INCLUDE ANYTHING AFTER THE FINAL ANSWER LINE.")
        
    else:
        prompt.append("INSTRUCTIONS: Provide ONLY the final answer to the question above. Do NOT include any explanations or reasoning steps.\n")

    prompt.append("FINAL ANSWER:")

    return "\n".join(prompt)

def extract_final_answer(model_response: str) -> str:
    lines = [line for line in model_response.splitlines() if line.strip() != ""]
    if not lines:
        return ""
    
    final_response_line = lines[-1]

    match = re.search(r"FINAL ANSWER:\s*(.*)", final_response_line)
    if match:
        return match.group(1).strip()
    return final_response_line.strip()