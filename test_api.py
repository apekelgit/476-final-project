from api_client import call_model_chat_completions

if __name__ == "__main__":
    test_prompt = "What is 17 + 28? Answer with just the number."
    result = call_model_chat_completions(test_prompt)
    print("OK:", result["ok"], "HTTP:", result["status"])
    print("Model Response:", (result["text"] or "").strip())
