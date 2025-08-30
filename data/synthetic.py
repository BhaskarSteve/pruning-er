from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")

resp = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    messages=[{"role": "user",
               "content": "Factor 12345. Explain your steps, final in \\boxed{}."}],
    temperature=0.6, top_p=0.95, max_tokens=8192,
    stream=False
)

msg = resp.choices[0].message
print(msg)
# print("--- reasoning ---")
# print(getattr(msg, "reasoning_content", None))
# print("--- final ---")
# print(getattr(msg, "content", None))
# print("---")
# print("---")