import openai

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="nokeyneeded",
)

response = client.chat.completions.create(
    model="phi3:medium",
    temperature=0.7,
    max_tokens=256,
    messages=[
        {"role": "system", "content": "You are a helpful assistant who is fluent in Russian."},
        {"role": "user", "content": "Напиши мне что-нибудь по-русски."},
    ],
)

print("Response:")
print(response.choices[0].message.content)