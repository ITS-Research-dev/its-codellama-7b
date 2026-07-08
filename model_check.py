import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LMSTUDIO_HOST", "http://localhost:1234/v1"),
    api_key="lm-studio"
)

def test_model():
    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "codellama-7b-instruct"),
        messages=[
            {"role": "user", "content": "Write a Python function to reverse a string."}
        ],
        temperature=0
    )
    print(response.choices[0].message.content)

if __name__ == "__main__":
    test_model()