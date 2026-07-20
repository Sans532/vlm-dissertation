import os
from dotenv import load_dotenv
load_dotenv()

print("Key found:", os.environ.get("GOOGLE_API_KEY"))

import google.genai as genai
client = genai.Client()
response = client.models.generate_content(model="gemini-3.1-flash-lite", contents="Say hello")
print(response.text)
