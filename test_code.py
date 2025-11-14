import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    print("❌ API key not found in .env file.")
    exit()

genai.configure(api_key=API_KEY)

try:
    model = genai.GenerativeModel("gemini-1.5-flash-latest")
    response = model.generate_content("Hello Gemini! Please tell me what model and plan I'm using.")
    print("✅ API Connected Successfully!")
    print("Response:", response.text)
except Exception as e:
    print("❌ Error:", e)
