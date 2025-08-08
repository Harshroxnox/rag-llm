from dotenv import load_dotenv
from google import genai

# Loads the environment variable
load_dotenv()

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

# response = client.models.generate_content(
#   model="gemini-2.5-flash",
#   contents="Explain how AI works in a few words"
# )

def query_response(question):
  response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=question
  )
  return response.text

# print(response.text)