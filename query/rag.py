from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from ingestPdf import getEmbedding
from google import genai
import os

# Loads the environment variable
load_dotenv()

PINCONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECOST_HOST = os.getenv("PINECOST_HOST")

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()
pc = Pinecone(api_key = PINCONE_API_KEY)


def query(question):
  question_vector = getEmbedding(question)

  index = pc.Index(host=PINECOST_HOST)

  resp = index.query(
    namespace="__default__",
    vector=question_vector,
    top_k=3,
    include_metadata=True,
    include_values=False
  )

  context = "\n\n".join([m.metadata.get("text", "") for m in resp.matches])

  prompt = f"""
  You are an assistant that answers questions based only on the provided context.
  If the answer is not found in the context, say "I could not find relevant information".
  Try to keep your answer precise and short.

  Context:
  {context}

  Question:
  {question}

  Answer:
  """

  response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents=prompt
  )

  return response.text

# Example Usage
# ans = query("What will happen to the unregistered students?")
# print(ans)
