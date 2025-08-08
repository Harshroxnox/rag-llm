from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.genai import types
import requests
import pymupdf
from google import genai
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

client = genai.Client()

def getEmbedding(text):
  result = client.models.embed_content(
    model="gemini-embedding-001",
    contents= [text],
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
  )
  return result.embeddings[0].values

def embedPdf(pdf_url):
  #-----------------------------------------Donwload PDF------------------------------------------------
  pdf_path = "temp.pdf"

  print("📥 Downloading PDF...")
  response = requests.get(pdf_url)
  with open(pdf_path, "wb") as f:
    f.write(response.content)

  #-----------------------------------------Extract Text from PDF---------------------------------------
  print("📄 Extracting text...")
  doc = pymupdf.open(pdf_path) # open a document
  full_text = ""

  for page_num, page in enumerate(doc, start=1): # iterate the document pages
    text = page.get_text() # get plain text (is in UTF-8)
    if text.strip():
      # Optional cleanup
      text = text.replace("\n", " ").strip()
      full_text += f"Page {page_num}: {text}\n"

  #-----------------------------------------Split Text into Chunks---------------------------------------
  print("✂️ Splitting text into chunks...")
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,   # characters, not tokens
      chunk_overlap=200
  )
  chunks = splitter.split_text(full_text)

  print(f"Total chunks: {len(chunks)}")

  #-----------------------------------Generate Vector Embeddings of Chunks-------------------------------
  embedded_chunks = []
  result = client.models.embed_content(
    model="gemini-embedding-001",
    contents= chunks,
    config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
  )

  for i, chunk in enumerate(chunks):
    emb = result.embeddings[i].values
    embedded_chunks.append({
      "id":f"chunk-{i}", 
      "values": emb, 
      "metadata": {"text": chunk}
    })

  #-----------------------------------Insert Vector Embeddings in Pinecone DB----------------------------
  pc = Pinecone(api_key = PINECONE_API_KEY)
  index_name = "gemini-dense-py"

  # Check if index exists
  name_list = []
  for indx in pc.list_indexes():
    name_list.append(indx.name)

  if index_name in name_list:
    pc.delete_index(index_name)
    print(f"Deleted existing index '{index_name}'")

  pc.create_index(
    name=index_name,
    vector_type="dense",
    dimension=3072,
    metric="cosine",
    spec=ServerlessSpec(
      cloud="aws",
      region="us-east-1"
    ),
    deletion_protection="disabled",
    tags={
      "environment": "development"
    }
  )

  index = pc.Index(index_name)
  index.upsert(vectors=embedded_chunks)

# Example Usage 
# url = r"https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
# embedPdf(url)
