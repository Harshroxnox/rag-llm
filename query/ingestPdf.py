from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
import voyageai
import pymupdf
import os
import time

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECOST_HOST = os.getenv("PINECOST_HOST")

vo = voyageai.Client()

def getEmbedding(text):
  result = vo.embed([text], model="voyage-3-large", input_type="document")
  return result.embeddings[0]

def createIndex():
  pc = Pinecone(api_key = PINECONE_API_KEY)
  index_name = "voyage-dense-py"

  pc.create_index(
    name=index_name,
    vector_type="dense",
    dimension=1024,
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

def embedPdf(pdf_url):
  start = time.time()
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
  end = time.time()
  print(f"Till Chunking: {end - start}s")

  #-----------------------------------Generate Vector Embeddings of Chunks-------------------------------
  start = time.time()
  embedded_chunks = []

  batch_size = 100
  result = []
  for i in range(0, len(chunks), batch_size):
    result += vo.embed(
      chunks[i:i + batch_size], model="voyage-3.5", input_type="document"
    ).embeddings

  for i, chunk in enumerate(chunks):
    emb = result[i]
    embedded_chunks.append({
      "id":f"chunk-{i}", 
      "values": emb, 
      "metadata": {"text": chunk, "delete": "true" }
    })

  end = time.time()
  print(f"Generating Vector Embeddings: {end - start}s")

  #-----------------------------------Insert Vector Embeddings in Pinecone DB----------------------------
  start = time.time()
  pc = Pinecone(api_key = PINECONE_API_KEY)
  index = pc.Index(host=PINECOST_HOST)

  index.delete(
    filter={
      "delete": {"$eq": "true"}
    },
    namespace="__default__" 
  )
  
  index.upsert(vectors=embedded_chunks)

  end = time.time()
  print(f"Inserting Vectors: {end - start}s")

# Example Usage 
# url = r"https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
# embedPdf(url)
