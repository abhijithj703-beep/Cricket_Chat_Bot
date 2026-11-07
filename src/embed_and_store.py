import json
import time
import chromadb
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.genai.errors import ClientError

load_dotenv()

INPUT_PATH = "../data/chunks.json"
DB_PATH = "../chroma_db"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection("laws_of_cricket")

for i, chunk in enumerate(chunks):
    if chunk["law_number"] is not None:
        chunk_id = f"law_{chunk['law_number']}"
    elif chunk["appendix"] is not None:
        chunk_id = f"appendix_{chunk['appendix']}"
    else:
        chunk_id = "preamble"

    existing = collection.get(ids=[chunk_id])
    if existing["ids"]:
        print(f"[{i + 1}/{len(chunks)}] Skipping {chunk_id} (already stored)")
        continue

    if not chunk["content"].strip():
        print(f"[{i + 1}/{len(chunks)}] Skipping {chunk_id} (empty content)")
        continue

    metadata = {
        "law_number": chunk["law_number"] if chunk["law_number"] is not None else -1,
        "appendix": chunk["appendix"] if chunk["appendix"] is not None else "",
        "title": chunk["title"]
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            vector = embeddings.embed_query(chunk["content"])
            break
        except ClientError as e:
            if "RESOURCE_EXHAUSTED" in str(e) and attempt < max_retries - 1:
                wait_time = 30
                print(f"Rate limited - waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
            else:
                raise

    collection.add(
        ids=[chunk_id],
        embeddings=[vector],
        documents=[chunk["content"]],
        metadatas=[metadata]
    )

    print(f"[{i + 1}/{len(chunks)}] Stored {chunk_id} - {chunk['title']}")

    time.sleep(2)