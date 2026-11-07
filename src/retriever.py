import chromadb
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

DB_PATH = "../chroma_db"

# Connect to existing ChromaDB
client = chromadb.PersistentClient(path=DB_PATH)

collection = client.get_collection("laws_of_cricket")

# Same embedding model used during ingestion
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


def retrieve_laws(question, top_k=3):

    # Convert user question into an embedding
    query_vector = embeddings.embed_query(question)

    # Search ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )

    return results


if __name__ == "__main__":

    question = "What happens if a batter hits the ball twice?"

    results = retrieve_laws(question)

    print("\nRetrieved Laws:\n")

    for i, document in enumerate(results["documents"][0]):

        metadata = results["metadatas"][0][i]

        print(f"Result {i + 1}")
        print(f"Law: {metadata['law_number']}")
        print(f"Title: {metadata['title']}")
        print(f"Content:\n{document}")
        print("-" * 80)