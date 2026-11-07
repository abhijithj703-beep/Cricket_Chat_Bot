import os
import time

import chromadb
from dotenv import load_dotenv
from google import genai
from google.genai.errors import ServerError
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# ============================================================
# 1. Load environment variables
# ============================================================

load_dotenv()


# ============================================================
# 2. Configuration
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

COLLECTION_NAME = "laws_of_cricket"

EMBEDDING_MODEL = "models/gemini-embedding-001"

GENERATION_MODEL = "gemini-3.6-flash"

MAX_GENERATION_RETRIES = 3


# ============================================================
# 3. Gemini client
# ============================================================

gemini_client = genai.Client(
    api_key=os.getenv("GOOGLE_API_KEY")
)


# ============================================================
# 4. Embedding model
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL
)


# ============================================================
# 5. ChromaDB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = chroma_client.get_collection(
    COLLECTION_NAME
)


# ============================================================
# 6. Ask question
# ============================================================

def ask_question(question: str, top_k: int = 3) -> dict:

    # --------------------------------------------------------
    # Step 1: Generate embedding for the user's question
    # --------------------------------------------------------

    query_embedding = embeddings.embed_query(question)


    # --------------------------------------------------------
    # Step 2: Retrieve relevant chunks from ChromaDB
    # --------------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    documents = results["documents"][0]


    # --------------------------------------------------------
    # Step 3: Combine retrieved chunks into context
    # --------------------------------------------------------

    context = "\n\n".join(documents)


    # --------------------------------------------------------
    # Step 4: Build RAG prompt
    # --------------------------------------------------------

    prompt = f"""
You are an assistant that answers questions about the Laws of Cricket.

Answer the question using ONLY the information provided in the context below.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided Laws of Cricket."

Do not make up information or use outside knowledge.

Context:
--------------------
{context}
--------------------

Question:
{question}

Answer:
"""


    # --------------------------------------------------------
    # Step 5: Generate answer with retry handling
    # --------------------------------------------------------

    response = None

    for attempt in range(1, MAX_GENERATION_RETRIES + 1):

        try:

            print(
                f"Generating answer "
                f"(attempt {attempt}/{MAX_GENERATION_RETRIES})..."
            )

            response = gemini_client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt
            )

            # Successful generation
            break


        except ServerError as e:

            print(
                f"Gemini server error on attempt "
                f"{attempt}/{MAX_GENERATION_RETRIES}: {e}"
            )

            # If this was the final attempt, return graceful failure
            if attempt == MAX_GENERATION_RETRIES:

                return {
                    "question": question,
                    "answer": None,
                    "retrieval_context": documents,
                    "status": "GENERATION_ERROR",
                    "error": str(e)
                }


            # ------------------------------------------------
            # Exponential backoff
            # ------------------------------------------------
            #
            # Attempt 1 fails → wait 5 seconds
            # Attempt 2 fails → wait 10 seconds
            #
            # Then try again.
            # ------------------------------------------------

            wait_time = 5 * (2 ** (attempt - 1))

            print(
                f"Retrying in {wait_time} seconds..."
            )

            time.sleep(wait_time)


    # ========================================================
    # 7. Extract generated answer
    # ========================================================

    answer = response.text


    # ========================================================
    # 8. Return structured result
    # ========================================================

    return {
        "question": question,
        "answer": answer,
        "retrieval_context": documents,
        "status": "SUCCESS",
        "error": None
    }


# ============================================================
# 9. Test the function directly
# ============================================================

if __name__ == "__main__":

    result = ask_question(
        "What happens if a batter hits the ball twice?"
    )


    print("\nQuestion:")
    print(result["question"])


    print("\nStatus:")
    print(result["status"])


    print("\nAnswer:")

    if result["answer"]:
        print(result["answer"])
    else:
        print("Generation failed.")


    print("\nRetrieved Context:")

    for i, context in enumerate(
        result["retrieval_context"]
    ):

        print(f"\n--- Context {i + 1} ---")
        print(context)


    if result["error"]:

        print("\nError:")
        print(result["error"])