from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

llm = ChatGoogleGenerativeAI(model='gemini-3.6-flash')
response = llm.invoke('Say hello in one short sentence')
print('Chat response:', response.content)

embeddings = GoogleGenerativeAIEmbeddings(model='models/gemini-embedding-001')
vector = embeddings.embed_query('test sentence')
print('Embedding length:', len(vector))