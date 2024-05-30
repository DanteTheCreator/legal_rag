from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
import os
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class NeuralSearcher:
    def __init__(self, collection_name):
        self.collection_name = collection_name
        # Initialize encoder model
        self.model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        # initialize Qdrant client
        self.qdrant_client = QdrantClient("http://localhost:6333")

    def search(self, text: str):
        # Convert text query into vector
        vector = self.model.encode(text).tolist()

        # Use `vector` for search for closest vectors in the collection
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            query_filter=None,  # If you don't want any filters for now
            limit=5,  # 5 the most closest results is enough
        )
        # `search_result` contains found vector ids with similarity scores along with the stored payload
        # In this function you are interested in payload only
        payloads = [hit.payload for hit in search_result]
        return payloads


Settings.embed_model = HuggingFaceEmbedding(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True
)

documents = SimpleDirectoryReader("data").load_data()

llm = OpenAI(api_base="http://localhost:1234/v1",
             api_key="lm-studio",
            #  model = 'lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf',
             )

Settings.llm = llm

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("SB იმ მოიწერა , რომ შეძლეს ლოგაუთის გამეორება შემდეგნაირად : პრაგმატიკის სლოტებში რამოდენიმე წუთი იყვნენ შემდეგ დააჭირეს home ღილაკს და დალოგაუთდა , გადავიდა ლოგინ ფეიჯზე , თუმცა რექვესთბი მაინც იგზავნება ბექიდან ანუ სესია არაა დასრულებული,კვლავ აქტიურია, ამის გამო ვერ ხედავენ თავიანთ მონიტორინგში მომხმარებლის ლოგაუთს. სავარაუდოა რომ ფრონტშია ხარვეზი .")

print(response)
