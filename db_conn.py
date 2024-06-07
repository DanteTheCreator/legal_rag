from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer

#* Qdrant info
URL ="http://10.80.17.130"
DIMENSION = 384
COLLECTION_NAME = "legal_data" 
METRIC_NAME ="COSINE"

# %%
client = QdrantClient(URL,port=6333)
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)

# %%

llm = OpenAI(api_base="http://10.80.17.130:1234/v1",
             base_url="http://10.80.17.130:1234/v1",
             api_key="lm-studio",
             #  model = 'lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf',
             )
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5", trust_remote_code=True)
Settings.llm = llm
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
    llm = llm,
)
response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize",
)

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
)

def query_store(query):
    response = query_engine.query(query)
    return response.response

