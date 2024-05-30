from llama_index_client import TextNode
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF for PDF processing
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from llama_index.core.schema import BaseNode

# Qdrant server setup
URL = "localhost"
PORT = 6333
DIMENSION = 384
COLLECTION_NAME = "data"
METRIC_NAME = "COSINE"

client = QdrantClient(URL, port=PORT)
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize generative model (T5)
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

class TextSplitter:
    def __init__(self, chunk_size=512):
        self.chunk_size = chunk_size

    def split_text(self, text):
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

text_splitter = TextSplitter(chunk_size=512)

def embed_text(text):
    return embedding_model.encode(text)

# Load your document (e.g., PDF)
doc_path = 'data/matsne.doc'
doc = fitz.open(doc_path)

text_chunks = []
doc_idxs = []

for doc_idx, page in enumerate(doc):
    page_text = page.get_text("text")
    cur_text_chunks = text_splitter.split_text(page_text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([doc_idx] * len(cur_text_chunks))

nodes = []

for idx, text_chunk in enumerate(text_chunks):
    node_embedding = embed_text(text_chunk)
    node = TextNode(text=text_chunk)
    nodes.append(node)

# Add nodes to the vector store
vector_store.add(nodes)

def search_vector_store(query, top_k=5):
    query_embedding = embed_text(query)
    results = vector_store.search(query_embedding, top_k=top_k)
    return [result['payload']['text'] for result in results]

def generate_answer(query):
    relevant_docs = search_vector_store(query)
    context = " ".join(relevant_docs)
    
    input_text = f"Context: {context} Query: {query}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=150)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

query = "What is the significance of form for the validity of transactions?"
answer = generate_answer(query)
print(answer)
