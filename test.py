from llama_index import VectorStoreIndex, SimpleVectorStore
from sentence_transformers import SentenceTransformer

index = VectorStoreIndex(vector_store=vector_store)

# Create a query engine
query_engine = index.as_query_engine()

# Function to retrieve relevant sections
def retrieve_relevant_sections(query, k=5):
    results = query_engine.query(query)
    return [result.text for result in results[:k]]

# Function to generate an answer using the local model
def generate_answer(query, model, tokenizer):
    relevant_sections = retrieve_relevant_sections(query)
    context = " ".join(relevant_sections)
    inputs = tokenizer.encode_plus(query + " " + context, return_tensors='pt')
    outputs = model.generate(**inputs)
    answer = tokenizer.decode(outputs[0])
    return answer

# Example query
query = "Which code includes Significance of form for the validity of transactions? In which article, book and chapter is it written?"
answer = generate_answer(query)

print("Query was:", query)
print("Answer was:", answer)
