from sentence_transformers import SentenceTransformer
import pinecone

# Initialize Pinecone
pinecone.init(
    api_key="pcsk_E8U2x_AWW4bTD3dHLuX9cF134D5P6oXaauat67iMLd8fDQYSbhZd8PhGYTgwgSvp5bMaP",
    environment="YOUR_ENV"
)

index_name = "jarvis-index"

if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=384)

index = pinecone.Index(index_name)

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/sample_docs.txt", "r") as f:
    lines = f.readlines()

vectors = []
for i, line in enumerate(lines):
    embedding = model.encode(line).tolist()
    vectors.append((str(i), embedding, {"text": line.strip()}))

index.upsert(vectors)

print("Data successfully stored in Pinecone")
