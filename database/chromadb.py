"""# *ChromaDB*"""

import chromadb

client = chromadb.Client()
collection = client.create_collection("sciq_supports")

collection.add(
    ids=[str(i) for i in range(0, 100)],  # IDs are just strings
    documents=dataset["support"][:100],
    metadatas=[{"type": "support"} for _ in range(0, 100)],
)
