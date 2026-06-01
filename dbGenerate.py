import ollama

EMBEDDING_MODEL = 'embeddinggemma'

VECTOR_DB = []

with open('db.txt', 'r') as file:
  dataset = file.readlines()
  print(f"Loaded {len(dataset)} lines from db.txt")


def add_chunk_to_database(chunk):
  embedding = ollama.embed(model=EMBEDDING_MODEL, input=chunk)['embeddings'][0]
  VECTOR_DB.append((chunk, embedding))

for i, chunk in enumerate(dataset):
  add_chunk_to_database(chunk)
  print(f'Added chunk {i+1}/{len(dataset)} to the database')

with open('vector_db.txt', 'w') as file:
  for chunk, embedding in VECTOR_DB:
    file.write(f"{chunk.strip()}|{embedding}\n")
