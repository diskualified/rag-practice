import ast
import ollama

EMBEDDING_MODEL = 'embeddinggemma'
LANGUAGE_MODEL = 'qwen3.5:4b'

VECTOR_DB = []

with open('vector_db.txt', 'r') as file:
  for line in file:
    chunk, embedding_str = line.strip().split('|')
    embedding = ast.literal_eval(embedding_str)
    VECTOR_DB.append((chunk, embedding))

def cosine_similarity(vec1, vec2):
  dot_product = sum(a * b for a, b in zip(vec1, vec2))
  magnitude1 = sum(a ** 2 for a in vec1) ** 0.5
  magnitude2 = sum(b ** 2 for b in vec2) ** 0.5
  if magnitude1 == 0 or magnitude2 == 0:
    return 0.0
  return dot_product / (magnitude1 * magnitude2)


def retrieve(query, top_n=3):
  query_embedding = ollama.embed(model=EMBEDDING_MODEL, input=query)['embeddings'][0]
  similarities = [(chunk, cosine_similarity(query_embedding, embedding)) for chunk, embedding in VECTOR_DB]
  similarities.sort(key=lambda x: x[1], reverse=True)
  return similarities[:top_n]


def main():
  input_query = input('Ask me a question (:q to quit): ')
  if input_query.lower() == ':q':
    return False

  retrieved_knowledge = retrieve(input_query)
  print('Retrieved knowledge:')
  for chunk, similarity in retrieved_knowledge:
    print(f' - (similarity: {similarity:.2f}) {chunk}')

  context = '\n'.join([f' - {chunk} ({similarity:.2f})' for chunk, similarity in retrieved_knowledge])
  instruction_prompt = f'''You are a helpful chatbot.
  Use only the following pieces of context to answer the question. Don't make up any new information:
  {context}
  '''

  stream = ollama.chat(
    model=LANGUAGE_MODEL,
    messages=[
      {'role': 'system', 'content': instruction_prompt},
      {'role': 'user', 'content': input_query},
    ],
    stream=True,
    think=True,
  )

  print('Chatbot response:')
  for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
  print()
  return True


if __name__ == '__main__':
  while main():
    pass
