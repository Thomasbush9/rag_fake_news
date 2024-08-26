import os
import torch
import ollama
from openai import OpenAI
from query_construction import query_maker
import json

# Initialize the Ollama API client
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3'
)


def load_vault_content(vault_file_path):
    with open(vault_file_path, "r", encoding='utf-8') as vault_file:
        return vault_file.readlines()


def classify_and_justify(query, relevant_context, ollama_model):
    # Prepare the prompt
    prompt = f"""
    Given the following context, determine whether the query is "true", "half true", or "false".
    After determining the class, provide a justification for your classification.

    Query: {query}

    Context: 
    {relevant_context}

    Please respond with the class ("true", "half true", or "false") and a brief justification for your choice.
    """

    # Call the model
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )

    response_text = response.choices[0].message.content.strip()
    return response_text.split("\n", 1)  # Returns the class and the justification


def get_relevant_context(query, embeddings, content, top_k=3):
    if embeddings.nelement() == 0:
        return ""

    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=query)['embedding']
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), embeddings)
    top_k = min(top_k, len(cos_scores))
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    relevant_context = "\n".join([content[idx].strip() for idx in top_indices])
    return relevant_context


# Test with the first query (query_0)
query_number = 0
query_name = f"query_{query_number}"
vault_file_path = os.path.join('vaults', f"{query_name}_vault.txt")
embeddings_file_path = f"{query_name}_embeddings_20240824_163028.pt"  # Assuming this is the correct file

# Load vault content and embeddings
vault_content = load_vault_content(vault_file_path)
vault_embeddings = torch.load(embeddings_file_path)

# Get the query from query_maker
queries = query_maker(n=5)  # Assuming this returns a dictionary of {number: query}
query = queries.get(query_number)

if query:
    # Retrieve relevant context
    relevant_context = get_relevant_context(query, vault_embeddings, vault_content)

    # Classify and justify
    classification, justification = classify_and_justify(query, relevant_context, 'llama3')

    # Print the result
    print(f"Query: {query}")
    print(f"Class: {classification}")
    print(f"Justification: {justification}")
else:
    print(f"Query {query_number} not found.")
