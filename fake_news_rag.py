import os
import torch
import ollama
import csv
from openai import OpenAI
from query_construction import query_maker

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

def process_all_queries(vaults_directory, embeddings_directory, ollama_model, output_csv):
    # Load the queries using query_maker
    queries = query_maker(n=5)  # Adjust `n` based on the number of queries you have

    # Prepare CSV file to write results
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['query', 'class', 'justification']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Iterate over each embeddings file in the directory
        for embeddings_file in os.listdir(embeddings_directory):
            if embeddings_file.endswith(".pt"):
                query_name = embeddings_file.split("_embeddings")[0]
                vault_file_path = os.path.join(vaults_directory, f"{query_name}_vault.txt")
                embeddings_path = os.path.join(embeddings_directory, embeddings_file)

                if not os.path.exists(vault_file_path):
                    print(f"Vault content for {query_name} not found. Skipping...")
                    continue

                # Load vault content and embeddings
                vault_content = load_vault_content(vault_file_path)
                vault_embeddings = torch.load(embeddings_path)

                # Retrieve the corresponding query
                query_number = int(query_name.split("_")[1])  # Assuming the query number is in the filename
                query = queries.get(query_number)

                if query:
                    # Retrieve relevant context
                    relevant_context = get_relevant_context(query, vault_embeddings, vault_content)

                    # Classify and justify
                    classification, justification = classify_and_justify(query, relevant_context, ollama_model)

                    # Write the result to the CSV file
                    writer.writerow({
                        'query': query,
                        'class': classification.strip(),
                        'justification': justification.strip()
                    })

                    print(f"Processed {query_name}: {classification.strip()}")
                else:
                    print(f"Query {query_number} not found.")

if __name__ == "__main__":
    vaults_directory = "vaults"
    embeddings_directory = "."
    ollama_model = 'llama3'
    output_csv = 'query_classification_results.csv'

    # Process all queries and save the results to a CSV file
    process_all_queries(vaults_directory, embeddings_directory, ollama_model, output_csv)

    print(f"Results have been saved to '{output_csv}'")
