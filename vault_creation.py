import re
import os
from query_construction import query_maker
from sel_try import get_google_search_links, scrape_url_content

def save_text_to_vault(text, query_name, directory='vaults'):
    # Ensure the vaults directory exists
    os.makedirs(directory, exist_ok=True)

    # Clean up the text and split into sentences
    text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
    sentences = re.split(r'(?<=[.!?]) +', text)  # Split text into sentences based on punctuation

    chunks = []
    current_chunk = ''

    # Chunk the text to keep each chunk under 1000 characters
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 < 1000:
            current_chunk += sentence + ' '
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ' '

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Create a filename based on the query name
    vault_filename = os.path.join(directory, f"{query_name}_vault.txt")

    # Save the chunks to the vault file
    with open(vault_filename, 'a', encoding='utf-8') as vault_file:
        for chunk in chunks:
            vault_file.write(chunk + '\n')  # Add a newline between chunks for clarity

    print(f'Text content appended to {vault_filename}')

# Main script to process queries and save the articles to their respective vaults
queries = query_maker(n=5)  # Assuming this returns a dictionary of {number: query}

for query_number, statement in queries.items():
    k = 5
    query_name = f"query_{query_number}"  # Use the query number to create a unique vault name
    links = get_google_search_links(statement, k)

    for link in links:
        content = scrape_url_content(link)
        if content:
            # Combine the title and full content of the page
            full_text = content['title'] + "\n\n" + content['full_content']
            # Save each webpage's content, divided into chunks, in the vault
            save_text_to_vault(full_text, query_name=query_name)

print("All queries have been processed and articles saved to their respective vaults.")
