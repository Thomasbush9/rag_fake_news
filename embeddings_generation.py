import os
import torch
from datetime import datetime
import ollama
from openai import OpenAI

# Configuration for the Ollama API client
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='llama3'
)

PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'
BLACK = "\033[0;30m"
RED = "\033[0;31m"
GREEN = "\033[0;32m"
BROWN = "\033[0;33m"
BLUE = "\033[0;34m"
PURPLE = "\033[0;35m"

LIGHT_GRAY = "\033[0;37m"
DARK_GRAY = "\033[1;30m"
LIGHT_RED = "\033[1;31m"
LIGHT_GREEN = "\033[1;32m"

LIGHT_BLUE = "\033[1;34m"
LIGHT_PURPLE = "\033[1;35m"
LIGHT_CYAN = "\033[1;36m"
LIGHT_WHITE = "\033[1;37m"

# Directory where vaults are stored
vaults_directory = "vaults"

# Iterate over each query-specific vault
for vault_name in os.listdir(vaults_directory):
    if vault_name.endswith("_vault.txt"):
        query_name = vault_name.replace("_vault.txt", "")
        vault_path = os.path.join(vaults_directory, vault_name)

        # Load the vault content
        print(f"{RED}Loading vault content for {query_name}...{RESET_COLOR}")
        with open(vault_path, "r", encoding='utf-8') as vault_file:
            vault_content = vault_file.readlines()

        # Generate embeddings for the vault content using Ollama
        print(f"{LIGHT_PURPLE}Generating embeddings for the vault content of {query_name}...{RESET_COLOR}")
        vault_embeddings = []
        for content in vault_content:
            response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
            vault_embeddings.append(response["embedding"])

        # Convert to tensor and save to file
        print(f"{RED}Converting embeddings to tensor for {query_name}...{RESET_COLOR}")
        vault_embeddings_tensor = torch.tensor(vault_embeddings)

        # Save the tensor to a file with a unique name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'{query_name}_embeddings_{timestamp}.pt'
        torch.save(vault_embeddings_tensor, filename)
        print(f"{RED}Embeddings for {query_name} have been saved to '{filename}'{RESET_COLOR}")

print(f"{GREEN}All vaults have been processed and embeddings saved.{RESET_COLOR}")
