import os
token = os.getenv("HF_TOKEN")
from huggingface_hub import HfApi

api = HfApi()

# Carica tutti i file dentro checkpoint-750
folder = "GRPO-llama/checkpoint-750"
repo_id = "saracandu/llama-3.1-8b-rebus-solver-adapters-grpo"

for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)
    api.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=filename,
        repo_id=repo_id,
        token=token,
        commit_message="Upload checkpoint 750"
    )
