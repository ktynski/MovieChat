import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import shutil
import json

def apply_delta(base_model_identifier, target_model_path, delta_repo_url):
    # Create a directory for the base model
    base_model_dir = "/content/llama-7b-hf-v0"
    
    # Remove the existing directory if it exists
    if os.path.exists(base_model_dir):
        os.system(f"rm -rf {base_model_dir}")

    # Clone the base model repository using Git
    os.system(f"git clone {base_model_identifier} {base_model_dir}")


    # Create a directory for the delta model
    delta_model_dir = "/content/vicuna-7b-delta-v0"
    
    # Remove the existing directory if it exists
    if os.path.exists(delta_model_dir):
        os.system(f"rm -rf {delta_model_dir}")

    # Clone the delta repository using Git
    os.system(f"git clone {delta_repo_url} {delta_model_dir}")

    print(f"Loading the base model from {base_model_dir}")
    base = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)


    print(f"Loading the delta from {delta_model_dir}")
    delta = AutoModelForCausalLM.from_pretrained(delta_model_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True)


    DEFAULT_PAD_TOKEN = "[PAD]"
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_dir, use_fast=False)
    num_new_tokens = base_tokenizer.add_special_tokens(dict(pad_token=DEFAULT_PAD_TOKEN))

    base.resize_token_embeddings(len(base_tokenizer))
    input_embeddings = base.get_input_embeddings().weight.data
    output_embeddings = base.get_output_embeddings().weight.data
    input_embeddings[-num_new_tokens:] = 0
    output_embeddings[-num_new_tokens:] = 0

    print("Applying the delta")
    for name, param in tqdm(base.state_dict().items(), desc="Applying delta"):
        assert name in delta.state_dict()
        param.data += delta.state_dict()[name]

    print(f"Saving the target model to {target_model_path}")
    base.save_pretrained(target_model_path)
    base_tokenizer.save_pretrained(target_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-identifier", type=str, required=True, help="Identifier for the base model on Hugging Face (e.g., 'decapoda-research/llama-13b-hf')")
    parser.add_argument("--target-model-path", type=str, required=True, help="Path to save the target model")
    parser.add_argument("--delta-repo-url", type=str, required=True, help="URL of the delta model repository on Hugging Face (e.g., 'https://huggingface.co/lmsys/vicuna-7b-delta-v0')")

    args = parser.parse_args()
    apply_delta(args.base_model_identifier, args.target_model_path, args.delta_repo_url)
