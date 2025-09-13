
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from hypercloning import cloneModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_id", type=str, default="facebook/opt-350m",
                        help="HF model id or local path for the source model.")
    parser.add_argument("--out_dir", type=str, default="../Basemodels/opt-350m-hypercloned-x2emb-x2ffn",
                        help="Directory to save the destination model.")
    parser.add_argument("--emb_mult", type=int, default=2,
                        help="Embedding dimension multiplier for cloning.")
    parser.add_argument("--ffn_mult", type=int, default=2,
                        help="FFN (up_project) multiplier for cloning.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load source model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.source_id, use_fast=True)
    source_model = AutoModelForCausalLM.from_pretrained(args.source_id)

    # Clone / expand the model
    destination_model = cloneModel(
        source_model,
        embedding_dim_multiplier=args.emb_mult,
        up_project_multiplier=args.ffn_mult
    )
    destination_model.eval()

    # (Optional) move to CPU before saving to free GPU memory
    destination_model.to("cpu")

    # Save model weights + config (will write config.json and model.safetensors)
    destination_model.save_pretrained(args.out_dir, safe_serialization=True)

    # Save tokenizer alongside (vocab.json/merges.txt or tokenizer.json, tokenizer_config.json, etc.)
    tokenizer.save_pretrained(args.out_dir)

    print(f"✅ Saved cloned model and tokenizer to: {args.out_dir}")

    # Quick sanity check (reload)
    md = AutoModelForCausalLM.from_pretrained(args.out_dir)
    _tok = AutoTokenizer.from_pretrained(args.out_dir)
    print(md)

    print("🔁 Reload test succeeded.")

if __name__ == "__main__":
    main()




