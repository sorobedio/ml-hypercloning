import os
import argparse
from typing import Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hypercloning import cloneModel


# ---------- utils: tie detection / (un)tying ----------

def _is_runtime_tied(model) -> bool:
    """Ground-truth check: are input & output embeddings the SAME tensor/storage?"""
    in_emb = model.get_input_embeddings()
    out_emb = model.get_output_embeddings()
    if in_emb is None or out_emb is None or not hasattr(out_emb, "weight"):
        return False
    iw, ow = in_emb.weight, out_emb.weight
    if iw is ow:
        return True
    try:
        return iw.untyped_storage().data_ptr() == ow.untyped_storage().data_ptr()
    except Exception:
        return False


def _get_io_weights(model) -> Tuple[torch.nn.Parameter, torch.nn.Parameter]:
    in_w = model.get_input_embeddings().weight
    out_w = model.get_output_embeddings().weight
    return out_w, in_w


def _find_param_name(model: torch.nn.Module, target: torch.nn.Parameter) -> Optional[str]:
    for n, p in model.named_parameters():
        if p is target:
            return n
    return None


def _dynamic_tie(model) -> None:
    """Declare actual parameter names for tying and apply."""
    out_w, in_w = _get_io_weights(model)
    out_name = _find_param_name(model, out_w)
    in_name  = _find_param_name(model, in_w)
    if out_name is None or in_name is None:
        raise RuntimeError("Could not resolve parameter names for dynamic tying.")
    model._dynamic_tied_weights_keys = [(out_name, in_name)]
    model.tie_weights()


def _ensure_untied(model) -> None:
    """Break sharing by cloning output head parameter; mark config."""
    out_w, _ = _get_io_weights(model)
    out_mod = model.get_output_embeddings()
    out_mod.weight = torch.nn.Parameter(out_w.detach().clone())
    if hasattr(model, "config"):
        model.config.tie_word_embeddings = False


def _try_force_tie(model) -> bool:
    """
    Try standard tie, then dynamic tie. Returns True if tied after attempts.
    Raises if shapes can't be tied.
    """
    out_w, in_w = _get_io_weights(model)
    if out_w.shape != in_w.shape:
        raise RuntimeError(
            f"Cannot tie: lm_head {tuple(out_w.shape)} vs embed {tuple(in_w.shape)}."
        )
    # 1) standard
    model.tie_weights()
    if _is_runtime_tied(model):
        return True
    # 2) dynamic
    _dynamic_tie(model)
    return _is_runtime_tied(model)


def _print_param_summary(model, tag: str) -> None:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg = getattr(model, "config", None)
    print(f"\n[{tag}] Parameters")
    print(f"  total     : {total:,}")
    print(f"  trainable : {trainable:,}")
    if cfg is not None:
        print(f"  tie_word_embeddings : {getattr(cfg, 'tie_word_embeddings', None)}")
        print(f"  vocab_size          : {getattr(cfg, 'vocab_size', None)}")
        print(f"  hidden_size         : {getattr(cfg, 'hidden_size', getattr(cfg, 'd_model', None))}")
        print(f"  embedding_dim       : {getattr(cfg, 'word_embed_proj_dim', getattr(cfg, 'hidden_size', None))}")
        print(f"  num_hidden_layers   : {getattr(cfg, 'num_hidden_layers', None)}")


# ---------- main script ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_id", type=str, default="facebook/opt-350m",
                        help="HF id or local path for the source model.")
    parser.add_argument("--out_dir", type=str, default="../Basemodels/opt-350m-hypercloned-x2emb-x2ffn",
                        help="Directory to save the cloned model.")
    parser.add_argument("--emb_mult", type=int, default=2,
                        help="Embedding dimension multiplier for cloning.")
    parser.add_argument("--ffn_mult", type=int, default=2,
                        help="FFN (up_project) multiplier for cloning.")
    parser.add_argument("--unsafe_fallback", action="store_true",
                        help="If save_pretrained fails due to unknown ties, retry with safe_serialization=False.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load source + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.source_id, use_fast=True)
    source = AutoModelForCausalLM.from_pretrained(args.source_id)

    # Determine source tying (runtime first, config as hint)
    source_runtime_tied = _is_runtime_tied(source)
    source_config_tie   = bool(getattr(source.config, "tie_word_embeddings", False))
    source_should_tie   = source_runtime_tied or source_config_tie

    # Clone / expand
    target = cloneModel(
        source,
        embedding_dim_multiplier=args.emb_mult,
        up_project_multiplier=args.ffn_mult
    ).eval()

    # Apply policy: if source was tied, try to tie target; else ensure untied
    if source_should_tie:
        try:
            tied = _try_force_tie(target)
            if not tied:
                # Shouldn't happen, but guard anyway
                print("⚠️ Could not establish tying; leaving untied.")
                _ensure_untied(target)
            else:
                if hasattr(target, "config"):
                    target.config.tie_word_embeddings = True
        except Exception as e:
            print(f"⚠️ Tying failed ({e}). Falling back to untied.")
            _ensure_untied(target)
    else:
        _ensure_untied(target)

    # Move to CPU to free GPU memory
    target.to("cpu")

    # Save model (prefer safetensors)
    try:
        target.save_pretrained(args.out_dir, safe_serialization=True)
    except RuntimeError as e:
        if "shared tensors" in str(e) and args.unsafe_fallback:
            print(f"⚠️ Safe save failed ({e}). Falling back to safe_serialization=False.")
            target.save_pretrained(args.out_dir, safe_serialization=False)
        else:
            raise

    # Save tokenizer
    tokenizer.save_pretrained(args.out_dir)
    print(f"✅ Saved cloned model and tokenizer to: {args.out_dir}")

    # Summaries and reload sanity check
    _print_param_summary(target, "CLONED (in-memory)")
    reloaded = AutoModelForCausalLM.from_pretrained(args.out_dir)
    _print_param_summary(reloaded, "RELOADED (from disk)")
    print("🔁 Reload test succeeded.")


if __name__ == "__main__":
    main()
