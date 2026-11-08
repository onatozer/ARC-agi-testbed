"""The official VARC checkpoint (lillian039/VARC, e.g. huggingface VisionARC/offline_train_ViT checkpoint_final.pt) as a
weights file for our models/varc.py, loadable with pretrain.py's +load_checkpoint=<out> +load_weights_only=true.
    python scripts/convert_varc_checkpoint.py <official .pt> <out .pt>

Two differences besides names:
  - colour ids: theirs 0-9 colours, 10 canvas background, 11 output-size border; ours 0 PAD (background), 1 EOS
    (border), 2-11 colours, so the colour embedding's rows and the head's classes are reordered;
  - 2D RoPE pairing: theirs rotates adjacent dims (2i, 2i+1), rows in the first half of a head and columns in the
    second; ours rotates i with i + head_dim/2 (rows in the first quarter, columns in the second). The same
    frequencies, so permuting q and k's dims within each head gives identical attention.
Their 400 task tokens become our table, so new tasks' rows start from their mean (load_model_state_dict's resize).
"""
import sys

import torch

OURS_TO_THEIRS_COLOR = [10, 11] + list(range(10))  # our id -> their id


def rope_perm(head_dim: int) -> torch.Tensor:
    """our q/k dim -> their dim within one head."""
    q = head_dim // 4  # rotated pairs per axis
    perm = torch.empty(head_dim, dtype=torch.long)
    for i in range(q):
        perm[i], perm[i + head_dim // 2] = 2 * i, 2 * i + 1  # row pairs
        perm[q + i], perm[q + i + head_dim // 2] = head_dim // 2 + 2 * i, head_dim // 2 + 2 * i + 1  # column pairs
    return perm


def convert(ck: dict) -> dict:
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model_state"].items()}
    c = ck["config"]
    dim, heads, colors = c["embed_dim"], c["num_heads"], OURS_TO_THEIRS_COLOR
    per_head = rope_perm(dim // heads)
    qk_perm = torch.cat([h * (dim // heads) + per_head for h in range(heads)])
    qkv_perm = torch.cat([qk_perm, dim + qk_perm, 2 * dim + torch.arange(dim)])  # q, k permuted; v as is
    patches = c["patch_size"] ** 2

    out = {
        "inner.pos_embed": sd["positional_embed"],
        "inner.color_embed.weight": sd["color_embed.weight"][colors],
        "inner.puzzle_emb.weights": sd["task_token_embed.weight"],
        "inner.patch_embed.weight": sd["patch_embed.proj.weight"],
        "inner.patch_embed.bias": sd["patch_embed.proj.bias"],
        "inner.norm.weight": sd["norm.weight"],
        "inner.norm.bias": sd["norm.bias"],
        "inner.head.weight": sd["head.weight"].view(patches, len(colors), dim)[:, colors].reshape(-1, dim),
        "inner.head.bias": sd["head.bias"].view(patches, len(colors))[:, colors].reshape(-1),
    }
    for i in range(c["depth"]):
        a, b = f"encoder.layers.{i}.", f"inner.layers.{i}."
        out[b + "attn.qkv.weight"] = sd[a + "self_attn.qkv.weight"][qkv_perm]
        out[b + "attn.qkv.bias"] = sd[a + "self_attn.qkv.bias"][qkv_perm]
        for theirs, ours in (("self_attn.proj", "attn.proj"), ("norm1", "norm1"), ("norm2", "norm2"), ("linear1", "mlp.0"), ("linear2", "mlp.3")):
            out[b + ours + ".weight"], out[b + ours + ".bias"] = sd[a + theirs + ".weight"], sd[a + theirs + ".bias"]
    return {"model." + k: v for k, v in out.items()}  # the loss head wraps the model as .model


if __name__ == "__main__":
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
    torch.save(convert(ck), sys.argv[2])
    print(f"{sys.argv[2]}: {len(convert(ck))} tensors")
