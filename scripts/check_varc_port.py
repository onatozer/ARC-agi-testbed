"""The same canvases through the official ARCViT (lillian039/VARC) and our VARCInner with convert_varc_checkpoint.py's
weights: the logits must match. Run on CPU; needs timm and einops for the official code.
    python scripts/check_varc_port.py <official repo> <official checkpoint>"""
import pathlib
import sys

import torch

official, ckpt_path = sys.argv[1:3]
sys.path.insert(0, official)
from src.ARC_ViT import ARCViT  # noqa: E402

sys.path.insert(1, str(pathlib.Path(__file__).resolve().parents[1]))
from models.varc import VARCConfig, VARCInner  # noqa: E402
from scripts.convert_varc_checkpoint import OURS_TO_THEIRS_COLOR, convert  # noqa: E402

torch.manual_seed(0)
ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
c = ck["config"]
theirs = ARCViT(num_tasks=400, image_size=c["image_size"], num_colors=c["num_colors"], embed_dim=c["embed_dim"], depth=c["depth"],
                num_heads=c["num_heads"], mlp_dim=c["mlp_dim"], dropout=c["dropout"], num_task_tokens=1, patch_size=c["patch_size"])
theirs.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model_state"].items()})
theirs.eval()

cfg = VARCConfig(batch_size=2, seq_len=900, vocab_size=12, num_puzzle_identifiers=400, puzzle_emb_ndim=512, board_height=30, board_width=30,
                 forward_dtype="float32")
ours = VARCInner(cfg)
state = {k[len("model.inner."):]: v for k, v in convert(ck).items() if k != "model.inner.puzzle_emb.weights"}
missing, unexpected = ours.load_state_dict(state, strict=False)
assert not unexpected and all(k.startswith("puzzle_emb.") for k in missing), (missing, unexpected)
ours.eval()

# Two canvases in their encoding: colour grids at different offsets and sizes on background 10
S, B = c["image_size"], 2
canvas_t = torch.full((B, S, S), 10, dtype=torch.long)
mask = torch.zeros(B, S, S, dtype=torch.long)
for b, (h, w, y, x) in enumerate([(7, 11, 3, 20), (25, 18, 30, 5)]):
    canvas_t[b, y:y + h, x:x + w] = torch.randint(0, 10, (h, w))
    mask[b, y:y + h, x:x + w] = 1
task_ids = torch.tensor([3, 177])
with torch.no_grad():
    lt = theirs(canvas_t, task_ids, attention_mask=mask)
    lt = lt[0] if isinstance(lt, tuple) else lt  # (B, 12, S, S)

    # Ours: the same canvas in our ids and the same task tokens, through VARCInner.forward minus the raw-frame mapping
    to_ours = torch.empty(12, dtype=torch.long)
    to_ours[torch.tensor(OURS_TO_THEIRS_COLOR)] = torch.arange(12)
    canvas_o = to_ours[canvas_t]
    G, P = ours.grid, cfg.patch_size
    x = ours.patch_embed(ours.color_embed(canvas_o).permute(0, 3, 1, 2)).flatten(2).transpose(1, 2) + ours.pos_embed
    attend = (canvas_o != 0).view(B, G, P, G, P).any(4).any(2).view(B, G * G)
    x = torch.cat((ck["model_state"]["_orig_mod.task_token_embed.weight"][task_ids].view(B, 1, -1), x), dim=1)
    attend = torch.cat((attend.new_ones(B, 1), attend), dim=1)
    cos_sin = ours.rotary()
    for layer in ours.layers:
        x = layer(x, cos_sin, attend[:, None, None, :])
    lo = ours.head(ours.norm(x)[:, 1:]).view(B, G, G, P, P, -1).permute(0, 1, 3, 2, 4, 5).reshape(B, S, S, -1)

lt_ours = lt.permute(0, 2, 3, 1)[..., torch.tensor(OURS_TO_THEIRS_COLOR)]  # their classes in our order
inside = mask.bool()
err = (lo - lt_ours).abs()[inside]
agree = (lo.argmax(-1) == lt_ours.argmax(-1))[inside].float().mean()
print(f"max |logit diff| on grid pixels {err.max():.2e} (logit scale {lt_ours[inside].abs().max():.1f}), argmax agreement {agree:.4f}")
assert err.max() < 1e-3, "the ported model does not reproduce the official one"
print("PORT OK")
