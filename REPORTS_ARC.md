# TreeRM on MiniARC 5x5: report

Branch `treerm-arc` (off `treerm-mcts-opt`, the optimized search). Questions: (1) how TreeRM does on MiniARC against
TRM and VARC at ~0.5-0.6M parameters; (2) whether the test-time training recipe of arXiv 2609.21181 (Embed-TTT:
freeze the backbone, fit the task embedding; then Embed-Full-TTT: freeze the embedding, fine-tune the backbone)
carries over to TreeRM; (3) whether freezing TreeRM's own heads (candidate generator, scorer, LPRM value head) helps
at test time or in regular training. Sections: 0 scoreboard, 1 setup and decisions, 2 baselines, 3 TTT, 4 head
freezing, 5 the open loop, 6 run log.

## 0. Scoreboard
(filled as runs finish)

## 1. Setup and decisions

- **5x5, not 10x10.** Every MiniARC task is natively 5x5 (all 1,354 grids of the 149 tasks); the 10x10 canvas of
  the user's earlier runs only adds padding and translation augmentation. At 5x5 the sequence is 25 tokens plus one
  puzzle-embedding slot (`arch.puzzle_emb_len=1`; the TRM default of 16 slots would be 15 zero tokens).
- **Two builds, 31 augmentations** (`dataset/build_miniarc_dataset.py`, dihedral x colour at 5x5):
  `data/miniarc5-aug-31` (standard: every task's demos in training, its 32 test copies in the test split, 4,768
  test rows, the protocol of the user's W&B runs) and `data/miniarc5-aug-31-ho30` (new `heldout_tasks=30`: 30
  seeded tasks never in training, only their test pairs in the test split; the other 119 tasks contribute demos and
  test pair to training; `heldout_tasks.json` lists them). Why 31 and not 1000: the sampler draws one example per
  task per epoch, so in 4000 steps at batch 128 each augmented copy's embedding gets ~30 SignSGD updates at 31 augs
  but ~1 at 1000, where the model would learn to ignore the embedding and Embed-TTT would have nothing to fit.
- **Parameter parity**: TRM 527,618 / TreeRM 594,182 / VARC 606,768 dense params (hidden 128; VARC 4 layers, MLP
  256, canvas 10 since its patch size 2 needs an even side). The puzzle-embedding table (4,769 x 128) is a buffer.
- **One recipe for all**: AdamAtan2, lr 1e-3 (the TreeRM recipe's; falls back to 3e-4 if a smoke diverges),
  warmup 100, constant after, batch 128, 4000 steps, EMA 0.999, eval every 2000 steps on the full test split (no
  `max_eval_steps`: it would drop puzzles from the augmentation vote), seed 0. TreeRM: K=4 flag candidates, 8
  training / 16 eval simulations, value bootstrap 0.5, candidate heads at 0.1x lr, compiled C++ search.
  `cfg_pretrain_miniarc.yaml`'s `optimizer: muon` is not used here (VARC has no Muon weights; the TreeRM recipe
  is AdamAtan2). ConvSwiGLU: the file set `activation_function` at the top level, which the model never read
  (every earlier MiniARC run was SwiGLU); it is now under `arch:`; the smoke decides whether it stays.
- **TTT recipe** `config/eval/ttt_mini.yaml`: 16 aux tasks, 20 epochs (2 warmup), batch 16, lr 3e-4 cosine, no
  weight decay, no EMA, one run and view per task. Variants through `train_overrides`: Full (joint), Embed
  (`freeze_weights=true`), Embed-Full (`freeze_alternation_epochs=10`: in the odd block only the weight optimizers
  step, so the embedding is frozen), heads frozen (`freeze_param_substrings=[candidate_,v_head]`, a new generic
  option in `pretrain.py`: matching parameters get `requires_grad=False` and join no optimizer).
- **Protocol**: one GPU job at a time (flock), W&B online to `ozeronatwanb/ARC-agi-testbed` for real runs, memory
  (peak GPU allocated/reserved, host RSS) logged by `pretrain.py` as `train/*_mb` and `eval/*_mb` and read back by
  `scripts/miniarc5/summarize.py`; wall-clock and peak RSS from `/usr/bin/time -v` in each run's `train.log`.

## 2. Baselines (standard protocol)

## 3. Test-time training (held-out protocol)

## 4. Head freezing in regular training

## 5. Open loop: ideas tried

## 6. Run log
