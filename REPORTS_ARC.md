# TreeRM on MiniARC 5x5: report

Branch `treerm-arc` (off `treerm-mcts-opt`, the optimized search). Questions: (1) how TreeRM does on MiniARC against
TRM and VARC at ~0.5-0.6M parameters; (2) whether the test-time training recipe of arXiv 2609.21181 (Embed-TTT:
freeze the backbone, fit the task embedding; then Embed-Full-TTT: freeze the embedding, fine-tune the backbone)
carries over to TreeRM; (3) whether freezing TreeRM's own heads (candidate generator, scorer, LPRM value head) helps
at test time or in regular training. Sections: 0 scoreboard, 1 setup and decisions, 2 baselines, 3 TTT, 4 head
freezing, 5 the open loop, 6 run log.

## 0. Scoreboard
Standard protocol (149 tasks, 4000 steps x batch 128, vote over 32 augmented copies), test-set pass@k:

| model | params | pass@1 | pass@2 | pass@5 | wall | peak GPU alloc | W&B run |
|---|---|---|---|---|---|---|---|
| TRM | 531k | 2.7% | 4.0% | 6.0% | 627 s | 343 MB | trm_std |
| VARC | 607k | 2.7% | 6.0% | 6.0% | 361 s | 105 MB | varc_std |
| TreeRM | 597k | 2.7% | 4.7% | 6.7% | 1277 s | 550 MB | treerm_std |

Held-out protocol (30 tasks never seen in pretraining) and TTT: section 3. Head freezing: section 4.

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
  `scripts/miniarc5/summarize.py`; wall-clock (the process alone, not the wait for the GPU lock) and peak RSS from `/usr/bin/time -v` in each run's `train.log`.

## 2. Baselines (standard protocol)

4000 steps x batch 128, eval on all 4,768 test rows (149 tasks x 32 copies), ARC vote across the copies. W&B runs
`trm_std`, `varc_std`, `treerm_std` in `ozeronatwanb/ARC-agi-testbed`.

| model | params | pass@1 @2k / @4k | pass@2 @2k / @4k | pass@5 @4k | exact (row) @4k | train token acc | wall | peak GPU alloc / RSS |
|---|---|---|---|---|---|---|---|---|
| TRM | 530,690 | 0.7% / 2.7% | 2.0% / 4.0% | 6.0% | 1.4% | 0.88 | 627 s | 343 MB / 1.88 GB |
| VARC | 606,768 | 2.0% / 2.7% | 2.7% / 6.0% | 6.0% | 1.1% | 0.80 | 361 s | 105 MB / 2.05 GB |
| TreeRM | 597,254 | 2.0% / 2.7% | 2.0% / 4.7% | 6.7% | 1.3% | – (GRAM head logs lm/kl/lprm losses) | 1277 s | 550 MB / 1.99 GB |

Reading: at this budget every model solves only a handful of the 149 tasks (pass@2 4-7%, i.e. 6-10 tasks; one task
is 0.67%), so the three are level within noise; the user's 10x10 EqR run reached pass@2 6% after 20k steps, so 5x5
at 4k steps is in the same regime. Token accuracy is high (0.7-0.9) while exact-grid accuracy is ~1%: the models
get most cells right and rarely a whole grid. TreeRM costs 2.0x TRM's wall-clock here (its eval searches on every
cycle of all 8 supervision steps over 4,768 rows, twice) and 1.6x its GPU memory. Seeds (phase 5) decide whether
any ordering among the three is real.

## 3. Test-time training (held-out protocol)

Pretraining on 119 tasks (`miniarc5-aug-31-ho30`), the 30 held-out tasks' test pairs as the eval split. The
in-training eval of these runs is the **no-TTT control**: the held-out tasks' puzzle identifiers were never
trained (their embeddings are the zero init), so the model sees each test input with an uninformative task
embedding.

| model (held-out pretraining) | no-TTT control: pass@1 / pass@2 @4k | token acc | train token acc | wall |
|---|---|---|---|---|
| TRM | 0 / 0 (of 30) | 0.65 | 0.84 | 767 s |
| VARC | 0 / 0 (of 30) | 0.60 | 0.78 | 277 s |
| TreeRM | 0 / 0 (of 30) | 0.64 | – | 817 s |

So without adaptation none of the three generalises to an unseen MiniARC task from its demos alone (as expected:
the demos are only ever seen through the task embedding, which is untrained for these tasks).

### TTT results (`ttt_mini`: 16 aux tasks, 20 epochs, batch 8, lr 3e-4 cosine; 160 steps per task)
Counts out of 30 held-out tasks (one task = 3.3 points); token acc = `all/accuracy` over the 30 x 16 test copies;
"train loss end" = mean over tasks of the last TTT step's lm loss (how well the demos were fit).

| model | variant | pass@1 | pass@2 | pass@5 | token acc | train loss end | s / task |
|---|---|---|---|---|---|---|---|
| TRM | no TTT (control) | 0 | 0 | 0 | 0.65 | – | – |
| TRM | Full-TTT (embedding + backbone) | 1 | 2 | 3 | 0.766 | 0.51 | 12.4 |
| TRM | Embed-TTT (embedding only) | 1 | 2 | 2 | 0.740 | 0.66 | 11.0 |
| TRM | Embed-Full-TTT (10 epochs embedding, then 10 backbone) | 1 | 1 | 1 | 0.743 | 0.64 | 11.2 |

TRM reading: Full and Embed solve the same two tasks (`find_the_most_frequent...`, `which_side_has_more...`);
Embed-Full keeps only the first. Fitting only the task embedding (SignSGD at 1e-2 on one 128-vector per aux copy)
recovers almost all of Full-TTT's gain at this scale, which is the paper's first claim (the embedding is what
carries the task). The Embed-Full ordering did not add anything for TRM here; one confound is that the backbone
phase runs on the decaying half of the cosine schedule (`lr_min_ratio 0`), so it trains at a much lower effective
lr than Full-TTT's backbone. Re-tried below with a flat lr if time allows.

| model | variant | pass@1 | pass@2 | pass@5 | token acc | train loss end | s / task |
|---|---|---|---|---|---|---|---|
| VARC | no TTT (control) | 0 | 0 | 0 | 0.60 | – | – |
| VARC | Full-TTT | 3 | 4 | 4 | 0.805 | 0.37 | 5.1 |
| VARC | Embed-TTT | 1 | 1 | 2 | 0.765 | 0.48 | 4.1 |
| VARC | Embed-Full-TTT | 1 | 3 | 4 | 0.788 | 0.42 | 4.6 |

VARC reading: the paper's model adapts best of the three (Full-TTT 4 of 30 at pass@2, fitting the demos to a
training loss of 0.37), but its embedding alone is not enough (1 task), unlike TRM's. Embed-Full recovers three of
Full's four tasks (`color_the_outer_edge`, `find_the_most_frequent`, `which_side_has_more`; Full also solves
`Change_color_of_object`) and adds none. So at this scale (0.6M params, 4k pretraining steps, 20 TTT epochs) the
paper's Embed-Full > Full ordering (63.8% vs 59.7% at 18M params) is not reproduced; the same cosine confound
applies. What does reproduce is the paper's mechanism claim: the task embedding carries most of the adaptation for
TRM, less so for VARC.

| model | variant | pass@1 | pass@2 | pass@5 | token acc | train loss end | s / task |
|---|---|---|---|---|---|---|---|
| TreeRM | no TTT (control) | 0 | 0 | 0 | 0.64 | – | – |
| TreeRM | Full-TTT | 2 | 3 | 4 | 0.754 | 0.52 | 17.0 |
| TreeRM | Embed-TTT | 0 | 0 | 0 | 0.731 | 0.64 | 16.5 |
| TreeRM | Embed-Full-TTT | 0 | 1 | 2 | 0.737 | 0.62 | 16.5 |
| TreeRM | Full-TTT, heads frozen (`candidate_`, `v_head`) | 2 | 3 | 3 | 0.754 | 0.51 | 16.6 |
| TreeRM | Embed-Full-TTT, heads frozen in the backbone phase | 0 | 0 | 1 | 0.736 | 0.62 | 16.4 |
| TreeRM | Embed-TTT, then heads only (backbone frozen) | 0 | 0 | 0 | 0.720 | 0.74 | 16.0 |

TreeRM reading, question 1 (does Embed-TTT carry over): **no.** With the backbone frozen the embedding alone solves
nothing (token accuracy rises 0.64 -> 0.73 but no grid is completed), and the embed-first ordering is worse than
joint fine-tuning (1 vs 3 at pass@2). Full-TTT puts TreeRM between TRM (2 at pass@2) and VARC (4), and it is the
only model that solves `1_3_5...` (`find_the_most_frequent` and `which_side_has_more` are solved by all three).
Why the embedding is not enough for TreeRM where it is for TRM: in TRM the task vector enters a single
deterministic recursion; in TreeRM the forward also runs a per-puzzle search whose candidates (flags: the TRM
update plus fixed directions, gated) and prior scorer were trained on the 119 pretraining tasks, so a new task
also needs the scorer/value heads and the backbone to move. Question 2 (heads): freezing the candidate heads and
the LPRM value head during Full-TTT changes nothing (2/3/3 vs 2/3/4, same token accuracy): those heads neither help
nor hurt adaptation at this budget, and TTT's gains come from the backbone (and the embedding). Training the heads
alone after the embedding (backbone frozen) gives nothing at all.
Caveats: 30 tasks (3.3 points each), one seed, 20 TTT epochs, a 0.6M-param model pretrained for 4k steps; the
cosine confound for the Embed-Full variants is addressed by the flat-lr reruns of 3b.

## 4. Head freezing in regular training

## 5. Open loop: ideas tried

## 6. Run log

- **Phase 0 smoke** (200 steps + one full eval over the 4,768-row test split, W&B disabled): TRM 530,690 params,
  56 s wall, peak RSS 1.85 GB; VARC 606,768 params, 19 s; TreeRM 597,254 params, training at 5.1 it/s at batch
  128 (the compiled C++ search). TreeRM's first eval crashed in the ARC evaluator: it breaks vote ties by
  `sigmoid(q_halt_logits)` and TreeRM (like GRAM) has no halt head. Fix: TreeRM reports the logit of its LPRM
  value as `q_halt_logits`, so the tie-break uses the model's own correctness estimate (VARC reports a constant
  0 there, so its tie-break is arbitrary). 2-task TTT smokes on the TRM checkpoint ran end to end through the new
  MiniARC task loader (Full and Embed variants).
  The three `smoke_*` 200-step runs are also the first sanity check of `freeze_param_substrings` etc. on real data.
- **TTT trap found by the smoke.** Every TTT variant on the 2-task smokes returned bit-identical metrics, and a
  probe (1 vs 30 epochs, lr 1e-2, embeddings-only) confirmed the eval never changed: the training loop ran zero
  steps. `PuzzleDataset` fills a batch with whole puzzles and drops under-filled tails, and 3 aux tasks x 3 demos
  never fill a batch of 16. The recipe's 16 aux tasks x ~3.5 demos would have trained (2 batches per epoch, a third
  of the demos dropped every epoch), so `ttt_mini` moves to batch 8 (5-6 steps per epoch, as VARC's recipe), and
  `evaluate_ttt.py` now raises if a task trains for 0 steps instead of silently evaluating the untouched checkpoint.
  A second trap compounded the first: `evaluate_ttt.py` resumes by suffix (a task whose `tasks/<name>.json` exists
  is skipped), so re-running a suffix after the fix re-aggregated the stale zero-step results and looked like the
  fix had not worked. Verified by a direct run with cleared result dirs: 30 epochs at lr 1e-2 take a task's eval
  `lm_loss` from 2.106 (1 epoch) to 1.510, with the training loss 2.07 -> 1.30 printed per task (`TTT run 0: 30
  steps, train lm_loss ...`, a new log line). Rule for this loop: never reuse a TTT suffix.
