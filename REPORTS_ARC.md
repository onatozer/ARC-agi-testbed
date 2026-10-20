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
| TreeRM, `puzzle_emb_lr=3e-2` (section 5; 2 seeds: 5.4 / 7.4 / 10.7 and 3.4 / 4.7 / 6.0) | 597k | 4.4% | 6.0% | 8.4% | ~1150 s | 550 MB | treerm_emblr3, treerm_emblr3_s1 |
| TRM, `puzzle_emb_lr=3e-2` (1 seed) | 531k | 4.7% | 8.1% | 10.1% | 578 s | 343 MB | trm_emblr3 |
| TreeRM, `puzzle_emb_lr=3e-2`, **8000 steps** | 597k | 6.7% | 10.1% | 12.8% | 2308 s | 550 MB | treerm_emblr3_8k |
| TRM, `puzzle_emb_lr=3e-2`, **8000 steps** (2 seeds: 6.7 / 9.4 / 15.4 and 7.4 / 8.1 / 12.1) | 531k | 7.0% | 8.7% | 13.8% | ~1380 s | 343 MB | trm_emblr3_8k, trm_emblr3_8k_s1 |
| TreeRM, K=8 + `puzzle_emb_lr=3e-2`, 4000 steps | 598k | 5.4% | 8.1% | 12.1% | 1218 s | 1150 MB | treerm_k8_emblr3 |
| **TreeRM, K=8 + `puzzle_emb_lr=3e-2`, 8000 steps** (2 seeds: 7.4 / 13.4 / 17.5 and 4.0 / 8.7 / 12.8) | 598k | **5.7%** | **11.1%** | **15.2%** | ~2680 s | 1150 MB | treerm_k8_emblr3_8k, treerm_k8_emblr3_8k_s1 |

Best single run: K=8 flag candidates, task-embedding lr 3e-2, 8000 steps: 20 of 149 tasks at pass@2 (TRM at the
same budget and embedding lr: 14; the recipe TreeRM at 4000 steps: 7). But the 4000-step seed replicate of that
configuration gives 2.7 / 3.4 / 7.4 (seed 0: 5.4 / 8.1 / 12.1), so over two seeds K=8 + 3e-2 averages 4.0 / 5.7 /
9.8, no better than 3e-2 alone (4.4 / 6.0 / 8.4): K=8 is not a reliable gain at 4k. The 8k seed replicate lands at
4.0 / 8.7 / 12.8 (13 tasks at pass@2 against seed 0's 20), so the 8k configuration averages 5.7 / 11.1 / 15.2 over
two seeds: a spread of 7 tasks between seeds of the same configuration, as large as any model gap in this table.
TRM-8k's replicate gives 7.4 / 8.1 / 12.1, so over two seeds TRM-8k averages 7.0 / 8.7 / 13.8 against TreeRM K=8-8k's
5.7 / 11.1 / 15.2: TreeRM +2.4 points at pass@2 (3.6 tasks) and +1.4 at pass@5, TRM +1.3 at pass@1, with per-seed
spreads of 1-7 tasks. A gap that small and that inconsistent across k is not a gap at two seeds; what does hold is
that TRM reaches these numbers in half the wall-clock and a third of the GPU memory. The robust finding stays the
embedding lr.

Seed noise is 2-4 tasks (TreeRM seed 1: 1.3 / 2.0 / 2.7; TRM seed 1: 2.7 / 4.0 / 5.4), so the three architectures are
level at this budget; the one change that moved TreeRM across seeds is a faster task-embedding lr (the recipe's
1e-2 gives each augmented copy's embedding only ~30 updates in 4000 steps).

Held-out protocol (30 tasks never seen in pretraining; no model solves any without TTT), test-time training, tasks
solved at pass@2 out of 30 (section 3):

| model | Full-TTT | Embed-TTT (embedding only) | Embed-Full-TTT (paper's ordering) | heads frozen (TreeRM) |
|---|---|---|---|---|
| TRM | 2 | 2 | 1 | – |
| VARC | 4 | 1 | 3 | – |
| TreeRM | 3 | 0 | 1 | Full 3, Embed-Full 0, heads-only 0 |

Verdicts so far: TTT works for all three (0 -> 2-4 of 30); the paper's embed-first ordering does not beat joint
fine-tuning at this scale for any of them, and for TreeRM the task embedding alone is not enough (its search and
prior need the backbone to move); freezing TreeRM's heads at test time changes nothing, and training them together
with the embedding on a frozen backbone fits the demos but solves no more tasks (section 3c). Head freezing in
regular training (sections 4 and 4b): the trained heads can be frozen at no cost, every iterative freezing
schedule (heads/backbone or embeddings/weights alternation, from scratch or as a continuation) is worse than
training everything, because the backbone loses half its updates.

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

Seed replicates (phase 5): TRM seed 1 gives 2.7% / 4.0% / 5.4% (seed 0: 2.7 / 4.0 / 6.0, steady); TreeRM seed 1 reaches pass@1 1.3% / pass@2 2.0% / pass@5 2.7% at 4000 steps (seed 0:
2.7 / 4.7 / 6.7), i.e. seed noise is 2-4 tasks, as large as every gap in the table above. At this budget the three
architectures are indistinguishable on MiniARC 5x5.

Augmentation sensitivity: TreeRM on `miniarc5-aug-100-t31` (100 augmented copies per task in training, the test
split capped at the same 32 copies) gives pass@1 1.3% / pass@2 4.0% / pass@5 4.7% and a lower token accuracy
(0.672 vs 0.719): three times the copies with a third of the embedding updates each is no better, as the
updates-per-copy argument of section 1 predicted. 31 stays.

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

### 3b. Flat learning rate (`lr_min_ratio=1.0`) and longer Embed-TTT
The cosine schedule spans both blocks of Embed-Full, so its backbone phase ran at a decaying lr; rerun with a flat
3e-4 (and Full with the same flat lr for a fair pair), plus Embed-TTT for 40 epochs.

| model | variant | pass@1 | pass@2 | pass@5 | token acc | train loss end |
|---|---|---|---|---|---|---|
| TRM | Embed-Full-TTT, flat lr | 1 | 3 | 3 | 0.747 | 0.56 |
| TRM | Full-TTT, flat lr | 1 | 1 | 2 | 0.758 | 0.36 |
| TRM | Embed-TTT, 40 epochs | 1 | 1 | 3 | 0.766 | 0.42 |

| TreeRM | Embed-Full-TTT, flat lr | 0 | 1 | 2 | 0.736 | 0.55 |
| TreeRM | Full-TTT, flat lr | 1 | 2 | 3 | 0.748 | 0.37 |
| TreeRM | Embed-TTT, 40 epochs | 0 | 0 | 1 | 0.744 | 0.50 |

| VARC | Embed-Full-TTT, flat lr | 2 | 4 | 4 | 0.805 | 0.37 |
| VARC | Full-TTT, flat lr | 4 | 4 | 7 | 0.813 | 0.34 |
| VARC | Embed-TTT, 40 epochs | 1 | 1 | 1 | 0.770 | 0.48 |

VARC: the flat lr lifts Embed-Full from 1 / 3 / 4 to 2 / 4 / 4, but flat-lr Full-TTT is the best adaptation of the night
(4 / 4 / 7: 23% of the held-out tasks at pass@5), so for VARC joint fine-tuning keeps the edge; the embed-first ordering wins only for TRM here.

TreeRM: the flat lr changes nothing for Embed-Full (0 / 1 / 2 either way, Full-TTT 2 / 3 / 4), so for TreeRM the
schedule was not the reason the embed-first ordering lags; the embedding alone cannot steer its search.

TRM: the flat lr fixes Embed-Full (1 -> 3 at pass@2), while flat-lr Full fits the demos hardest (train loss 0.36)
and solves the fewest tasks (1): with the schedule held equal, the paper's embed-first ordering beats joint
fine-tuning for TRM by two tasks, and the mechanism the paper gives (fitting the embedding first keeps the backbone
from overfitting the few demos) is what the training losses show. Still one seed and 30 tasks.

### 3c. Which TTT variants train TreeRM's heads, and the joint embedding-plus-heads variant

A note on what each TTT variant does to the candidate generator, scorer and LPRM value head (`candidate_*`,
`v_head`), since the question came up: **Full-TTT trains them** (every parameter, at their 0.1x lr scale);
**Embed-TTT freezes them** by definition (`freeze_weights`: only the task embedding moves); **Embed-Full-TTT**
freezes them in the embedding block and trains them in the backbone block. The "heads frozen" rows of the table
above remove them from Full / Embed-Full. What was missing is the TreeRM-native analogue of Embed-TTT: the backbone
frozen and the embedding **and** the heads trained together, so the search can re-tune its prior and value to the
new task while the representation stays put. That variant (`embed_heads_e3`) and the three standard variants are
run on the held-out TreeRM trained with `puzzle_emb_lr=3e-2`, with the same embedding lr at test time
(`+train_overrides.puzzle_emb_lr=3e-2`), flat lr for the two-block variants.

| TreeRM (held-out, emb lr 3e-2) | variant | pass@1 | pass@2 | pass@5 | token acc | train loss end |
|---|---|---|---|---|---|---|
| | no-TTT control (in-training eval) | 0 | 0 | 0 | 0.65 | – |
| | Full-TTT (heads train) | 2 | 3 | 4 | 0.779 | 0.39 |
| | Embed-TTT (heads frozen) | 0 | 0 | 2 | 0.761 | 0.57 |
| | Embed-Full-TTT, flat lr | 1 | 1 | 2 | 0.753 | 0.46 |
| | **Embed+heads-TTT**: backbone frozen, embedding + heads train, flat lr | 1 | 1 | 2 | 0.763 | 0.40 |

Reading: the faster embedding lr does not change TreeRM's TTT ordering (Full 2 / 3 / 4, the three variants that
keep the backbone frozen for part or all of the run 0-1 at pass@2). The joint embedding-plus-heads variant answers
the question of whether the heads were what Embed-TTT was missing: letting them train brings the demo fit from
0.57 down to Full-TTT's 0.40, so the search heads can absorb a new task's demos, but the test outputs do not follow
(1 / 1 / 2 against Full's 2 / 3 / 4). For TreeRM the backbone is what has to move at test time; training the heads
with or without it is not the lever.

## 4. Head freezing in regular training

1000-step continuations of the standard TreeRM checkpoint (`+load_checkpoint`, weights only, warmup 50), same
recipe otherwise, one full eval at the end. Freezing untrained heads from scratch would leave the search with a
random prior and value, so the question is asked on a trained model. "alt" = `freeze_alternation_epochs=25`
(~100-step blocks: embeddings only, then weights only).

| continuation | pass@1 | pass@2 | pass@5 | exact (row) | token acc | train lm loss end | wall |
|---|---|---|---|---|---|---|---|
| checkpoint at 4000 steps (start) | 2.7% | 4.7% | 6.7% | 1.3% | 0.719 | 0.456 | – |
| control: everything trains | 3.4% | 5.4% | 8.1% | 1.1% | 0.722 | 0.432 | 528 s |
| heads frozen (`candidate_`, `v_head`: 66,822 values) | 1.3% | 4.7% | 9.4% | 1.4% | 0.738 | 0.415 | 431 s |
| alt: embeddings / weights alternation | 2.7% | 4.0% | 6.7% | 1.2% | 0.723 | 0.570 | 442 s |
| alt + heads frozen in the weight phase | 2.7% | 4.0% | 6.7% | 1.2% | 0.736 | 0.501 | 421 s |

Reading: all three numbers of the heads-frozen run sit within seed noise of the control (one task = 0.67%); the
heads can be frozen after pretraining without loss, which also says the candidate generator and value head had
converged by 4000 steps (they train at 0.1x lr anyway). The alternation (the paper's freezing idea applied to
training) does not help: both alternation runs end exactly where the checkpoint started (2.7 / 4.0-4.7 / 6.7) with
a higher training loss, because the backbone trains in only half of the steps; freezing the heads on top changes
nothing. **Question 2, training side: no effect either way.** The control that trains everything is the best
continuation (3.4 / 5.4 / 8.1), i.e. the model simply wanted more steps.

### 4b. Head alternation from scratch (`freeze_alternation_substrings`)

The continuations above only ever froze the heads of a trained model. To see the effect of freezing them
*iteratively throughout training*, `pretrain.py` now takes `freeze_alternation_substrings`: together with
`freeze_alternation_epochs` it alternates between training everything except the matched parameters and training
only them (gradients of the sitting-out set are dropped before the step, so their optimizer state does not move
either; `train/freeze_phase` logs the phase). From-scratch 4000-step runs at the `puzzle_emb_lr=3e-2` recipe,
control `treerm_emblr3` (5.4 / 7.4 / 10.7, seed 1: 3.4 / 4.7 / 6.0):

| run | schedule | pass@1 | pass@2 | pass@5 | token acc | train lm loss end | wall |
|---|---|---|---|---|---|---|---|
| `treerm_emblr3` (control) | everything, every step | 5.4% | 7.4% | 10.7% | 0.738 | 0.40 | 1124 s |
| `treerm_e3_althd` | 100-step blocks: backbone+embeddings / heads only | 2.0% | 4.7% | 7.4% | 0.721 | 0.80 | 1178 s |
| `treerm_e3_altemb` | 100-step blocks: embeddings / weights (the paper's schedule) | 1.3% | 2.0% | 2.0% | 0.708 | 0.77 | 1262 s |
| `treerm_e3_2k` -> `treerm_e3_2k_ctrl` | 2000 steps (1.3 / 2.7 / 5.4), then 2000 more with everything | 2.7% | 4.7% | 8.1% | 0.733 | 0.34 | 555 + 895 s |
| `treerm_e3_2k` -> `treerm_e3_2k_heads` | 2000 steps (1.3 / 2.7 / 5.4), then 2000 more with the heads frozen | 5.4% | 6.7% | 10.1% | 0.732 | 0.33 | 555 + 853 s |

Reading. Both alternations lose, and for the same reason the embeddings/weights alternation lost in section 4: the
backbone trains in only half of the steps (final lm loss 0.77-0.80 against 0.40), and the heads-only blocks cannot
lower the lm loss at all, so alternating is a 2000-step run with extra cost. Iterative head freezing throughout
training is therefore the wrong shape for TreeRM: the heads are cheap (67k values at 0.1x lr) and their training
does not compete with the backbone's. Freezing them for the **second half** is the one variant that is not worse:
against its paired control (same checkpoint, same warmup restart) it is +4 / +3 / +3 tasks (5.4 / 6.7 / 10.1 vs
2.7 / 4.7 / 8.1), token accuracy and lm loss equal; against the uninterrupted 4000-step control it is level (5.4 /
7.4 / 10.7). With the seed spread at 2-7 tasks this is one seed's worth of evidence that the heads converge early
and can be frozen at no cost, 13% faster per step (853 vs 895 s), and no evidence that freezing them *helps*. Note
the paired control itself sits 3-4 tasks below the uninterrupted run: the warmup restart at 2000 steps costs
something, which is why the paired comparison is the one to read.

**Question 2, final:** in training, TreeRM's heads can be frozen once trained (section 4 and here) but every schedule
that freezes them, or anything else, iteratively during training is worse than training everything every step; at
test time, training the heads with the embedding fits the demos but does not transfer (section 3c).

## 5. Open loop: ideas tried

TreeRM variants on the standard build, same recipe otherwise, judged on pass@2 against `treerm_std` (4.7%, seed
noise 2-4 tasks).

| variant | what | pass@1 / pass@2 / pass@5 @4k | train it/s | verdict |
|---|---|---|---|---|
| `mcts_horizon=full` | rollouts to the end of the last supervision step (up to 24 updates) | stopped at step 1296 | 1.3 (vs 5.1) | **too slow for this GPU**: 4x the training cost and an eval with depth-24 searches; killed to keep the queue moving |
| `halt_max_steps=16` | twice the supervision steps (and twice the eval searches) | 2.0 / 2.0 / 5.4 | 4.3 | no gain (within noise of the recipe's 2.7 / 4.7 / 6.7), 23% more wall-clock (1573 s) |
| `num_candidates=8` | eight flag candidates per z_H update | 4.0 / 4.0 / 4.7 | 4.9 | best pass@1 of any run (6 tasks) but no gain at pass@2/5; eval GPU memory doubles (1,150 MB allocated vs 550); within noise |
| `mcts_simulations=16, eval 32` | twice the search, train and eval | 1.3 / 2.7 / 6.0 | 4.5 | no gain, 14% more wall-clock (1459 s), eval memory 706 MB: the search depth is not what limits TreeRM here |
| `puzzle_emb_lr=3e-2` | task embeddings at 3x the lr (SignSGD) | **5.4 / 7.4 / 10.7** | 5.1 | **best run of the night**: 11 tasks at pass@2 vs the recipe's 7 (seeds 0/1: 7 and 3), token acc 0.738 vs 0.719. Each augmented copy's embedding gets only ~30 updates in 4000 steps, so a larger step helps. Seed 1: 3.4 / 4.7 / 6.0, so over two seeds 3e-2 averages pass@1 4.4% / pass@2 6.0% / pass@5 8.4% against the recipe's 2.0 / 3.3 / 4.7: a modest gain that holds across seeds. TRM at 3e-2 (TreeRM gain or recipe fix?): phase 7 |
| `puzzle_emb_lr=1e-1` | task embeddings at 10x the lr | 2.7 / 4.7 / 6.7 | 5.1 | back at the recipe's level (token acc 0.740): 3e-2 is the sweet spot, 1e-1 overshoots |
| TRM, `puzzle_emb_lr=3e-2` | the same change on the baseline | 4.7 / 8.1 / 10.1 | 7.3 | TRM gains as much (its seeds at 1e-2: 4.0 / 4.0 at pass@2), so the faster embedding lr is a **recipe fix for both recurrent models**, not a TreeRM advantage; with it TRM (8.1) and TreeRM (7.4 / 4.7 over two seeds) are level again |

| `num_candidates=8` + `puzzle_emb_lr=3e-2` | the two best knobs together | 5.4 / 8.1 / 12.1 | 4.9 | best 4k TreeRM run at pass@5 (18 tasks); level with emblr3 seed 0 at pass@1/2; eval memory 1,150 MB |
| TreeRM, `puzzle_emb_lr=3e-2`, 8000 steps | twice the budget | 4.0 / 6.7 / 10.1 @4k, 5.4 / 7.4 / 12.1 @6k, **6.7 / 10.1 / 12.8 @8k** | 5.0 | still climbing at 8k (15 tasks at pass@2, token acc 0.757, 2308 s): the budget, not the architecture, limits these numbers; TRM at 8k queued for the fair pair |

| TRM, `puzzle_emb_lr=3e-2`, 8000 steps | the fair pair for the row above | 4.7 / 6.0 / 8.7 @4k, 5.4 / 7.4 / 13.4 @6k, **6.7 / 9.4 / 15.4 @8k** | 7.5 | level with TreeRM at 8k (pass@1 equal, TreeRM +1 task at pass@2, TRM +4 at pass@5) in 58% of the wall-clock (1338 s) and 62% of the GPU memory. **Seed 1: 7.4 / 8.1 / 12.1** (2.0 / 4.7 / 6.0 @4k, 4.0 / 8.1 / 10.1 @6k; token acc 0.760, 1423 s) |

| TreeRM, K=8 + `puzzle_emb_lr=3e-2`, 8000 steps | the best configuration at the larger budget | 5.4 / 6.7 / 9.4 @4k, 5.4 / 10.7 / 12.1 @6k, **7.4 / 13.4 / 17.5 @8k** | 4.6 | **best run of the night**: 20 tasks at pass@2 vs TRM-8k's 14 and K=4 TreeRM-8k's 15; token acc 0.765, exact-row 2.8%; 2633 s, eval memory 1,150 MB. **Seed 1: 4.0 / 8.7 / 12.8** (3.4 / 4.7 / 8.7 @4k, 4.0 / 7.4 / 9.4 @6k; token acc 0.757, 2718 s): the seed spread (20 vs 13 tasks at pass@2) is as large as the gap to TRM-8k (14) |

Where that leaves question 3 after the loop: on MiniARC 5x5 at ~0.6M params and 4000 steps, TreeRM's search buys
nothing measurable over TRM (both ~5-8% pass@2 with the fixed embedding lr), at 2x the wall-clock and 1.6x the
memory; VARC is the cheapest and adapts best at test time. The one thing the loop found that moves every
recurrent model is the task-embedding lr. Phase 8 (running) tests whether longer training or K=8 opens a gap, and
whether the faster embedding lr changes TreeRM's TTT picture.

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

## 7. L-module test-time training (branch `l-reinit-ttt`)

**Idea (user, 2026-10-19).** Keep the pretrained H block, heads and puzzle embeddings; at TTT re-initialise the L block
(the one that reads the input, `z_L = f_L(z_L, z_H + x)`) and train only it on the task's demos, so the L block
becomes the task's program instead of a perturbation of the shared backbone. For GRAM, additionally train that L
block on the single unlabeled test input by raising the frozen LPRM head's predicted correctness.

**What it needed.** TRM/GRAM share one block for both updates (`H_layers` was a declared but unused config field), so
`arch.H_layers > 0` now builds a separate H block (`inner.H_blocks`, the H update only; 0 = the old shared model, no new
checkpoint keys). `evaluate_ttt.py` gained `reinit_substrings` (checkpoint keys not loaded: those modules keep a fresh
init), `train_only_substrings` (every other parameter frozen; the puzzle-embedding table, a buffer with its own SignSGD
optimizer, keeps training as in every section-3 variant) and `lprm_steps` (passes over the task's test inputs with
the model in eval mode, prior guidance, maximising `-log` of its LPRM value by gradient on the trainable parameters).
The held-out build was rebuilt in this worktree (`data/miniarc5-aug-31-ho30`, 30 tasks, seeds as section 1).

**Runs** (`scripts/miniarc5/queue_lreinit.sh`; the section-1 recipe, `puzzle_emb_lr=3e-2`, 4000 steps, W&B online):
`trm_sep_ho` (TRM, `H_layers=2`), `trm_ho2` (TRM shared, the control re-trained on the rebuilt build),
`gram_sep_ho` (GRAM, `H_layers=2`, the `sc_gram_pf` recipe: pooled LPRM head, final target, prior sample fraction
0.25, 4 bootstrap draws, PR #67's token-sum KL x0.07). TTT with `ttt_mini` over the 30 held-out tasks: `lr_full`
(everything trains), `lr_lonly` (`train_only_substrings=[inner.L_level.]`), `lr_lreinit` (plus
`reinit_substrings=[inner.L_level.]`); GRAM also `lr_lonly_lprm` and `lr_lreinit_lprm` (`lprm_steps=20`).
The separate-H models carry ~0.79M dense params vs 0.53M shared (the L block alone is 527,360, the count the TTT log
prints as trainable in the L-only variants); the within-checkpoint `lr_full` row is the comparison that matters.

**Smoke (2-task, 40-step checkpoints):** the L-reinit TTT drops exactly the 12 L-block keys, trains 527,360
parameters, and the LPRM phase moves the mean value (0.82 -> 0.98 on a 40-step model, where the critic is trivially
satisfiable); every variant evaluates and reports pass@k.

### Results
Raw numbers: each variant's `ttt_metrics.json` under
`~/ARC-agi-testbed/autoresearch_runs/miniarc5/<run>/outputs/*/*/*/eval_preds/step_4000_lr_<variant>/` (per-task
`tasks/<name>.json` next to it), the per-task training lines in `<run dir>/ttt_lr_<variant>.log`, and the eval rows
under `eval/lr_<variant>/*` in each run's W&B page. Pretrain wall: `trm_sep_ho` 503 s, `trm_ho2` 677 s,
`gram_sep_ho` 2541 s (the 4 bootstrap draws), no errors. Counts out of 30 held-out tasks; `ttt_mini` throughout.

| checkpoint | variant | pass@1 | pass@2 | pass@5 | token acc |
|---|---|---|---|---|---|
| TRM, shared, section 3 (`trm_ho`, emb lr 1e-2) | no TTT | 0 | 0 | 0 | 0.65 |
| TRM, shared, section 3 | Full-TTT (embedding + backbone) | 1 | 2 | 3 | 0.766 |
| TRM, shared, section 3 | Embed-TTT (embedding only) | 1 | 2 | 2 | 0.740 |
| TRM, shared, section 3 | Embed-Full-TTT (10 epochs embedding, then 10 backbone) | 1 | 1 | 1 | 0.743 |
| TRM, separate H (`trm_sep_ho`) | `lr_full`: everything trains (control) | 3 | 7 | 8 | 0.803 |
| TRM, separate H | `lr_lonly`: pretrained L block only | 3 | 3 | 5 | 0.793 |
| TRM, separate H | `lr_lreinit`: L block re-initialised, trained alone | 1 | 1 | 2 | 0.703 |
| TRM, shared (`trm_ho2`, emb lr 3e-2) | `lr_full` (control, rebuilt build) | 1 | 2 | 3 | 0.776 |
| GRAM, separate H (`gram_sep_ho`) | `lr_full` (control) | 1 | 1 | 4 | 0.778 |
| GRAM, separate H | `lr_lonly` | 2 | 2 | 4 | 0.779 |
| GRAM, separate H | `lr_lreinit` | 0 | 0 | 0 | 0.627 |
| GRAM, separate H | `lr_lonly_lprm` (+20 LPRM passes) | (running) | | | |
| GRAM, separate H | `lr_lreinit_lprm` (+20 LPRM passes) | (queued) | | | |

Second reading. The shared TRM re-trained on the rebuilt build repeats section 3 (1/2/3 at pass@2 then 1/2/3 now), so
the separate-H model's 3/7/8 under full fine-tuning is the separate H block, not seed spread: with its own H block
the model has more to adapt and adapts better (one seed, 30 tasks: 5 tasks is above the ~2-task spread seen between
seeds in section 3b, but a second seed is the natural check). GRAM with a separate H block adapts less (1/1/4 full;
L-only 2/2/4 is level with it), and re-initialising its L block solves nothing (token accuracy 0.63, below the
no-TTT 0.65 of the shared TRM). The label-free LPRM phase lifts the mean LPRM value (0.85 -> 0.92 per task on the
L-only variant); whether it lifts pass@k is the last two rows.

First reading (TRM): re-initialising the L block hurts. Training only the pretrained L block keeps about half of full
fine-tuning's gain (3 vs 7 at pass@2), and starting it from scratch keeps almost none (1): 160 steps on 16 aux
copies of ~3 demos do not fit a fresh 2-layer block well enough (token accuracy 0.70 vs 0.80). The separate-H model's
own full fine-tune, 7 of 30 at pass@2, is well above the shared TRM's 2 of 30 in section 3; the `trm_ho2` control on
the same rebuilt build says whether that is the separate H block or seed spread.
