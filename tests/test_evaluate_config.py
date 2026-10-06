from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import evaluate
from evaluators.eval_config import EvalConfig

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "config")


def _eval_cfg(*overrides: str) -> EvalConfig:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="cfg_eval", overrides=list(overrides))
    return EvalConfig(**OmegaConf.to_container(cfg, resolve=True))


def test_cli_overrides_reach_the_eval_group_interpolations():
    cfg = _eval_cfg("checkpoint=ckpt.pth", "different_init=5", "noise_scale=0", "wandb_mode=disabled")
    assert cfg.checkpoint == "ckpt.pth" and cfg.different_init == 5 and cfg.wandb_mode == "disabled"
    assert cfg.param_iter["arch.noise_scale"] == 0
    assert cfg.suffix == "D16_B5_N0_S1.0"


def test_eval_group_selects_another_config():
    cfg = _eval_cfg("eval=sudoku_full_noise05")
    assert cfg.param_iter["arch.halt_max_steps"] == 64 and cfg.different_init == 128


class _FakeRun:
    def __init__(self):
        self.defined = []

    def define_metric(self, name, step_metric):
        self.defined.append((name, step_metric))


def _api(state):
    class Api:
        def run(self, path):
            if state is None:
                raise ValueError(f"no run {path}")
            run = _FakeRun()
            run.state = state
            return run

    return Api


META = {"wandb": {"id": "abc", "project": "proj", "entity": "me"}}


def test_wandb_attach_skips_without_a_usable_run(monkeypatch):
    def fail_init(**_):
        raise AssertionError("wandb.init must not be called")

    monkeypatch.setattr(evaluate.wandb, "init", fail_init)
    assert evaluate._attach_wandb_run(EvalConfig(wandb_mode="disabled"), META, "run_dir") is None
    assert evaluate._attach_wandb_run(EvalConfig(), {}, "run_dir") is None
    monkeypatch.setattr(evaluate.wandb, "Api", _api(None))
    assert evaluate._attach_wandb_run(EvalConfig(), META, "run_dir") is None
    # A live training run would have its history written over by a second resuming process
    monkeypatch.setattr(evaluate.wandb, "Api", _api("running"))
    assert evaluate._attach_wandb_run(EvalConfig(), META, "run_dir") is None


def test_wandb_attach_resumes_the_checkpoint_run(monkeypatch):
    seen = {}

    def fake_init(**kwargs):
        seen.update(kwargs)
        return _FakeRun()

    monkeypatch.setattr(evaluate.wandb, "Api", _api("finished"))
    monkeypatch.setattr(evaluate.wandb, "init", fake_init)
    run = evaluate._attach_wandb_run(EvalConfig(), META, "run_dir")
    assert seen == {"id": "abc", "project": "proj", "entity": "me", "resume": "must", "dir": "run_dir"}
    assert run.defined == [("eval/*", "step")]
