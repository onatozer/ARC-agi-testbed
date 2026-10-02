from typing import Dict, Optional

import torch

from dataset.build_smallarc_dataset import SourceSeparator
from evaluators.arc import ARC


class SmallARC(ARC):
    """ARC evaluator that also reports pass@K per source (miniarc, arc1_eval, ...) of the SmallARC dataset."""

    def result(self, save_path: Optional[str], rank: int, world_size: int, group: Optional[torch.distributed.ProcessGroup] = None) -> Optional[Dict[str, float]]:
        all_results = super().result(save_path, rank, world_size, group)
        if all_results is None:
            return None

        by_source = {}
        for name, scores in self.puzzle_correct.items():
            by_source.setdefault(name.split(SourceSeparator)[0], []).append(scores)

        for source, scores in by_source.items():
            for i, k in enumerate(self.pass_Ks):
                all_results[f"SmallARC/{source}/pass@{k}"] = sum(s[i] for s in scores) / len(scores)

        return all_results
