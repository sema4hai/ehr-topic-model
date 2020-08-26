# ehr_topic_model/hpt/score_save_pipelines.py

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd
import yaml
from ehr_topic_model.hpt.base_tuner import BaseTuner
from ehr_topic_model.hpt.pipeline_tuners import CountVectorizerLdaTuner
from ehr_topic_model.util.util import coherence
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

CFG: Dict[str, Any] = {}


def _score_model(
    tuner: BaseTuner,
    output_dpath: Path,
    study_name: str,
    storage: str,
    seed: Optional[int] = 0,
    direction: Optional[str] = "minimize",
    n_jobs: Optional[int] = 1,
    resume: Optional[bool] = False,
) -> None:
    """Tune and score models; resume as necessary."""
    est: Pipeline = tuner.tune(
        study_name=study_name,
        storage=storage,
        seed=seed,
        direction=direction,
        n_jobs=n_jobs,
        resume=resume,
    )
    model_name = est[-1].__class__.__name__
    print("{}: [Mimno Coherence: {}]".format(model_name, tuner.study.best_value))
    joblib.dump(value=est, filename=Path(output_dpath, "{}.pkl".format(model_name)))


def main(trials: int, study_name: str, storage: str, seed: Optional[int] = 0) -> None:
    """Train models."""
    pass


if __name__ == "__main__":
    trials: int = int(sys.argv[1])
    study_name: str = sys.argv[2]
    storage: str = sys.argv[3]
    seed: int = 0
    resume: bool = bool(sys.argv[5].lower())

    try:
        seed = int(sys.argv[4])
    except IndexError:
        pass

    # main(trials=trials, split=split, folds=folds, seed=seed)
