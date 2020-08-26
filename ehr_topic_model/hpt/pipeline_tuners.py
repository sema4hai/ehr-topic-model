# ehr_topic_model/hpt/pipeline_tuners.py

from typing import Dict, Mapping, Tuple

from ehr_topic_model.hpt.base_tuner import BaseTuner
from ehr_topic_model.util.util import coherence
from optuna import Trial
from pandas import Series
from sklearn.pipeline import Pipeline


class CountVectorizerLdaTuner(BaseTuner):
    def __init__(
        self,
        trials: int,
        pipeline: Pipeline,
        hparams: Mapping[str, Tuple[float, float]],
        X: Series,
    ) -> None:
        super().__init__(trials=trials, pipeline=pipeline, hparams=hparams, X=X)

    def objective(self, trial: Trial) -> float:
        suggest: Dict[str, float] = {
            "vect__max_df": trial.suggest_uniform(
                name="max_df",
                low=self.hparams["max_df"][0],
                high=self.hparams["max_df"][1],
            ),
            "decomp__doc_topic_prior": trial.suggest_loguniform(
                name="alpha",
                low=self.hparams["alpha"][0],
                high=self.hparams["alpha"][1],
            ),
            "decomp__topic_word_prior": trial.suggest_loguniform(
                name="beta", low=self.hparams["beta"][0], high=self.hparams["beta"][1]
            ),
            "decomp__n_components": trial.suggest_int(
                name="topics",
                low=self.hparams["topics"][0],
                high=self.hparams["topics"][1],
            ),
            "decomp__max_iter": trial.suggest_int(
                name="max_iter",
                low=self.hparams["max_iter"][0],
                high=self.hparams["max_iter"][1],
            ),
            "decomp__learning_decay": trial.suggest_uniform(
                name="decay",
                low=self.hparams["decay"][0],
                high=self.hparams["decay"][1],
            ),
            "decomp__learning_offset": trial.suggest_float(
                name="offset",
                low=self.hparams["offset"][0],
                high=self.hparams["offset"][1],
            ),
        }

        est: Pipeline = self.pipeline(**suggest).fit(self.X)

        return coherence(pipeline=est, X=self.X)
