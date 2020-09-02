# ehr_topic_model/hpt/pipeline_tuners.py

from typing import Dict, Mapping, Tuple, Union

from ehr_topic_model.hpt.base_tuner import BaseTuner
from ehr_topic_model.util import coherence
from optuna import Trial
from pandas import Series
from sklearn.pipeline import Pipeline


class CountVectorizerLdaTuner(BaseTuner):
    """
    Base object for pipeline tuners.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
       Topic model pipeline.
    trials : int
       Number of optuna trials to perform.
    hparams : dict of {str:dict of {str:tuple}}
       Dictionary of hyperparameters for each component.
    X : pd.Series
       Data.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
       Topic model pipeline.
    trials : int
       Number of optuna trials to perform.
    hparams : dict of {str:dict of {str:tuple}}
       Dictionary of hyperparameters for each component.
    X : pd.Series
       Data.
    """

    def __init__(
        self,
        trials: int,
        pipeline: Pipeline,
        hparams: Mapping[
            str, Mapping[str, Union[Tuple[float, float], Tuple[int, int]]]
        ],
        X: Series,
    ) -> None:
        super().__init__(trials=trials, pipeline=pipeline, hparams=hparams, X=X)

    def objective(self, trial: Trial) -> float:
        """
        Optuna optimization method.

        Parameters
        ----------
        trial : optuna.Trial
            An optuna trial object.

        Returns
        -------
        float
            The performance evaluation metric value for a single trial.
        """
        suggest: Dict[str, float] = {
            "vect__max_df": trial.suggest_uniform(
                name="vect__max_df",
                low=self.hparams["vectorizer_hparams"]["max_df"][0],
                high=self.hparams["vectorizer_hparams"]["max_df"][1],
            ),
            "decomp__doc_topic_prior": trial.suggest_loguniform(
                name="decomp__doc_topic_prior",
                low=self.hparams["lda_hparams"]["alpha"][0],
                high=self.hparams["lda_hparams"]["alpha"][1],
            ),
            "decomp__topic_word_prior": trial.suggest_loguniform(
                name="decomp__topic_word_prior",
                low=self.hparams["lda_hparams"]["beta"][0],
                high=self.hparams["lda_hparams"]["beta"][1],
            ),
            "decomp__n_components": trial.suggest_int(
                name="decomp__n_components",
                low=self.hparams["lda_hparams"]["num_topics"][0],
                high=self.hparams["lda_hparams"]["num_topics"][1],
            ),
            "decomp__max_iter": trial.suggest_int(
                name="decomp__max_iter",
                low=self.hparams["lda_hparams"]["iterations"][0],
                high=self.hparams["lda_hparams"]["iterations"][1],
            ),
            "decomp__learning_decay": trial.suggest_uniform(
                name="decomp__learning_decay",
                low=self.hparams["lda_hparams"]["decay"][0],
                high=self.hparams["lda_hparams"]["decay"][1],
            ),
            "decomp__learning_offset": trial.suggest_float(
                name="decomp__learning_offset",
                low=self.hparams["lda_hparams"]["offset"][0],
                high=self.hparams["lda_hparams"]["offset"][1],
            ),
        }
        est: Pipeline = self.pipeline.set_params(**suggest).fit(self.X)
        return coherence(pipeline=est, X=self.X)
