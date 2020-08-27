# ehr_topic_model/hpt/base_tuner.py

from abc import abstractmethod
from typing import Mapping, NoReturn, Optional, Union

from optuna import Study, Trial, create_study
from pandas import Series
from sklearn.pipeline import Pipeline


class BaseTuner:
    """
    Base object for pipeline tuners.
    """

    def __init__(
        self,
        trials: int,
        pipeline: Pipeline,
        hparams: Mapping[str, Mapping[str, Union[str, tuple]]],
        X: Series,
    ) -> None:
        """BaseTuner constructor."""
        self.pipeline: Pipeline = pipeline
        self.trials: int = trials
        self.hparams: Mapping[str, Mapping[str, Union[str, tuple]]] = hparams
        self.X: Series = X

    @abstractmethod
    def objective(self, trial: Trial) -> Union[float, NoReturn]:
        raise NotImplementedError

    def tune(
        self,
        study_name: str,
        storage: str,
        seed: Optional[int],
        direction: Optional[str],
        n_jobs: Optional[int],
        resume: Optional[bool] = True,
        **kwargs
    ) -> Pipeline:
        """Tune hyperparameters."""
        self.study: Study = create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=resume,
        )
        self.study.optimize(
            objective=self.objective, n_trials=self.trials, n_jobs=n_jobs
        )

        step: str
        seed_params: Mapping[str, Optional[int]] = {
            "{}__random_state".format(step[0]): seed
            for step in self.pipeline
            if "random_state" in step[1].get_params().keys()
        }

        return self.pipeline(**self.study.best_params, **seed_params).fit(self.X)
