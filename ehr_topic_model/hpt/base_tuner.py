# ehr_topic_model/hpt/base_tuner.py

from abc import abstractmethod
from typing import Dict, Mapping, NoReturn, Union

from optuna import Study, Trial, create_study
from pandas import Series
from sklearn.pipeline import Pipeline


class BaseTuner:
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
        pipeline: Pipeline,
        trials: int,
        hparams: Mapping[str, Mapping[str, Union[str, tuple]]],
        X: Series,
    ) -> None:
        self.pipeline: Pipeline = pipeline
        self.trials: int = trials
        self.hparams: Mapping[str, Mapping[str, Union[str, tuple]]] = hparams
        self.X: Series = X

    @abstractmethod
    def objective(self, trial: Trial) -> Union[float, NoReturn]:
        """Optuna objective method. Implement in child classes."""
        raise NotImplementedError

    def tune(
        self,
        study_name: str,
        storage: str,
        seed: int,
        direction: str,
        n_jobs: int,
        resume: bool = True,
        **kwargs
    ) -> Pipeline:
        """
        Tune hyperparameters.

        Parameters
        ----------
        study_name : str
            Name of the study.
        storage : str
            SQLite URI for local storage.
        seed : int
            Random state.
        direction : str
            Direction of optimization.
        n_jobs : int
            Parallelization.
        resume : bool
            Resume previously saved study or begin new study.
        **kwargs
            Hyperparameter keyword arguments passed to the topic model Pipeline.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Initialized topic model pipeline with set hyperparameters.
        """
        self.study: Study = create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=resume,
        )
        self.study.optimize(func=self.objective, n_trials=self.trials, n_jobs=n_jobs)

        # Set proper seed parameter names for use as kwargs.
        step: str
        seed_params: Dict[str, int] = {
            "{}__random_state".format(step[0]): seed
            for step in self.pipeline.steps
            if "random_state" in step[1].get_params().keys()
        }

        # Quick print
        print("\nBest Params:\n")
        k: str
        v: Union[float, int]
        for k, v in self.study.best_params.items():
            print("{k}:\t{v}".format(k=k, v=v))

        return self.pipeline.set_params(**self.study.best_params, **seed_params).fit(
            self.X
        )
