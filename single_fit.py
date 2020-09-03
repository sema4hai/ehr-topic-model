# single_fit.py

import sys
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Set, Tuple

import pandas as pd
import yaml
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

from ehr_topic_model.util import (coherence, load_data, load_stopwords,
                                  print_model_performance, save_model,
                                  save_topics, topic_top_words)

SINGLE_FIT_CONFIG: Dict[str, Any] = {}
BEST_HYPERPARAMETERS: Dict[str, Any] = {}


def _fit_pipeline(project_home: Path, pipeline: Pipeline, X: pd.Series) -> None:
    """
    Set pipeline hyperparameters, fit to data, and save to disk.

    Parameters
    ----------
    project_home : pathlib.Path
        Project home directory.
    pipeline : sklearn.pipeline.Pipeline
        Topic model pipeline.
    X : pandas.Series
        Data.
    """
    pipeline.set_params(**BEST_HYPERPARAMETERS).fit(X)
    model_name: str = pipeline[-1].__class__.__name__
    metric_val: float = coherence(pipeline=pipeline, X=X)
    print_model_performance(model_name=model_name, metric_val=metric_val)
    fmt_topics_topwords: str = topic_top_words(
        model=pipeline[1],
        feature_names=pipeline[0].get_feature_names(),
        n_top_words=SINGLE_FIT_CONFIG["number_of_topic_top_words"],
    )
    output_dpath: Path = Path(project_home, SINGLE_FIT_CONFIG["output_dir"])
    output_dpath.mkdir(exist_ok=True)
    save_topics(
        topics=fmt_topics_topwords, model_name=model_name, output_dpath=output_dpath,
    )
    save_model(model=pipeline, model_name=model_name, output_dpath=output_dpath)


def main(pipeline_idx: int) -> None:
    """
    Train topic model on fixed hyperparameters.

    Parameters
    ----------
    pipeline_idx : int
        Index of pipeline to use.
    """
    # Exit program if pipeline_idx = 1 as tfidf+NMF hasn't been implemented yet.
    if pipeline_idx == 1:
        sys.exit(
            (
                "The TF-IDF + NMF pipeline hasn't been implemented yet, sorry! "
                "Please rerun the program with --pipeline set to 0."
            )
        )

    project_home: Path = Path(__file__).parent
    X: pd.Series = load_data(
        Path(project_home, "data", SINGLE_FIT_CONFIG["data_file"])
    ).iloc[:, 0]
    custom_stopwords: Set[str] = load_stopwords(
        Path(project_home, "config", SINGLE_FIT_CONFIG["stopwords"])
    )

    # Initialize pipelines, ordered by index.
    pipelines: Tuple[Pipeline, ...] = (
        Pipeline(
            [
                ("vect", CountVectorizer(stop_words=custom_stopwords)),
                (
                    "decomp",
                    LatentDirichletAllocation(
                        random_state=SINGLE_FIT_CONFIG["model"]["seed"],
                        n_jobs=SINGLE_FIT_CONFIG["model"]["n_jobs"],
                    ),
                ),
            ]
        ),
    )

    _fit_pipeline(project_home=project_home, pipeline=pipelines[pipeline_idx], X=X)


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(
        description="Train topic models on EHR notes with set hyperparameters.",
        formatter_class=RawTextHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="Configuration YAML file path (default: None)",
    )
    parser.add_argument(
        "-b",
        "--best_hyperparameters",
        default=None,
        type=str,
        help="Best hyperparameters YAML file path (default: None)",
    )
    parser.add_argument(
        "-p",
        "--pipeline",
        choices={0, 1},
        type=int,
        default=0,
        help=dedent(
            """\
            Topic modeling pipeline (default: 0)
            0 -> Term Counts + Latent Dirichlet Allocation
            1 -> TF-IDF + Non-negative Matrix Factorization
            """
        ),
    )

    args: Namespace = parser.parse_args()
    with Path(args.config).open() as config_f:
        SINGLE_FIT_CONFIG = yaml.load(stream=config_f, Loader=yaml.FullLoader)
    with Path(args.best_hyperparameters).open() as bhp_f:
        BEST_HYPERPARAMETERS = yaml.load(stream=bhp_f, Loader=yaml.FullLoader)

    main(pipeline_idx=args.pipeline)
