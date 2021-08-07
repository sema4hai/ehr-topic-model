# main.py

import sys
from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, Set, Tuple

import pandas as pd
import yaml
from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from ehr_topic_model.hpt import BaseTuner, CountVectorizerLdaTuner
from ehr_topic_model.util import (load_data, load_stopwords,
                                  print_model_performance, save_model,
                                  save_topics, topic_top_words)

# Global configuration dict.
CONFIG: Dict[str, Any] = {}


def _score_model(tuner: BaseTuner, output_dpath: Path) -> None:
    """
    Tune, score, and save the topic model.

    Parameters
    ----------
    tuner : ehr_topic_model.hpt.BaseTuner
        The tuner object for the topic model.
    output_dpath : pathlib.Path
        The output directory.
    """
    est: Pipeline = tuner.tune(
        **CONFIG["tuner"]["study"], **CONFIG["tuner"]["optimize"]
    )
    model_name: str = est[-1].__class__.__name__
    tuner.save_and_print_best_hparams(dpath=output_dpath, model_name=model_name)
    print_model_performance(model_name=model_name, metric_val=tuner.study.best_value)

    # Printing topics and top words to stdout.
    # Formatted topic string is extracted to write to file.
    fmt_topics_topwords: str = topic_top_words(
        model=est[1],
        feature_names=est[0].get_feature_names(),
        n_top_words=CONFIG["number_of_topic_top_words"],
    )
    output_dpath.mkdir(exist_ok=True)
    save_topics(
        topics=fmt_topics_topwords, model_name=model_name, output_dpath=output_dpath
    )
    save_model(model=est, model_name=model_name, output_dpath=output_dpath)


def main(pipeline_idx: int) -> None:
    """
    Train topic model.

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
    X: pd.Series = load_data(Path(project_home, "data", CONFIG["data_file"])).iloc[:, 0]
    custom_stopwords: Set[str] = load_stopwords(
        Path(project_home, "config", CONFIG["stopwords"])
    )

    # Initialize tuner objects, ordered by index.
    tuners: Tuple[BaseTuner, ...] = (
        CountVectorizerLdaTuner(
            trials=CONFIG["tuner"]["trials"],
            pipeline=Pipeline(
                [
                    ("vect", CountVectorizer(stop_words=custom_stopwords)),
                    ("decomp", LatentDirichletAllocation(learning_method="online")),
                ]
            ),
            hparams=dict(
                (k, v)
                for k, v in CONFIG.items()
                if len(k.split("_")) > 1 and k.split("_")[1] == "hparams"
            ),
            X=X,
        ),
    )

    _score_model(
        tuner=tuners[pipeline_idx],
        output_dpath=Path(project_home, CONFIG["output_dir"]),
    )


# CLI
if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(
        description="Train topic models on EHR notes.",
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
        CONFIG = yaml.load(stream=config_f, Loader=yaml.FullLoader)

    main(pipeline_idx=args.pipeline)
