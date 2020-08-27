# ehr_topic_model/hpt/score_save_pipelines.py

from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
import yaml
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from ehr_topic_model.hpt import BaseTuner, CountVectorizerLdaTuner
from ehr_topic_model.util import remove_nums

CONFIG: Dict[str, Any] = {}


def _score_model(tuner: BaseTuner, output_dpath: Path) -> None:
    """Tune and score models; resume as necessary."""
    est: Pipeline = tuner.tune(
        **CONFIG["tuner"]["study"], **CONFIG["tuner"]["optimize"]
    )
    model_name = est[-1].__class__.__name__
    print("{}: [Mimno Coherence: {}]".format(model_name, tuner.study.best_value))

    # TODO:
    # - print topic top words
    # - save topic top words to table file?
    # - rename this function to something more sensible, or maybe this would be
    #   different function

    joblib.dump(value=est, filename=Path(output_dpath, "{}.pkl".format(model_name)))


def main(pipeline_idx: int = 0) -> None:
    """Train models."""
    # Load data
    project_home: Path = Path(__file__).parent
    X: pd.DataFrame = pd.read_csv(
        Path(project_home, "data", CONFIG["data_file"]), index_col=0
    )
    _ = X.apply(func=remove_nums, axis="columns")  # remove numbers

    # TODO: refactor preprocessing steps into single tokenizer class

    # Load custom stopwords
    custom_stopwords: List[str] = []
    with Path(project_home, "config", CONFIG["stopwords"]).open() as sw_f:
        line: str
        for line in sw_f:
            custom_stopwords.append(line.rstrip("\n"))

    output_dpath: Path = Path(project_home, "models")
    tuners: Tuple[BaseTuner, ...] = (
        CountVectorizerLdaTuner(
            trials=CONFIG["tuner"]["trials"],
            pipeline=Pipeline(
                [
                    ("vect", CountVectorizer(stop_words=custom_stopwords)),
                    ("decomp", LatentDirichletAllocation()),
                ]
            ),
            hparams=dict(
                (k, v)
                for k, v in CONFIG.items()
                if len(k.split("_")) > 1 and k.split("_")[1] == "hparams"
            ),
            X=X.iloc[:, 0],
        ),
    )

    _score_model(tuner=tuners[pipeline_idx], output_dpath=output_dpath)


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(
        description="tmp description", formatter_class=RawTextHelpFormatter
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
            0 -> Term Frequency + Latent Dirichlet Allocation
            1 -> TF-IDF + Non-negative Matrix Factorization
            """
        ),
    )
    args: Namespace = parser.parse_args()
    with Path(args.config).open() as config_f:
        CONFIG = yaml.load(stream=config_f, Loader=yaml.FullLoader)

    main(pipeline_idx=args.pipeline)
