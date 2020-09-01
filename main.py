# main.py

from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
import yaml
from sklearn.decomposition import LatentDirichletAllocation
# from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from ehr_topic_model.hpt import BaseTuner, CountVectorizerLdaTuner
from ehr_topic_model.util import remove_nums, topic_top_words

CONFIG: Dict[str, Any] = {}


def _save_topics(topics: str, output_dpath: Path) -> None:
    with Path(output_dpath, "topics.tsv").open("w") as topic_f:
        topic_f.write(topics)


def _score_model(tuner: BaseTuner, output_dpath: Path) -> None:
    """Tune and score models; resume as necessary."""
    est: Pipeline = tuner.tune(
        **CONFIG["tuner"]["study"], **CONFIG["tuner"]["optimize"]
    )
    model_name: str = est[-1].__class__.__name__
    print_pad: str = "\n=====\n"
    print(
        "{pad}{model}\nMimno Coherence: {metric_val}{pad}".format(
            pad=print_pad, model=model_name, metric_val=tuner.study.best_value
        )
    )
    topics: str = topic_top_words(
        model=est[1],
        feature_names=est[0].get_feature_names(),
        n_top_words=CONFIG["number_of_topic_top_words"],
    )
    _save_topics(topics=topics, output_dpath=output_dpath)

    joblib.dump(value=est, filename=Path(output_dpath, "{}.pkl".format(model_name)))


def _load_stopwords(project_home: Path) -> List[str]:
    """Load stopwords from file."""
    stopwords: List[str] = []
    line: str
    with Path(project_home, "config", CONFIG["stopwords"]).open() as sw_f:
        for line in sw_f:
            stopwords.append(line.rstrip("\n"))

    return stopwords


def main(pipeline_idx: int = 0) -> None:
    """Train models."""
    # Load data
    project_home: Path = Path(__file__).parent
    X: pd.DataFrame = pd.read_csv(
        Path(project_home, "data", CONFIG["data_file"]), index_col=0
    )
    _ = X.apply(func=remove_nums, axis="columns")  # remove numbers

    # Load custom stopwords
    custom_stopwords: List[str] = _load_stopwords(project_home=project_home)

    tuners: Tuple[BaseTuner, ...] = (
        CountVectorizerLdaTuner(
            trials=CONFIG["tuner"]["trials"],
            pipeline=Pipeline(
                [
                    ("vect", CountVectorizer(stop_words=custom_stopwords),),
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

    output_dpath: Path = Path(project_home, "models")
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
