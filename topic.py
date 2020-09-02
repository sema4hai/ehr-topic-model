# topic.py

from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from ehr_topic_model.util import remove_nums

INFERENCE_CONFIG: Dict[str, Any] = {}


def _load_topics() -> pd.DataFrame:
    """Load topics as column headers of a new table"""
    col_names: pd.Series = pd.read_table(
        Path("models", INFERENCE_CONFIG["topics_file"]), header=None, usecols=[0]
    ).iloc[:, 0]
    col_names = col_names.append(pd.Series(["Most_Likely_Topic"]))

    df: pd.DataFrame = pd.DataFrame(columns=col_names)
    df.index.name = "note_id"

    return df


def _load_model() -> Pipeline:
    return joblib.load(filename=Path("models", INFERENCE_CONFIG["model_file"]))


def _load_data() -> pd.DataFrame:
    X: pd.DataFrame = pd.read_csv(
        Path("data", INFERENCE_CONFIG["raw_data"]), index_col=0
    )
    _ = X.apply(func=remove_nums, axis="columns")  # remove numbers from text
    return X


def _perform_inference(
    row: pd.Series, inference_df: pd.DataFrame, model: Pipeline
) -> None:
    topic_prob: np.ndarray = model.transform(row)[0]
    highest_prob_topic: str = inference_df.columns[np.argmax(topic_prob)]
    inference_df.loc[row.name] = np.append(topic_prob, highest_prob_topic)


def _save_inference(df: pd.DataFrame, output_dpath: Path) -> None:
    df.to_csv(
        path_or_buf=Path(
            output_dpath,
            "{model_name}_{inference_file_name}_topic_probabilities.tsv".format(
                model_name=INFERENCE_CONFIG["model_file"][:-4],
                inference_file_name=INFERENCE_CONFIG["raw_data"][:-4],
            ),
        ),
        sep="\t",
        float_format="%g",
    )


def main() -> None:
    inference_df: pd.DataFrame = _load_topics()
    model: Pipeline = _load_model()
    raw_data: pd.DataFrame = _load_data()

    _ = raw_data.apply(
        func=_perform_inference, axis="columns", inference_df=inference_df, model=model
    )
    _save_inference(df=inference_df, output_dpath=Path(Path(__file__).parent, "output"))


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(
        description="tmp (inference) description", formatter_class=RawTextHelpFormatter
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="Configuration YAML file path (default: None)",
    )

    args: Namespace = parser.parse_args()
    with Path(args.config).open() as config_f:
        INFERENCE_CONFIG = yaml.load(stream=config_f, Loader=yaml.FullLoader)[
            "inference"
        ]

    main()


# inference file

# load pipeline from models/
# print note ID, most likely topic, all topic probs
# (for each note)
