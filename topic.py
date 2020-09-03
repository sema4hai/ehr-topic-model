# topic.py

from argparse import ArgumentParser, Namespace, RawTextHelpFormatter
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline

from ehr_topic_model.util import load_data

INFERENCE_CONFIG: Dict[str, Any] = {}


def _load_topics(project_home: Path) -> pd.DataFrame:
    """
    Load topics as column headers of a new dataframe.

    Parameters
    ----------
    project_home : pathlib.Path
        Project home directory.

    Returns
    -------
    pd.DataFrame
        An empty dataframe with topic column headers, note ID index, and a
        "most likely topic" column.
    """
    col_names: pd.Series = pd.read_table(
        filepath_or_buffer=Path(project_home, INFERENCE_CONFIG["topics_file"]),
        header=None,
        usecols=[0],
    ).iloc[:, 0]
    col_names = col_names.append(pd.Series(["Most_Likely_Topic"]))
    df: pd.DataFrame = pd.DataFrame(columns=col_names)
    df.index.name = "note_id"
    return df


def _load_model(project_home: Path) -> Pipeline:
    """
    Load the serialized topic model.

    Parameters
    ----------
    project_home : pathlib.Path
        Project home directory.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The previously trained topic model.
    """
    return joblib.load(Path(project_home, INFERENCE_CONFIG["model_file"]))


def _perform_inference(
    row: pd.Series, inference_df: pd.DataFrame, model: Pipeline
) -> None:
    """
    Perform inference on a given note document.
    To be used in conjunction with pandas.DataFrame.apply()

    Parameters
    ----------
    row : pd.Series
        A row within the DataFrame containing text.
    inference_df : pd.DataFrame
        The final dataframe storing topic probabilities for the given text.
    model : sklearn.pipeline.Pipeline
        The fitted topic model to perform inference.
    """
    topic_prob: np.ndarray = model.transform(row)[0]
    highest_prob_topic: str = inference_df.columns[np.argmax(topic_prob)]
    inference_df.loc[row.name] = np.append(topic_prob, highest_prob_topic)


def _save_inference(df: pd.DataFrame, output_dpath: Path) -> None:
    """
    Save inference results to disk.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to save to disk.
    output_dpath : pathlib.Path
        The output directory path.
    """
    output_dpath.mkdir(exist_ok=True)
    df.to_csv(
        path_or_buf=Path(
            output_dpath,
            "{model_name}_{inference_file_name}_topic_probabilities.tsv".format(
                model_name=INFERENCE_CONFIG["model_file"].split("/")[-1][:-4],
                inference_file_name=INFERENCE_CONFIG["data_file"][:-4],
            ),
        ),
        sep="\t",
        float_format="%g",
    )


def main() -> None:
    """Perform inference with the trained topic model."""
    project_home: Path = Path(__file__).parent
    inference_df: pd.DataFrame = _load_topics(project_home)
    model: Pipeline = _load_model(project_home)
    raw_data: pd.DataFrame = load_data(
        Path(project_home, "data", INFERENCE_CONFIG["data_file"])
    )
    _ = raw_data.apply(
        func=_perform_inference, axis="columns", inference_df=inference_df, model=model
    )
    _save_inference(
        df=inference_df,
        output_dpath=Path(Path(__file__).parent, INFERENCE_CONFIG["output_dir"]),
    )


if __name__ == "__main__":
    parser: ArgumentParser = ArgumentParser(
        description=(
            "Find topic probabilities for given EHR notes "
            "through an existing topic model."
        ),
        formatter_class=RawTextHelpFormatter,
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
        INFERENCE_CONFIG = yaml.load(stream=config_f, Loader=yaml.FullLoader)

    main()
