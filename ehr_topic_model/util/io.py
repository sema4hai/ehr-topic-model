# ehr_topic_model/util/io.py

from pathlib import Path
from typing import Set

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .util import remove_nums


def load_data(fpath: Path) -> pd.DataFrame:
    """
    Load data from CSV.
    File should contain columns: `note_id` and `full_note norm`.

    Parameters
    ----------
    fpath : pathlib.Path
        The data CSV file path.

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame containing the note documents.
        Note ID serves as the index.
    """
    X: pd.DataFrame = pd.read_csv(
        filepath_or_buffer=fpath,
        index_col="note_id",
        usecols=["note_id", "full_note_norm"],
    )
    _ = X.apply(func=remove_nums, axis="columns")  # remove numbers from text
    return X


def load_stopwords(fpath: Path) -> Set[str]:
    """
    Load stopwords from file.

    Parameters
    ----------
    fpath : pathlib.Path
        The stopwords file.

    Returns
    -------
    set of str
        Set of stopwords.
    """
    stopwords: Set[str]
    line: str
    with fpath.open() as sw_f:
        stopwords = {line.rstrip("\n") for line in sw_f}
    return stopwords


def save_topics(topics: str, model_name: str, output_dpath: Path) -> None:
    """
    Write topics and top words to TSV file.

    Parameters
    ----------
    topics : str
        Formatted topics string. In the format of "Topic_#:\t{top words}\n".
    model_name : str
        The name of the topic model
    output_dpath : pathlib.Path
        The output directory.
    """
    with Path(output_dpath, "{}_topics.tsv".format(model_name)).open("w") as topic_f:
        topic_f.write(topics)


def save_model(model: Pipeline, model_name: str, output_dpath: Path) -> None:
    """
    Save serialized model to disk.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        The fit model to serialize and save to disk.
    model_name : str
        The name of the topic model.
    output_dpath : pathlib.Path
        The output directory path.
    """
    joblib.dump(value=model, filename=Path(output_dpath, "{}.pkl".format(model_name)))
