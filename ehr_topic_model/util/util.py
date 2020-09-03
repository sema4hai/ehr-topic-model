# ehr_topic_model/util/util.py

from typing import List, Union

from numpy import ndarray
from pandas import Series
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from tmtoolkit.topicmod.evaluate import metric_coherence_mimno_2011 as mimno_tc


def coherence(pipeline: Pipeline, X: Series) -> float:
    """
    Calculate topic coherence (Mimno 2011).

    Parameters
    ----------
    pipeline sklearn.pipeline.Pipeline
        Scikit-learn pipeline with a vectorizer and a decomposition component fit to
        data.
    X: pandas.Series
        The data the component is fit to; used to extract the document-term matrix.

    Returns
    -------
    The negated topic coherence value from Mimno 2011, to be minimized.
    """
    return -mimno_tc(
        topic_word_distrib=pipeline.named_steps["decomp"].components_,
        dtm=pipeline.named_steps["vect"].transform(X),
        return_mean=True,
    )


def remove_nums(row: Series) -> None:
    """
    Removes numbers from documents.
    To be used in conjunction with pandas.DataFrame.apply()

    Parameters
    ----------
    X : pandas.Series
        A row within the DataFrame containing text.
    """
    c: str
    row.iat[0] = "".join(c for c in row.iat[0] if not c.isdigit())


def topic_top_words(
    model: Union[NMF, LatentDirichletAllocation],
    feature_names: List[str],
    n_top_words: int,
) -> str:
    """
    Print and format topic top words.

    Parameters
    ----------
    model : sklearn.decomposition.NMF or sklearn.decomposition.LatentDirichletAllocation
        The topic model.
    feature_names : list of str
        The words/terms that exist within the documents the model is fit to.
    n_top_words : int
        Number of top words to display.

    Returns
    -------
    str
        Formatted topics and top words string to write to file.
    """
    topic_idx: int
    topic: ndarray
    message: str
    i: int
    output: str = ""
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic_{}\t".format(topic_idx)
        message += " ".join(
            [feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
        )
        print(message)
        output += message + "\n"
    print()  # extra newline for stdout formatting
    return output


def print_model_performance(model_name: str, metric_val: float) -> None:
    """
    Print model performance metric to stdout.

    Parameters
    ----------
    model_name : str
        The name of the topic model.
    metric_val : float
        The performance metric.
    """
    print_pad: str = "\n=====\n"  # padding
    print(
        "{pad}Model:\t\t\t{model}\nTopic Coherence:\t{metric_val}{pad}".format(
            pad=print_pad, model=model_name, metric_val=metric_val
        )
    )
