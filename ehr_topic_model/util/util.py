# ehr_topic_model/util/util.py

from typing import List, Union

from numpy import ndarray
from pandas import Series
from scipy.sparse.csr import csr_matrix
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.pipeline import Pipeline
from tmtoolkit.topicmod.evaluate import metric_coherence_mimno_2011 as mimno_tc


def coherence(pipeline: Pipeline, X: Series) -> float:
    """Calculate topic coherence (Mimno 2011)."""
    dtm: csr_matrix = pipeline.named_steps["vect"].transform(X)
    topic_word_distrib: ndarray = pipeline.named_steps["decomp"].components_
    return -mimno_tc(topic_word_distrib=topic_word_distrib, dtm=dtm, return_mean=True)


def remove_nums(X: Series) -> None:
    c: str
    X.iat[0] = "".join(c for c in X.iat[0] if not c.isdigit())


def topic_top_words(
    model: Union[NMF, LatentDirichletAllocation],
    feature_names: List[str],
    n_top_words: int,
) -> str:
    topic_idx: int
    topic: ndarray
    message: str
    i: int
    output: str = ""
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic {}: ".format(topic_idx)
        message += " ".join(
            [feature_names[i] for i in topic.argsort()[: -n_top_words - 1 : -1]]
        )
        print(message)
        output += message + "\n"

    return output
