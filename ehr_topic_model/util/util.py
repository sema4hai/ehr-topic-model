# ehr_topic_model/util/util.py

from numpy import ndarray
from pandas import Series
from scipy.sparse.csr import csr_matrix
from sklearn.pipeline import Pipeline
from tmtoolkit.topicmod.evaluate import metric_coherence_mimno_2011 as mimno_tc


def coherence(pipeline: Pipeline, X: Series) -> float:
    """Calculate topic coherence (Mimno 2011)."""
    dtm: csr_matrix = pipeline.named_steps["vect"].transform(X)
    topic_word_distr: ndarray = pipeline.named_steps["decomp"].components_
    return -mimno_tc(topic_word_distrib=topic_word_distr, dtm=dtm, return_mean=True)
