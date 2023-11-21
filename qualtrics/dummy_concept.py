from joblib import Parallel, delayed
from pydantic import BaseModel
from typing import Callable

import numpy as np
from scipy.interpolate import interp1d
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import KFold

# The maximum number of cross-validation models to train.
MAX_NUM_CROSS_VAL_MODELS = 15
F_BETA_WEIGHT = 0.5
C = 50

class ConceptMetrics(BaseModel):
  """Metrics for a concept."""

  # The average F1 score for the concept computed using cross validation.
  f1: float
  precision: float
  recall: float
  roc_auc: float

class LogisticEmbeddingModel:
  """A model that uses logistic regression with embeddings."""

  _metrics = None
  _threshold: float = 0.5

  def __init__(self) -> None:
    # See `notebooks/Toxicity.ipynb` for an example of training a concept model.
    self._model = LogisticRegression(
      class_weight='balanced',
      C=C,
      tol=1e-5,
      warm_start=True,
      max_iter=5_000,
      n_jobs=-1
    )

  def score_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
    """Get the scores for the provided embeddings."""
    y_probs = self._model.predict_proba(embeddings)[:, 1]
    # Map [0, threshold, 1] to [0, 0.5, 1].
    interpolate_fn = interp1d([0, self._threshold, 1], [0, 0.4999, 1])
    return interpolate_fn(y_probs)

  def _setup_training(
    self, X_train: np.ndarray, labels: list[bool]
  ) -> tuple[np.ndarray, np.ndarray]:
    y_train = np.array(labels)
    # Shuffle the data in unison.
    #p = np.random.permutation(len(X_train))
    #X_train = X_train[p]
    #y_train = y_train[p]
    return X_train, y_train

  def fit(self, embeddings: np.ndarray, labels: list[bool]) -> None:
    """Fit the model to the provided embeddings and labels."""
    label_set = set(labels)
    if len(label_set) < 2:
      dim = embeddings.shape[1]
      random_vector = np.random.randn(dim).astype(np.float32)
      random_vector /= np.linalg.norm(random_vector)
      embeddings = np.vstack([embeddings, random_vector])
      labels.append(False if True in label_set else True)

    if len(labels) != len(embeddings):
      raise ValueError(
        f'Length of embeddings ({len(embeddings)}) must match length of labels ({len(labels)})'
      )
    self._X_train, self._y_train = self._setup_training(embeddings, labels)
    self._model.fit(self._X_train, self._y_train)
    self._metrics, self._threshold, self._fold, self._y_test, self._y_pred = self._compute_metrics(embeddings, labels)

  def _compute_metrics(
    self, embeddings: np.ndarray, labels: list[bool]
  ) -> tuple[ConceptMetrics, float, np.ndarray, np.ndarray, np.ndarray]:
    """Return the concept metrics."""
    labels_np = np.array(labels)
    #n_splits = min(len(labels_np), MAX_NUM_CROSS_VAL_MODELS)
    n_splits = len(labels_np)
    fold = KFold(n_splits, shuffle=True, random_state=42)

    def _fit_and_score(
      model: LogisticRegression,
      X_train: np.ndarray,
      y_train: np.ndarray,
      X_test: np.ndarray,
      y_test: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
      if len(set(y_train)) < 2:
        return np.array([]), np.array([])
      model.fit(X_train, y_train)
      y_pred = model.predict_proba(X_test)[:, 1]
      return y_test, y_pred

    # Compute the metrics for each validation fold in parallel.
    jobs: list[Callable] = []
    test_folds = []
    for train_index, test_index in fold.split(embeddings):
      X_train, y_train = embeddings[train_index], labels_np[train_index]
      X_train, y_train = self._setup_training(X_train, y_train)
      X_test, y_test = embeddings[test_index], labels_np[test_index]
      test_folds.append(test_index)
      model = clone(self._model)
      jobs.append(delayed(_fit_and_score)(model, X_train, y_train, X_test, y_test))
    results = Parallel(n_jobs=-1)(jobs)

    y_test = np.concatenate([y_test for y_test, _ in results], axis=0)
    y_pred = np.concatenate([y_pred for _, y_pred in results], axis=0)
    if len(set(y_test)) < 2:
      return None, 0.5, None, None, None
    roc_auc_val = roc_auc_score(y_test, y_pred)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    numerator = (1 + F_BETA_WEIGHT**2) * precision * recall
    denom = (F_BETA_WEIGHT**2 * precision) + recall
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom != 0))
    max_f1: float = np.max(f1_scores)
    max_f1_index = np.argmax(f1_scores)
    max_f1_thresh: float = thresholds[max_f1_index]
    max_f1_prec: float = precision[max_f1_index]
    max_f1_recall: float = recall[max_f1_index]
    metrics = ConceptMetrics(
      f1=max_f1,
      precision=max_f1_prec,
      recall=max_f1_recall,
      roc_auc=float(roc_auc_val),
    )
    return metrics, max_f1_thresh, np.array(test_folds), y_test, y_pred