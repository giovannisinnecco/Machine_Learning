"""Classificador k-nearest neighbors e validacao cruzada por sujeito.

O modulo implementa distancia euclidiana, votacao majoritaria e avaliacao por
acuracia balanceada. A validacao cruzada recebe os identificadores dos sujeitos
e mantem todas as janelas de uma pessoa dentro do mesmo fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _as_feature_matrix(X: np.ndarray) -> np.ndarray:
    matrix = np.asarray(X, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("X deve ser uma matriz bidimensional nao vazia.")
    if not np.isfinite(matrix).all():
        raise ValueError("X contem NaN ou infinito.")
    return matrix


def _as_label_vector(y: np.ndarray, expected_size: int | None = None) -> np.ndarray:
    labels = np.asarray(y).reshape(-1)
    if labels.size == 0:
        raise ValueError("y nao pode ser vazio.")
    if expected_size is not None and labels.shape[0] != expected_size:
        raise ValueError("X e y devem possuir o mesmo numero de amostras.")
    return labels


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Iterable | None = None,
) -> np.ndarray:
    """Constroi uma matriz em que linhas sao classes reais e colunas, preditas."""

    true = _as_label_vector(y_true)
    pred = _as_label_vector(y_pred, true.shape[0])
    class_values = (
        np.unique(np.concatenate((true, pred)))
        if labels is None
        else np.asarray(tuple(labels))
    )
    if class_values.size == 0 or np.unique(class_values).size != class_values.size:
        raise ValueError("labels deve conter classes distintas.")

    class_to_index = {value: index for index, value in enumerate(class_values.tolist())}
    matrix = np.zeros((class_values.size, class_values.size), dtype=np.int64)
    for true_value, predicted_value in zip(true.tolist(), pred.tolist(), strict=True):
        if true_value not in class_to_index or predicted_value not in class_to_index:
            raise ValueError("Foi encontrado um rotulo que nao esta em labels.")
        matrix[class_to_index[true_value], class_to_index[predicted_value]] += 1
    return matrix


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula a media do recall obtido em cada classe observada."""

    true = _as_label_vector(y_true)
    pred = _as_label_vector(y_pred, true.shape[0])
    recalls = []
    for label in np.unique(true):
        mask = true == label
        recalls.append(float(np.mean(pred[mask] == label)))
    return float(np.mean(recalls))


class KNNClassifier:
    """kNN com distancia euclidiana e desempate pela menor classe."""

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        batch_size: int = 256,
        standardize: bool = True,
    ) -> None:
        if n_neighbors <= 0:
            raise ValueError("n_neighbors deve ser positivo.")
        if batch_size <= 0:
            raise ValueError("batch_size deve ser positivo.")
        self.n_neighbors = int(n_neighbors)
        self.batch_size = int(batch_size)
        self.standardize = bool(standardize)

        self.classes_: np.ndarray | None = None
        self.X_train_: np.ndarray | None = None
        self.y_encoded_: np.ndarray | None = None
        self.train_squared_norms_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def _check_is_fitted(self) -> None:
        if (
            self.classes_ is None
            or self.X_train_ is None
            or self.y_encoded_ is None
            or self.train_squared_norms_ is None
        ):
            raise RuntimeError("O modelo ainda nao foi ajustado.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> KNNClassifier:
        """Armazena o conjunto de referencia usado para buscar vizinhos."""

        features = _as_feature_matrix(X)
        labels = _as_label_vector(y, features.shape[0])
        if self.n_neighbors > features.shape[0]:
            raise ValueError("n_neighbors nao pode exceder o numero de amostras.")

        if self.standardize:
            self.mean_ = features.mean(axis=0)
            self.scale_ = features.std(axis=0)
            self.scale_ = np.where(self.scale_ == 0.0, 1.0, self.scale_)
            features = (features - self.mean_) / self.scale_
        else:
            self.mean_ = None
            self.scale_ = None

        self.classes_, self.y_encoded_ = np.unique(labels, return_inverse=True)
        self.X_train_ = np.ascontiguousarray(features, dtype=np.float64)
        self.y_encoded_ = self.y_encoded_.astype(np.int64, copy=False)
        self.train_squared_norms_ = np.einsum(
            "ij,ij->i", self.X_train_, self.X_train_
        )
        return self

    def _transform_query(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        features = _as_feature_matrix(X)
        assert self.X_train_ is not None
        if features.shape[1] != self.X_train_.shape[1]:
            raise ValueError("X possui um numero de atributos diferente do treino.")
        if self.standardize:
            assert self.mean_ is not None
            assert self.scale_ is not None
            features = (features - self.mean_) / self.scale_
        return np.ascontiguousarray(features, dtype=np.float64)

    def _ordered_neighbor_labels(
        self, X: np.ndarray, max_neighbors: int
    ) -> np.ndarray:
        """Retorna os vizinhos ordenados por distancia para cada consulta."""

        self._check_is_fitted()
        query = self._transform_query(X)
        assert self.X_train_ is not None
        assert self.y_encoded_ is not None
        assert self.train_squared_norms_ is not None
        if max_neighbors <= 0 or max_neighbors > self.X_train_.shape[0]:
            raise ValueError("Quantidade de vizinhos invalida.")

        neighbor_labels = np.empty(
            (query.shape[0], max_neighbors), dtype=np.int64
        )
        for start in range(0, query.shape[0], self.batch_size):
            stop = min(start + self.batch_size, query.shape[0])
            batch = query[start:stop]
            query_norms = np.einsum("ij,ij->i", batch, batch)[:, None]
            squared_distances = (
                query_norms
                + self.train_squared_norms_[None, :]
                - 2.0 * (batch @ self.X_train_.T)
            )
            np.maximum(squared_distances, 0.0, out=squared_distances)

            candidate_indices = np.argpartition(
                squared_distances, kth=max_neighbors - 1, axis=1
            )[:, :max_neighbors]
            candidate_distances = np.take_along_axis(
                squared_distances, candidate_indices, axis=1
            )
            order = np.argsort(candidate_distances, axis=1, kind="stable")
            ordered_indices = np.take_along_axis(candidate_indices, order, axis=1)
            neighbor_labels[start:stop] = self.y_encoded_[ordered_indices]
        return neighbor_labels

    def predict_for_k_values(
        self, X: np.ndarray, k_values: Iterable[int]
    ) -> dict[int, np.ndarray]:
        """Prediz varios valores de k reutilizando uma unica busca de vizinhos."""

        self._check_is_fitted()
        values = tuple(sorted({int(k) for k in k_values}))
        if not values or values[0] <= 0:
            raise ValueError("k_values deve conter inteiros positivos.")
        assert self.X_train_ is not None
        if values[-1] > self.X_train_.shape[0]:
            raise ValueError("Um valor de k excede o numero de amostras de treino.")

        neighbors = self._ordered_neighbor_labels(X, values[-1])
        assert self.classes_ is not None
        predictions: dict[int, np.ndarray] = {}
        for k in values:
            encoded_predictions = np.empty(neighbors.shape[0], dtype=np.int64)
            for row_index, row in enumerate(neighbors[:, :k]):
                votes = np.bincount(row, minlength=self.classes_.size)
                # np.argmax escolhe a primeira posicao em caso de empate, isto e,
                # a menor classe porque np.unique mantem as classes ordenadas.
                encoded_predictions[row_index] = int(np.argmax(votes))
            predictions[k] = self.classes_[encoded_predictions]
        return predictions

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz a classe por votacao majoritaria entre os k vizinhos."""

        return self.predict_for_k_values(X, (self.n_neighbors,))[self.n_neighbors]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Avalia o classificador por acuracia balanceada."""

        return balanced_accuracy_score(y, self.predict(X))


def make_group_folds(
    subjects: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Cria folds de indices mantendo cada sujeito em um unico fold."""

    subject_vector = np.asarray(subjects, dtype=np.int64).reshape(-1)
    if subject_vector.size == 0:
        raise ValueError("subjects nao pode ser vazio.")
    unique_subjects = np.unique(subject_vector)
    if n_splits < 2 or n_splits > unique_subjects.size:
        raise ValueError("n_splits deve estar entre 2 e o numero de sujeitos.")

    shuffled_subjects = unique_subjects.copy()
    np.random.default_rng(seed).shuffle(shuffled_subjects)
    validation_subject_folds = np.array_split(shuffled_subjects, n_splits)

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for validation_subjects in validation_subject_folds:
        validation_mask = np.isin(subject_vector, validation_subjects)
        validation_indices = np.flatnonzero(validation_mask)
        train_indices = np.flatnonzero(~validation_mask)
        folds.append((train_indices, validation_indices))
    return tuple(folds)


@dataclass(frozen=True)
class KNNValidationResult:
    """Resultados da validacao cruzada para todos os valores de k."""

    k_values: tuple[int, ...]
    fold_scores: np.ndarray
    mean_scores: np.ndarray
    standard_deviations: np.ndarray
    best_k: int


def cross_validate_knn(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    *,
    k_values: Iterable[int],
    n_splits: int = 5,
    seed: int = 42,
    batch_size: int = 256,
    standardize: bool = True,
) -> KNNValidationResult:
    """Avalia valores de k com validacao cruzada agrupada por sujeito.

    Quando ``standardize=True``, media e desvio sao recalculados dentro de cada
    fold usando apenas a parte de treinamento daquele fold. Portanto, deve-se
    fornecer preferencialmente os dados ainda nao padronizados por uma etapa
    externa.
    """

    features = _as_feature_matrix(X)
    labels = _as_label_vector(y, features.shape[0])
    subject_vector = np.asarray(subjects, dtype=np.int64).reshape(-1)
    if subject_vector.shape[0] != features.shape[0]:
        raise ValueError("X e subjects devem possuir o mesmo numero de amostras.")

    values = tuple(sorted({int(k) for k in k_values}))
    if not values or values[0] <= 0:
        raise ValueError("k_values deve conter inteiros positivos.")

    folds = make_group_folds(subject_vector, n_splits=n_splits, seed=seed)
    minimum_training_size = min(train_indices.size for train_indices, _ in folds)
    if values[-1] > minimum_training_size:
        raise ValueError(
            "Um valor de k excede o menor conjunto de treinamento dos folds."
        )

    fold_scores = np.empty((len(values), len(folds)), dtype=np.float64)
    for fold_index, (train_indices, validation_indices) in enumerate(folds):
        model = KNNClassifier(
            n_neighbors=values[-1],
            batch_size=batch_size,
            standardize=standardize,
        ).fit(features[train_indices], labels[train_indices])
        predictions = model.predict_for_k_values(features[validation_indices], values)
        for value_index, k in enumerate(values):
            fold_scores[value_index, fold_index] = balanced_accuracy_score(
                labels[validation_indices], predictions[k]
            )

    mean_scores = fold_scores.mean(axis=1)
    standard_deviations = fold_scores.std(axis=1)
    # Em caso de empate, escolhe-se o menor k para manter a regra simples.
    best_index = max(
        range(len(values)),
        key=lambda index: (mean_scores[index], -values[index]),
    )
    return KNNValidationResult(
        k_values=values,
        fold_scores=fold_scores,
        mean_scores=mean_scores,
        standard_deviations=standard_deviations,
        best_k=values[best_index],
    )
