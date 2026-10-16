"""Regressao logistica multiclasse com funcao Softmax.

A implementacao segue as decisoes de modelagem da Atividade 2:

* entropia cruzada multiclasse;
* regularizacao L2 somente sobre a matriz de pesos;
* pesos e bias inicializados em zero;
* gradiente descendente full-batch;
* parada por numero maximo de epocas ou tolerancia na variacao da perda.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
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


def softmax(logits: np.ndarray) -> np.ndarray:
    """Calcula probabilidades Softmax de forma numericamente estavel."""

    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("logits deve ser uma matriz bidimensional.")
    shifted = values - values.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def cross_entropy_loss(
    encoded_labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    epsilon: float = 1e-15,
) -> float:
    """Calcula a entropia cruzada media para rotulos codificados de 0 a C-1."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(encoded_labels, dtype=np.int64).reshape(-1)
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.shape[0]:
        raise ValueError("Probabilidades e rotulos possuem dimensoes incompativeis.")
    if labels.size == 0 or labels.min() < 0 or labels.max() >= probabilities.shape[1]:
        raise ValueError("Os rotulos codificados nao correspondem as probabilidades.")

    selected = probabilities[np.arange(labels.shape[0]), labels]
    return float(-np.mean(np.log(np.clip(selected, epsilon, 1.0))))


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


class SoftmaxRegression:
    """Regressao logistica multiclasse treinada por gradiente full-batch."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        l2: float = 0.0,
        max_epochs: int = 1_000,
        tolerance: float = 1e-7,
        verbose: bool = False,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate deve ser positivo.")
        if l2 < 0:
            raise ValueError("l2 nao pode ser negativo.")
        if max_epochs <= 0:
            raise ValueError("max_epochs deve ser positivo.")
        if tolerance < 0:
            raise ValueError("tolerance nao pode ser negativa.")

        self.learning_rate = float(learning_rate)
        self.l2 = float(l2)
        self.max_epochs = int(max_epochs)
        self.tolerance = float(tolerance)
        self.verbose = bool(verbose)

        self.classes_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None
        self.bias_: np.ndarray | None = None
        self.loss_history_: list[float] = []
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.loss_: float | None = None

    def _check_is_fitted(self) -> None:
        if self.weights_ is None or self.bias_ is None or self.classes_ is None:
            raise RuntimeError("O modelo ainda nao foi ajustado.")

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("As classes ainda nao foram definidas.")
        indices = np.searchsorted(self.classes_, y)
        valid = indices < self.classes_.size
        matches = np.zeros(y.shape[0], dtype=bool)
        matches[valid] = self.classes_[indices[valid]] == y[valid]
        if not matches.all():
            raise ValueError("y contem uma classe desconhecida pelo modelo.")
        return indices.astype(np.int64, copy=False)

    def _loss_and_gradients(
        self, X: np.ndarray, encoded_y: np.ndarray
    ) -> tuple[float, np.ndarray, np.ndarray]:
        self._check_is_fitted()
        assert self.weights_ is not None
        assert self.bias_ is not None

        probabilities = softmax(X @ self.weights_ + self.bias_)
        data_loss = cross_entropy_loss(encoded_y, probabilities)
        regularization_loss = 0.5 * self.l2 * float(np.sum(self.weights_**2))

        error = probabilities.copy()
        error[np.arange(X.shape[0]), encoded_y] -= 1.0
        error /= X.shape[0]
        weight_gradient = X.T @ error + self.l2 * self.weights_
        # O bias e deliberadamente excluido da regularizacao.
        bias_gradient = error.sum(axis=0)
        return data_loss + regularization_loss, weight_gradient, bias_gradient

    def fit(self, X: np.ndarray, y: np.ndarray) -> SoftmaxRegression:
        """Ajusta o modelo usando todas as amostras em cada atualizacao."""

        features = _as_feature_matrix(X)
        labels = _as_label_vector(y, features.shape[0])
        self.classes_, encoded_y = np.unique(labels, return_inverse=True)
        if self.classes_.size < 2:
            raise ValueError("O treinamento requer pelo menos duas classes.")

        # Inicializacao em zero, conforme a decisao de modelagem.
        self.weights_ = np.zeros(
            (features.shape[1], self.classes_.size), dtype=np.float64
        )
        self.bias_ = np.zeros(self.classes_.size, dtype=np.float64)
        self.loss_history_ = []
        self.n_iter_ = 0
        self.converged_ = False

        previous_loss: float | None = None
        for epoch in range(1, self.max_epochs + 1):
            loss, weight_gradient, bias_gradient = self._loss_and_gradients(
                features, encoded_y
            )
            if not np.isfinite(loss):
                raise FloatingPointError(
                    "A perda deixou de ser finita; reduza a taxa de aprendizado."
                )
            self.loss_history_.append(loss)

            if previous_loss is not None and abs(previous_loss - loss) <= self.tolerance:
                self.converged_ = True
                break

            self.weights_ -= self.learning_rate * weight_gradient
            self.bias_ -= self.learning_rate * bias_gradient
            self.n_iter_ = epoch
            previous_loss = loss

            if self.verbose and (epoch == 1 or epoch % 100 == 0):
                print(f"epoch={epoch} loss={loss:.8f}")

        self.loss_ = self.loss(features, labels)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Retorna os logits produzidos pelo modelo linear."""

        self._check_is_fitted()
        features = _as_feature_matrix(X)
        assert self.weights_ is not None
        assert self.bias_ is not None
        if features.shape[1] != self.weights_.shape[0]:
            raise ValueError("X possui um numero de atributos diferente do treino.")
        return features @ self.weights_ + self.bias_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna a probabilidade estimada para cada classe."""

        return softmax(self.decision_function(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Prediz os rotulos originais usados no treinamento."""

        self._check_is_fitted()
        assert self.classes_ is not None
        encoded_predictions = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[encoded_predictions]

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calcula entropia cruzada mais L2 para os parametros atuais."""

        self._check_is_fitted()
        features = _as_feature_matrix(X)
        labels = _as_label_vector(y, features.shape[0])
        assert self.weights_ is not None
        encoded_y = self._encode_labels(labels)
        probabilities = self.predict_proba(features)
        return cross_entropy_loss(encoded_y, probabilities) + 0.5 * self.l2 * float(
            np.sum(self.weights_**2)
        )

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Avalia o modelo por acuracia balanceada."""

        return balanced_accuracy_score(y, self.predict(X))


@dataclass(frozen=True)
class SoftmaxTrial:
    """Resultado de uma combinacao de hiperparametros na validacao."""

    learning_rate: float
    l2: float
    max_epochs: int
    tolerance: float
    validation_score: float
    final_loss: float
    epochs_run: int
    converged: bool


@dataclass(frozen=True)
class SoftmaxSearchResult:
    """Melhor modelo e historico completo da busca em grade."""

    best_model: SoftmaxRegression
    best_trial: SoftmaxTrial
    trials: tuple[SoftmaxTrial, ...]


def tune_softmax(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    learning_rates: Iterable[float],
    l2_values: Iterable[float],
    max_epochs_values: Iterable[int],
    tolerance_values: Iterable[float] = (1e-7,),
) -> SoftmaxSearchResult:
    """Seleciona hiperparametros exclusivamente pelo conjunto de validacao."""

    grid = list(
        product(
            tuple(learning_rates),
            tuple(l2_values),
            tuple(max_epochs_values),
            tuple(tolerance_values),
        )
    )
    if not grid:
        raise ValueError("A grade de hiperparametros nao pode ser vazia.")

    models: list[SoftmaxRegression] = []
    trials: list[SoftmaxTrial] = []
    for learning_rate, l2, max_epochs, tolerance in grid:
        model = SoftmaxRegression(
            learning_rate=learning_rate,
            l2=l2,
            max_epochs=max_epochs,
            tolerance=tolerance,
        ).fit(X_train, y_train)
        score = model.score(X_validation, y_validation)
        assert model.loss_ is not None
        trial = SoftmaxTrial(
            learning_rate=float(learning_rate),
            l2=float(l2),
            max_epochs=int(max_epochs),
            tolerance=float(tolerance),
            validation_score=score,
            final_loss=model.loss_,
            epochs_run=model.n_iter_,
            converged=model.converged_,
        )
        models.append(model)
        trials.append(trial)

    best_index = max(
        range(len(trials)),
        key=lambda index: (
            trials[index].validation_score,
            -trials[index].final_loss,
            -trials[index].epochs_run,
        ),
    )
    return SoftmaxSearchResult(
        best_model=models[best_index],
        best_trial=trials[best_index],
        trials=tuple(trials),
    )
