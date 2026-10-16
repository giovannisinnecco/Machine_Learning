import numpy as np


def add_intercept(X):
    """
    Adiciona uma coluna de 1s à matriz de entrada X
    para representar o termo de intercepto w0.
    """
    return np.column_stack([
        np.ones(len(X)),
        X
    ])


def fit_linear_regression(X, y):
    """
    Ajusta um modelo de regressão linear por mínimos quadrados.

    Retorna o vetor theta:
    [w0, w1, ..., wK]
    """
    X_aug = add_intercept(X)

    theta, residuals, rank, singular_values = np.linalg.lstsq(
        X_aug,
        y,
        rcond=None
    )

    return theta


def predict_linear_regression(X, theta):
    """
    Gera previsões usando os parâmetros theta
    previamente aprendidos.
    """
    X_aug = add_intercept(X)

    y_pred = X_aug @ theta

    return y_pred