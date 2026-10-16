"""Carregamento e preparacao dos dados UCI HAR usados na Atividade 2.

O modulo preserva o teste oficial e separa a validacao somente entre os
sujeitos do treino oficial. Dessa forma, janelas pertencentes a
uma mesma pessoa nunca aparecem em conjuntos experimentais diferentes.

As duas representacoes solicitadas na atividade sao suportadas:

* ``features``: 561 atributos previamente extraidos pelo conjunto UCI HAR;
* ``raw``: 768 valores obtidos pela concatenacao de seis sinais temporais
  (total_acc_x/y/z e body_gyro_x/y/z), com 128 amostras por sinal.

A padronizacao, quando habilitada, e ajustada exclusivamente no conjunto de
treino e depois aplicada aos conjuntos de validacao e teste.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


Representation = Literal["features", "raw"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "human+activity+recognition+using+smartphones"
    / "UCI HAR Dataset"
    / "UCI HAR Dataset"
)

CLASS_NAMES: tuple[str, ...] = (
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
)
EXPECTED_LABELS = np.arange(1, 7, dtype=np.int64)
EXPECTED_SUBJECTS = np.arange(1, 31, dtype=np.int64)

FEATURE_COUNT = 561
SAMPLES_PER_SIGNAL = 128
RAW_SIGNAL_NAMES: tuple[str, ...] = (
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
)
RAW_FEATURE_COUNT = len(RAW_SIGNAL_NAMES) * SAMPLES_PER_SIGNAL


@dataclass(frozen=True)
class SubjectSplit:
    """Identificadores dos sujeitos usados em cada conjunto experimental."""

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]


@dataclass(frozen=True)
class PreparedData:
    """Dados preparados e metadados necessarios para reproduzir o processo."""

    representation: Representation
    X_train: np.ndarray
    y_train: np.ndarray
    subjects_train: np.ndarray
    X_validation: np.ndarray
    y_validation: np.ndarray
    subjects_validation: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    subjects_test: np.ndarray
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...]
    subject_split: SubjectSplit
    train_mean: np.ndarray | None
    train_scale: np.ndarray | None


def _resolve_dataset_dir(dataset_dir: str | Path | None) -> Path:
    path = DEFAULT_DATASET_DIR if dataset_dir is None else Path(dataset_dir)
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Diretorio do UCI HAR nao encontrado: {path}")
    return path


def _load_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de dados nao encontrado: {path}")
    matrix = np.loadtxt(path, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"Era esperada uma matriz bidimensional em {path}")
    return matrix


def _load_integer_vector(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de dados nao encontrado: {path}")
    return np.asarray(np.loadtxt(path, dtype=np.int64)).reshape(-1)


def _load_targets_and_subjects(
    dataset_dir: Path, partition: Literal["train", "test"]
) -> tuple[np.ndarray, np.ndarray]:
    partition_dir = dataset_dir / partition
    labels = _load_integer_vector(partition_dir / f"y_{partition}.txt")
    subjects = _load_integer_vector(partition_dir / f"subject_{partition}.txt")
    if labels.shape[0] != subjects.shape[0]:
        raise ValueError(
            f"Rotulos e sujeitos possuem numeros diferentes de linhas em {partition}."
        )
    return labels, subjects


def _validate_complete_dataset(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    expected_feature_count: int,
) -> None:
    if X.shape != (y.shape[0], expected_feature_count):
        raise ValueError(
            "Dimensoes inconsistentes: "
            f"X={X.shape}, y={y.shape}, esperado=(*, {expected_feature_count})."
        )
    if subjects.shape[0] != y.shape[0]:
        raise ValueError("O numero de sujeitos nao coincide com o numero de amostras.")
    if not np.isfinite(X).all():
        raise ValueError("A matriz de atributos contem NaN ou infinito.")
    if not np.array_equal(np.unique(y), EXPECTED_LABELS):
        raise ValueError(
            f"Rotulos encontrados {np.unique(y).tolist()}, mas eram esperados 1 a 6."
        )
    if not np.array_equal(np.unique(subjects), EXPECTED_SUBJECTS):
        raise ValueError("O conjunto deve conter exatamente os sujeitos de 1 a 30.")

    for subject in EXPECTED_SUBJECTS:
        subject_labels = np.unique(y[subjects == subject])
        if not np.array_equal(subject_labels, EXPECTED_LABELS):
            raise ValueError(
                f"O sujeito {subject} nao possui amostras das seis atividades."
            )


def _feature_names_from_file(dataset_dir: Path) -> tuple[str, ...]:
    path = dataset_dir / "features.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo de nomes dos atributos nao encontrado: {path}")

    names: list[str] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                names.append(parts[1])

    if len(names) != FEATURE_COUNT:
        raise ValueError(
            f"features.txt possui {len(names)} nomes; eram esperados {FEATURE_COUNT}."
        )
    return tuple(names)


def load_feature_data(
    dataset_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Carrega e reune os 561 atributos das particoes oficiais do UCI HAR."""

    root = _resolve_dataset_dir(dataset_dir)
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    subjects: list[np.ndarray] = []

    for partition in ("train", "test"):
        X_partition = _load_matrix(root / partition / f"X_{partition}.txt")
        y_partition, subject_partition = _load_targets_and_subjects(root, partition)
        if X_partition.shape[0] != y_partition.shape[0]:
            raise ValueError(f"Atributos e rotulos nao estao alinhados em {partition}.")
        matrices.append(X_partition)
        labels.append(y_partition)
        subjects.append(subject_partition)

    X = np.vstack(matrices)
    y = np.concatenate(labels)
    subject_ids = np.concatenate(subjects)
    _validate_complete_dataset(X, y, subject_ids, FEATURE_COUNT)
    return X, y, subject_ids, _feature_names_from_file(root)


def _raw_feature_names() -> tuple[str, ...]:
    return tuple(
        f"{signal}[{sample_index:03d}]"
        for signal in RAW_SIGNAL_NAMES
        for sample_index in range(SAMPLES_PER_SIGNAL)
    )


def load_raw_signal_data(
    dataset_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Carrega os seis sinais ACC/GYR e forma matrizes com 768 atributos.

    O sinal ``total_acc`` e usado para o acelerometro porque corresponde ao
    sinal medido pelo aparelho. ``body_acc`` e uma versao derivada, obtida apos
    a remocao da componente gravitacional.
    """

    root = _resolve_dataset_dir(dataset_dir)
    matrices: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    subjects: list[np.ndarray] = []

    for partition in ("train", "test"):
        signal_dir = root / partition / "Inertial Signals"
        signal_matrices = [
            _load_matrix(signal_dir / f"{signal}_{partition}.txt")
            for signal in RAW_SIGNAL_NAMES
        ]
        row_counts = {matrix.shape[0] for matrix in signal_matrices}
        column_counts = {matrix.shape[1] for matrix in signal_matrices}
        if len(row_counts) != 1 or column_counts != {SAMPLES_PER_SIGNAL}:
            raise ValueError(
                f"Sinais inerciais com dimensoes inconsistentes em {partition}."
            )

        X_partition = np.hstack(signal_matrices)
        y_partition, subject_partition = _load_targets_and_subjects(root, partition)
        if X_partition.shape[0] != y_partition.shape[0]:
            raise ValueError(f"Sinais e rotulos nao estao alinhados em {partition}.")
        matrices.append(X_partition)
        labels.append(y_partition)
        subjects.append(subject_partition)

    X = np.vstack(matrices)
    y = np.concatenate(labels)
    subject_ids = np.concatenate(subjects)
    _validate_complete_dataset(X, y, subject_ids, RAW_FEATURE_COUNT)
    return X, y, subject_ids, _raw_feature_names()


def create_subject_split(
    subjects: np.ndarray,
    *,
    seed: int = 42,
    train_subject_count: int = 20,
    validation_subject_count: int = 4,
    test_subject_count: int = 6,
) -> SubjectSplit:
    """Cria uma divisao reprodutivel e mutuamente exclusiva por sujeito."""

    unique_subjects = np.unique(np.asarray(subjects, dtype=np.int64))
    requested_total = (
        train_subject_count + validation_subject_count + test_subject_count
    )
    if requested_total != unique_subjects.size:
        raise ValueError(
            f"A divisao solicita {requested_total} sujeitos, mas foram encontrados "
            f"{unique_subjects.size}."
        )
    if min(train_subject_count, validation_subject_count, test_subject_count) <= 0:
        raise ValueError("Todos os conjuntos devem receber pelo menos um sujeito.")

    shuffled = unique_subjects.copy()
    np.random.default_rng(seed).shuffle(shuffled)
    train_end = train_subject_count
    validation_end = train_end + validation_subject_count

    return SubjectSplit(
        train=tuple(sorted(int(value) for value in shuffled[:train_end])),
        validation=tuple(
            sorted(int(value) for value in shuffled[train_end:validation_end])
        ),
        test=tuple(sorted(int(value) for value in shuffled[validation_end:])),
    )


def create_official_subject_split(
    dataset_dir: str | Path | None = None,
    *,
    seed: int = 42,
    validation_subject_count: int = 4,
) -> SubjectSplit:
    """Preserva o teste oficial e reserva sujeitos do treino para validacao."""
    root = _resolve_dataset_dir(dataset_dir)
    _, train_subjects = _load_targets_and_subjects(root, "train")
    _, test_subjects = _load_targets_and_subjects(root, "test")
    train_ids = np.unique(train_subjects)
    test_ids = np.unique(test_subjects)
    if np.intersect1d(train_ids, test_ids).size:
        raise ValueError("As particoes oficiais possuem sujeitos em comum.")
    if not 0 < validation_subject_count < train_ids.size:
        raise ValueError("A validacao deve reservar parte dos sujeitos do treino oficial.")
    shuffled = np.random.default_rng(seed).permutation(train_ids)
    split = SubjectSplit(
        train=tuple(sorted(int(x) for x in shuffled[validation_subject_count:])),
        validation=tuple(sorted(int(x) for x in shuffled[:validation_subject_count])),
        test=tuple(int(x) for x in test_ids),
    )
    _validate_subject_split(split, EXPECTED_SUBJECTS)
    return split


def _validate_subject_split(split: SubjectSplit, available_subjects: np.ndarray) -> None:
    train = set(split.train)
    validation = set(split.validation)
    test = set(split.test)
    if train & validation or train & test or validation & test:
        raise ValueError("Ha sujeitos repetidos entre treino, validacao e teste.")
    if train | validation | test != set(int(x) for x in available_subjects):
        raise ValueError("A divisao por sujeito nao cobre exatamente os sujeitos disponiveis.")


def split_by_subject(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    split: SubjectSplit,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Aplica uma divisao por sujeito a matrizes ja alinhadas."""

    if X.shape[0] != y.shape[0] or y.shape[0] != subjects.shape[0]:
        raise ValueError("X, y e subjects devem possuir o mesmo numero de amostras.")
    _validate_subject_split(split, np.unique(subjects))

    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, subject_ids in (
        ("train", split.train),
        ("validation", split.validation),
        ("test", split.test),
    ):
        mask = np.isin(subjects, subject_ids)
        if not mask.any():
            raise ValueError(f"O conjunto {name} ficou sem amostras.")
        result[name] = (X[mask], y[mask], subjects[mask])
    return result


def fit_standardizer(X_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula media e escala usando somente as amostras de treino."""

    if X_train.ndim != 2 or X_train.shape[0] == 0:
        raise ValueError("X_train deve ser uma matriz bidimensional nao vazia.")
    mean = X_train.mean(axis=0)
    scale = X_train.std(axis=0)
    scale = np.where(scale == 0.0, 1.0, scale)
    return mean, scale


def transform_with_standardizer(
    X: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Aplica parametros de padronizacao previamente ajustados."""

    if X.ndim != 2 or mean.shape != (X.shape[1],) or scale.shape != (X.shape[1],):
        raise ValueError("Dimensoes incompativeis para a padronizacao.")
    transformed = (X - mean) / scale
    if not np.isfinite(transformed).all():
        raise ValueError("A padronizacao produziu NaN ou infinito.")
    return transformed


def class_distribution(y: np.ndarray) -> dict[int, dict[str, float | int]]:
    """Retorna contagem e proporcao de cada classe presente em ``y``."""

    labels, counts = np.unique(np.asarray(y, dtype=np.int64), return_counts=True)
    total = counts.sum()
    return {
        int(label): {"count": int(count), "proportion": float(count / total)}
        for label, count in zip(labels, counts, strict=True)
    }


def _validate_class_balance(
    partitions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    y_complete: np.ndarray,
    *,
    tolerance: float,
) -> None:
    if tolerance < 0:
        raise ValueError("A tolerancia de proporcao nao pode ser negativa.")

    global_counts = np.bincount(y_complete, minlength=7)[1:7]
    global_proportions = global_counts / global_counts.sum()

    for name, (_, labels, _) in partitions.items():
        if not np.array_equal(np.unique(labels), EXPECTED_LABELS):
            raise ValueError(f"O conjunto {name} nao contem as seis classes.")
        counts = np.bincount(labels, minlength=7)[1:7]
        proportions = counts / counts.sum()
        max_difference = float(np.max(np.abs(proportions - global_proportions)))
        if max_difference > tolerance:
            raise ValueError(
                f"A distribuicao de classes em {name} difere da global em ate "
                f"{max_difference:.4f}, acima da tolerancia {tolerance:.4f}."
            )


def prepare_data(
    representation: Representation = "features",
    *,
    dataset_dir: str | Path | None = None,
    split_seed: int = 42,
    validation_subject_count: int = 4,
    standardize: bool = True,
    class_proportion_tolerance: float = 0.03,
) -> PreparedData:
    """Executa o fluxo completo de carregamento, divisao e padronizacao.

    Parameters
    ----------
    representation:
        ``"features"`` para os 561 atributos extraidos ou ``"raw"`` para os
        seis sinais temporais concatenados.
    dataset_dir:
        Caminho alternativo para a raiz do ``UCI HAR Dataset``. Quando omitido,
        o caminho e resolvido relativamente ao projeto.
    split_seed:
        Semente usada somente para separar sujeitos do treino oficial.
    validation_subject_count:
        Numero de sujeitos do treino oficial reservados para validacao.
    standardize:
        Se verdadeiro, ajusta media e desvio no treino e transforma os tres
        conjuntos com esses mesmos parametros.
    class_proportion_tolerance:
        Maior desvio absoluto permitido entre a proporcao de uma classe em uma
        particao e sua proporcao no conjunto completo.
    """

    if representation == "features":
        X, y, subjects, feature_names = load_feature_data(dataset_dir)
    elif representation == "raw":
        X, y, subjects, feature_names = load_raw_signal_data(dataset_dir)
    else:
        raise ValueError("representation deve ser 'features' ou 'raw'.")

    subject_split = create_official_subject_split(
        dataset_dir, seed=split_seed,
        validation_subject_count=validation_subject_count,
    )
    partitions = split_by_subject(X, y, subjects, subject_split)
    _validate_class_balance(
        partitions,
        y,
        tolerance=class_proportion_tolerance,
    )

    X_train, y_train, subjects_train = partitions["train"]
    X_validation, y_validation, subjects_validation = partitions["validation"]
    X_test, y_test, subjects_test = partitions["test"]

    train_mean: np.ndarray | None = None
    train_scale: np.ndarray | None = None
    if standardize:
        train_mean, train_scale = fit_standardizer(X_train)
        X_train = transform_with_standardizer(X_train, train_mean, train_scale)
        X_validation = transform_with_standardizer(
            X_validation, train_mean, train_scale
        )
        X_test = transform_with_standardizer(X_test, train_mean, train_scale)

    return PreparedData(
        representation=representation,
        X_train=X_train,
        y_train=y_train,
        subjects_train=subjects_train,
        X_validation=X_validation,
        y_validation=y_validation,
        subjects_validation=subjects_validation,
        X_test=X_test,
        y_test=y_test,
        subjects_test=subjects_test,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        subject_split=subject_split,
        train_mean=train_mean,
        train_scale=train_scale,
    )
