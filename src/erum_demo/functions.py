import numpy as np


def lin_fit(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def exp_fit(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(-b * x) + c


def sigmoid_fit(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a / (1 + np.exp(-(x - b))) + c
