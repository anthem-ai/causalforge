"""
Code Skeleton from:  https://github.com/vanderschaarlab/mlforhealthlabpub/blob/main/alg/ganite/ganite/utils/metrics.py
"""

import numpy as np
from scipy import stats


def sqrt_PEHE(y: np.ndarray, hat_y: np.ndarray) -> float:
    """
    Precision in Estimation of Heterogeneous Effect(Numpy version).
    PEHE reflects the ability to capture individual variation in treatment effects.
    Args:
        y: expected outcome.
        hat_y: estimated outcome.
    """
    return np.sqrt(np.mean(((y[:, 1] - y[:, 0]) - (hat_y[:, 1] - hat_y[:, 0]) ** 2)))


def sqrt_PEHE_with_diff(y: np.ndarray, hat_y: np.ndarray) -> float:
    """
    Precision in Estimation of Heterogeneous Effect(Numpy version).
    PEHE reflects the ability to capture individual variation in treatment effects.
    Args:
        y: expected outcome.
        hat_y: estimated outcome difference.
    """
    return np.sqrt(np.mean(((y[:, 1] - y[:, 0]) - hat_y) ** 2))


def PEHE_with_ite(ite: np.ndarray, 
                  hat_ite: np.ndarray,
                  sqrt=True) -> float:
    """
    Precision in Estimation of Heterogeneous Effect(Numpy version).
    PEHE reflects the ability to capture individual variation in treatment effects.
    Args:
        ite: expected ITEs
        hat_ite: estimated ITEs
        sqrt: square
    """
    PEHE = np.mean((ite - hat_ite) ** 2)
    if sqrt:
        return np.sqrt(PEHE)
    else:
        return PEHE


def RPol(t: np.ndarray, y: np.ndarray, hat_y: np.ndarray) -> np.ndarray:
    """
    Policy risk(RPol).
    RPol is the average loss in value when treating according to the policy implied by an ITE estimator.
    Args:
        t: treatment vector.
        y: expected outcome.
        hat_y: estimated outcome.
    Output:

    """
    hat_t = np.sign(hat_y[:, 1] - hat_y[:, 0])
    hat_t = 0.5 * (hat_t + 1)
    new_hat_t = np.abs(1 - hat_t)

    # Intersection
    idx1 = hat_t * t
    idx0 = new_hat_t * (1 - t)

    # risk policy computation
    RPol1 = (np.sum(idx1 * y) / (np.sum(idx1) + 1e-8)) * np.mean(hat_t)
    RPol0 = (np.sum(idx0 * y) / (np.sum(idx0) + 1e-8)) * np.mean(new_hat_t)

    return 1 - (RPol1 + RPol0)


def eps_ATE(y: np.ndarray, hat_y: np.ndarray) -> np.ndarray:
    """
    Average Treatment Effect.
    ATE measures what is the expected causal effect of the treatment across all individuals in the population.
    Args:
        y: expected outcome.
        hat_y: estimated outcome.
    """
    return np.abs(np.mean(y[:, 1] - y[:, 0]) - np.mean(hat_y[:, 1] - hat_y[:, 0]))


def eps_ATE_diff(ite: np.ndarray, hat_ite: np.ndarray) -> np.ndarray:
    return np.abs(np.mean(ite) - np.mean(hat_ite))


def ATT(t: np.ndarray, y: np.ndarray, hat_y: np.ndarray) -> np.ndarray:
    """
    Average Treatment Effect on the Treated(ATT).
    ATT measures what is the expected causal effect of the treatment for individuals in the treatment group.
    Args:
        t: treatment vector.
        y: expected outcome.
        hat_y: estimated outcome.
    """
    # Original ATT
    ATT_value = np.sum(t * y) / (np.sum(t) + 1e-8) - np.sum((1 - t) * y) / (
        np.sum(1 - t) + 1e-8
    )
    # Estimated ATT
    ATT_estimate = np.sum(t * (hat_y[:, 1] - hat_y[:, 0])) / (np.sum(t) + 1e-8)
    return np.abs(ATT_value - ATT_estimate)

