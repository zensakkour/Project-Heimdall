"""
Likelihood models for fusion.
"""
from __future__ import annotations

import math


def gaussian_likelihood(residual: float, sigma: float) -> float:
    if sigma <= 0.0:
        return 0.0
    return math.exp(-0.5 * (residual / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))


def von_mises_likelihood(angle_rad: float, kappa: float) -> float:
    if kappa <= 0.0:
        return 1.0 / (2.0 * math.pi)
    return math.exp(kappa * math.cos(angle_rad)) / (2.0 * math.pi * _besseli0(kappa))


def _besseli0(x: float) -> float:
    # Approximation for modified Bessel function of the first kind (order 0).
    ax = abs(x)
    if ax < 3.75:
        y = (x / 3.75) ** 2
        return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
    y = 3.75 / ax
    return (math.exp(ax) / math.sqrt(ax)) * (
        0.39894228
        + y * (0.01328592 + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377)))))))
    )


