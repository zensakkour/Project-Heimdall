"""
Basic solar position estimation (approximate).
"""
from __future__ import annotations

from datetime import datetime
from math import acos, atan2, cos, degrees, radians, sin
from typing import Tuple


def sun_position(dt: datetime, latitude: float, longitude: float) -> Tuple[float, float]:
    """Return (azimuth_deg, elevation_deg) for a naive local datetime.

    This is a lightweight approximation suitable for coarse verification.
    """
    # Convert to fractional year (radians)
    day_of_year = dt.timetuple().tm_yday
    hour = dt.hour + (dt.minute / 60.0) + (dt.second / 3600.0)
    gamma = 2.0 * 3.141592653589793 * (day_of_year - 1 + (hour - 12.0) / 24.0) / 365.0

    # Equation of time (minutes)
    eq_time = 229.18 * (
        0.000075
        + 0.001868 * cos(gamma)
        - 0.032077 * sin(gamma)
        - 0.014615 * cos(2 * gamma)
        - 0.040849 * sin(2 * gamma)
    )

    # Solar declination (radians)
    decl = (
        0.006918
        - 0.399912 * cos(gamma)
        + 0.070257 * sin(gamma)
        - 0.006758 * cos(2 * gamma)
        + 0.000907 * sin(2 * gamma)
        - 0.002697 * cos(3 * gamma)
        + 0.00148 * sin(3 * gamma)
    )

    # Time offset (minutes); assume local time, approximate UTC offset from longitude
    time_offset = eq_time + 4.0 * longitude
    true_solar_minutes = hour * 60.0 + time_offset
    hour_angle = radians((true_solar_minutes / 4.0) - 180.0)

    lat_rad = radians(latitude)
    cos_zenith = sin(lat_rad) * sin(decl) + cos(lat_rad) * cos(decl) * cos(hour_angle)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = acos_safe(cos_zenith)
    elevation = 90.0 - degrees(zenith)

    azimuth = degrees(
        atan2(
            sin(hour_angle),
            cos(hour_angle) * sin(lat_rad) - tan_safe(decl) * cos(lat_rad),
        )
    )
    azimuth = (azimuth + 180.0) % 360.0
    return azimuth, elevation


def acos_safe(value: float) -> float:
    if value < -1.0:
        value = -1.0
    elif value > 1.0:
        value = 1.0
    return acos(value)


def tan_safe(value: float) -> float:
    c = cos(value)
    if abs(c) < 1e-6:
        return 0.0
    return sin(value) / c


