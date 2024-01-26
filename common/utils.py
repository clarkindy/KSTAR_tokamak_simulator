import numpy as np


def ccw(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    return x1 * y2 - y1 * x2


def point_in_polygon(
    x_point: np.ndarray,
    y_point: np.ndarray,
    x_poly: np.ndarray,
    y_poly: np.ndarray,
) -> np.ndarray:
    cond = (y_poly[0] < y_poly[1])[np.newaxis, ...]
    y = np.where(cond, y_poly, y_poly[::-1])[:, np.newaxis, :]
    x = np.where(cond, x_poly, x_poly[::-1])[:, np.newaxis, :]
    y_point_ = y_point[:, np.newaxis]
    x_point_ = x_point[:, np.newaxis]
    dy = y - y_point_[np.newaxis, ...]
    dx = x - x_point_[np.newaxis, ...]
    ccw_ = ccw(dx[0], dy[0], dx[1], dy[1])
    is_on_edge = (
        (ccw_ == 0)
        & (x[0] <= x_point_)
        & (x_point_ <= x[1])
        & (y[0] <= y_point_)
        & (y_point_ <= y[1])
    )
    passed_ccw = (ccw_ >= 0) & ((y[0] <= y_point_) & (y_point_ < y[1]))
    is_crossing = is_on_edge | passed_ccw
    return np.sum(is_crossing, axis=1, dtype=np.int_) % 2 == 1
