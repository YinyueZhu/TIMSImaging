import pytest
import numpy as np
import pandas as pd

from timsimaging.utils import CoordsGraph, local_maxima


def gaussian_peak2d(size=100, center=None, sigma=2):
    """Generate a 2D Gaussian image."""
    x = np.linspace(0, size - 1, size)
    y = np.linspace(0, size - 1, size)
    x, y = np.meshgrid(x, y)

    if center is None:
        x0 = y0 = size // 2
    else:
        x0, y0 = center

    gaussian = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))
    return gaussian

def clamp_img(img, threshold=1e-2):
    return np.where(img>threshold, img, 0)


# simulate data wtih 3 peaks in 2D
@pytest.fixture
def simulate_data():
    center1 = (50, 50)
    center2 = (50, 70)
    center3 = (70, 70)
    centers = [center1, center2, center3]
    peaklist = pd.Series([1,2,3], dtype=float, index=pd.MultiIndex.from_tuples(centers, names=["X", "Y"]))
    peak1 = clamp_img(gaussian_peak2d(center=center1))
    peak2 = clamp_img(gaussian_peak2d(center=center2))
    peak3 = clamp_img(gaussian_peak2d(center=center3))
    img = peak1 + 2 * peak2 + 3 * peak3

    label = np.zeros_like(img)
    label[np.where((img == peak1) & (img > 0))] = 1
    label[np.where((img == 2 * peak2) & (img > 0))] = 2
    label[np.where((img == 3 * peak3) & (img > 0))] = 3
    label[label == 0] = np.nan

    return {
        "data": img,
        "peaks": peaklist,
        "label": label,
    }


def test_graph_grouping(simulate_data):
    df = pd.DataFrame(simulate_data["label"])
    df.index.name='Y'
    df.columns.name='X'
    labels = df.stack().to_numpy(dtype=int)
    coords = df.stack().reset_index()[['X', 'Y']]
    graph = CoordsGraph(coords, tolerance=2, metric="chebyshev")
    assert np.all(labels == graph.group_nodes())

def test_local_maxima(simulate_data):
    df = pd.DataFrame(simulate_data["data"])
    df.index.name='Y'
    df.columns.name='X'    
    maxima = local_maxima(df)
    maxima = maxima.reset_index().set_index(["X", "Y"])
    
    expected = simulate_data["peaks"]
    expected = expected.reset_index().set_index(["X", "Y"])
    assert maxima.equals(expected)
