import alphatims.utils
import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.ndimage import maximum_filter
from numba import jit, njit

from numba.typed import List
from typing import Iterable, Literal

# def rms_norm(x):
#     return x/np.std(x)


class CoordsGraph:
    """A class for distance-based graph in high dimensional space"""

    def __init__(
        self,
        coordinates: np.ndarray,  # (n_sample, n_feature)
        tolerance: Iterable[int | float] = None,  # (n_feature,)
        metric: Literal["euclidean", "chebyshev"] = "euclidean",
    ):
        # kd_trees = list(KDTree(points[:, i]) for i in points.shape[1])

        # for i, tree in enumerate(kd_trees):
        #     # intersection of multiple connection list is hard to merge
        #     self.conn_list = tree.query_ball_point(points[:, i], r=tolerance[i], p=2.0)
        #     tree.sparse_distance_matrix()
        if tolerance is not None:
            self.coords = coordinates / tolerance
        else:
            self.coords = coordinates

        tree = KDTree(self.coords)
        if metric == "euclidean":
            p = 2.0
        elif metric == "chebyshev":  # rectangular window
            p = np.inf

        dist_mx = tree.sparse_distance_matrix(tree, max_distance=1, p=p, output_type="coo_matrix")
        # a node can not connect with itself
        dist_mx.data[dist_mx.data == 0] = 1
        dist_mx.setdiag(0)
        self.adjacency_mx = dist_mx != 0

    def __len__(self):
        return self.coords.shape[0]

    def group_nodes(self, breath_first=False, count_threshold=5) -> np.ndarray:

        if breath_first is True:
            search_func = bfs
        else:
            search_func = dfs
        group_labels = search_func(
            n_nodes=len(self),
            indices=self.adjacency_mx.indices,
            indptr=self.adjacency_mx.indptr,
            count_threshold=count_threshold,
        )

        return group_labels


# traverse a graph represented as a sparse matrix
@jit(nopython=True)
def dfs_single(indices, indptr, visited, start):
    stack = [start]
    subgraph = list()
    while stack:
        node = stack.pop()
        if not visited[node]:
            subgraph.append(node)
            visited[node] = True
            #  get neighbors
            for i in indices[indptr[node] : indptr[node + 1]]:
                if not visited[i]:
                    stack.append(i)
    return subgraph  # a Numba list


@jit(nopython=True)
def dfs(n_nodes, indices, indptr, count_threshold=5):
    # buffer for visited flags
    visited = np.zeros(n_nodes, dtype=np.bool_)
    # buffer for group labels
    group_labels = np.zeros(n_nodes, dtype=np.int32)
    current_label = 1
    while not np.all(visited):
        start = np.nonzero(~visited)[0][0]
        stack = [start]
        subgraph = []
        # get one connected component
        while stack:
            node = stack.pop()
            if not visited[node]:
                subgraph.append(node)
                visited[node] = True
                # get neighbors
                for i in indices[indptr[node] : indptr[node + 1]]:
                    if not visited[i]:
                        stack.append(i)
        # filtering by node count of the group
        if len(subgraph) >= count_threshold:
            # fancy indexing not supported, using a loop instead
            for j in subgraph:
                group_labels[j] = current_label
            current_label += 1

    return group_labels


@jit(nopython=True)
def bfs(n_nodes, indices, indptr, count_threshold=5):
    # buffer for visited flags
    visited = np.zeros(n_nodes, dtype=np.bool_)
    # buffer for group labels
    group_labels = np.zeros(n_nodes, dtype=np.int32)
    current_label = 1
    while not np.all(visited):
        start = np.nonzero(~visited)[0][0]
        queue = [start]
        queue_head = 0
        subgraph = []
        # get one connected component
        while queue_head < len(queue):
            # use a pointer to mimic queue.popleft()
            node = queue[queue_head]
            queue_head += 1
            if not visited[node]:
                subgraph.append(node)
                visited[node] = True
                # get neighbors
                for i in indices[indptr[node] : indptr[node + 1]]:
                    if not visited[i]:
                        queue.append(i)
        # filtering by node count of the group
        if len(subgraph) >= count_threshold:
            # fancy indexing not supported, using a loop instead
            for j in subgraph:
                group_labels[j] = current_label
            current_label += 1

    return group_labels


def build_scan_index(scan_lows, scan_highs, scan_max_index):
    """Invert per-peak scan extents into a scan -> peaks lookup.

    Only scans covered by at least one peak are kept, so the integration kernel
    never visits an empty mobility bin.

    :return: ``(covered_scans, scan_indptr, scan_peaks)``, where the peaks
        touching ``covered_scans[i]`` are ``scan_peaks[scan_indptr[i]:scan_indptr[i+1]]``
    :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    lows = np.clip(scan_lows, 0, scan_max_index - 1)
    highs = np.clip(scan_highs, 0, scan_max_index - 1)
    spans = highs - lows + 1

    peaks = np.repeat(np.arange(spans.shape[0], dtype=np.int64), spans)
    # offset within each peak's own scan span
    offsets = np.arange(spans.sum(), dtype=np.int64) - np.repeat(np.cumsum(spans) - spans, spans)
    # union of scan indices of all peaks
    scans = np.repeat(lows, spans) + offsets

    order = np.argsort(scans, kind="stable")
    scans = scans[order]
    peaks = peaks[order]

    covered_scans, starts = np.unique(scans, return_index=True)
    # between scan_indptr[i] and scan_indptr[i]+1: indices of all peaks that cover i-th scan
    scan_indptr = np.append(starts, scans.shape[0])
    return covered_scans, scan_indptr, peaks


@alphatims.utils.pjit
def integrate_peaks(
    frame,
    push_indptr,
    tof_indices,
    intensity_values,
    scan_max_index,
    tof_lows,
    tof_highs,
    covered_scans,
    scan_indptr,
    scan_peaks,
    frame_rows,
    out,
):
    """Sum the raw intensities of one frame into its row of a (pixel, peak) matrix.

    Each peak is a rectangle in (tof, scan) space. Rather than querying the
    whole dataset once per peak, this visits each of the frame's pushes once
    and, for the peaks covering that push's scan, binary-searches the push's
    (sorted) tof indices.

    Decorated with :func:`alphatims.utils.pjit`, so the caller passes an
    iterable of frame indices as the first argument; the frames are then spread
    over threads and reported through alphatims' progress callback. Each frame
    owns exactly one row of `out`, so the accumulation is race-free.

    `frame_rows` maps a frame index to its row in `out`, or -1 to skip it.
    """
    row = frame_rows[frame]
    if row < 0:
        return
    push_base = frame * scan_max_index
    for i in range(covered_scans.shape[0]):
        push = push_base + covered_scans[i] # only lookup scans with peaks detected
        start = push_indptr[push]
        end = push_indptr[push + 1]
        if start == end:
            continue
        for k in range(scan_indptr[i], scan_indptr[i + 1]): # collect intensities for k-th peak
            peak = scan_peaks[k]
            idx = start + np.searchsorted(tof_indices[start:end], tof_lows[peak])
            total = 0.0
            while idx < end and tof_indices[idx] <= tof_highs[peak]:
                total += intensity_values[idx] # collect like scanning
                idx += 1
            out[row, peak] += total


def local_maxima(dense_mx: pd.DataFrame, window_size=[5, 5]) -> pd.Series:
    """Find positions and values of local maxima of an dense array
    `dense_mx` is a (M,N) dataframe so that the positions could be other than ordinal indices

    :param dense_mx: the dense array, with axis domains
    :type dense_mx: pd.DataFrame
    :param window_size: size of the 2D maximum filter, defaults to [5, 5]
    :type window_size: list, optional
    :return: a Series of maxima values, with multiindex of their postions
    :rtype: pd.Series
    """
    if isinstance(dense_mx, pd.DataFrame):
        pass
    else:
        dense_mx = pd.DataFrame(dense_mx)  # if input is without axis domains
    maxima = maximum_filter(dense_mx, size=window_size)  # (M, N)
    maxima = dense_mx.where(
        (dense_mx == maxima) & dense_mx > 0
    )  # (M, N) positions other than local maxima are np.nan
    maxima_pos = maxima.stack()  # (y,x) multiindex peaklist
    return maxima_pos
