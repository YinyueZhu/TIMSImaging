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
    """A class for distance-based graph in high dimensional space
    """
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

    def group_nodes(self, breath_first=False, count_thrshold=5) -> np.ndarray:

        if breath_first is True:
            search_func = bfs
        else:
            search_func = dfs
        group_labels = search_func(
            n_nodes=len(self),
            indices=self.adjacency_mx.indices,
            indptr=self.adjacency_mx.indptr,
            count_threshold=count_thrshold,
        )

        return group_labels

# traverse a graph represented as a sparse matrix
@jit(nopython=True)
def dfs_single(indices, indptr, visited, start):
    stack = [start]
    subgraph = List()
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
        dense_mx = pd.DataFrame(dense_mx) # if input is without axis domains
    maxima = maximum_filter(dense_mx, size=window_size)  # (M, N) 
    maxima = dense_mx.where((dense_mx == maxima) & dense_mx > 0) # (M, N) positions other than local maxima are np.nan
    maxima_pos = maxima.stack()  # (y,x) multiindex peaklist
    return maxima_pos
