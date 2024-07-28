import alphatims.utils
import alphatims.bruker

import numpy as np
import pandas as pd
import sqlite3
import os

from collections import deque
from scipy.ndimage import maximum_filter, minimum_filter1d
from scipy.spatial import KDTree
from typing import List, Iterable, Literal

from bokeh.plotting import show
from .plotting import spectrum, mobilogram, heatmap, image

__all__ = ["MSIDataset", "IntensityArray"]

class MSIDataset:
    """The class for a MSI dataset"""

    def __init__(self, path: str):
        """

        :param path: path of the .d directory
        :type path: str
        """
        self.data = alphatims.bruker.TimsTOF(path, use_hdf_if_available=False)
        # not inherit TimsTOF directly for future detachment

        # read pixel coordinates
        con = sqlite3.connect(os.path.join(path, "analysis.tdf"))

        self.pos = pd.read_sql("SELECT * FROM MaldiFrameInfo", con)[
            ["Frame", "XIndexPos", "YIndexPos"]
        ]
        # imaging resolution in μm
        img_res = pd.read_sql("SELECT * FROM MaldiFrameLaserInfo", con)["SpotSize"][0]
        mz_res = (self.data.mz_max_value - self.data.mz_min_value) / len(
            self.data.mz_values
        )
        mob_res = (self.data.mobility_max_value - self.data.mobility_min_value) / len(
            self.data.mobility_values
        )

        # self.resolution = [mz_res, mob_res]
        self.resolution = {"xy": img_res, "mz": mz_res, "1/K0": mob_res}

    # get one frame
    def __getitem__(self, index):
        return IntensityArray(self.data[index])

    # get metadata
    # def __getattribute__(self, name):
    #     return self.data.__getattribute__(name)

    def tic(self):
        # issue: the ratio is right, abs value greater
        total_intensities = self.data.frames["SummedIntensities"][1:]
        return total_intensities

    def mean_spectra(self, as_frame=False, intensity_threshold=0.05):
        """compute mean spectra over the whole dataset

        :param as_frame: if True, return a pd.DataFrame, otherwise an IntensityArray, defaults to False
        :type as_frame: bool, optional
        :param intensity_threshold: Filter out intensities that appear in little fraction of pixels, defaults to 0.05
        :type intensity_threshold: float, optional
        :return: _description_
        :rtype: _type_
        """
        # intensities = pd.Series(
        #     self.data.intensity_values,
        #     name="intensity_values",
        #     dtype=np.uint32,
        # )  # np.ndarray[np.uint16]
        # tof_indices = self.data.tof_indices  # np.ndarray[np.uint32]
        # # decompress scan indeices in CSR format
        # indptr = alphatims.bruker.indptr_lookup(
        #     self.data.push_indptr, np.arange(len(self.data), dtype=np.uint32)
        # )
        # # frame_indices = indptr // self.data.scan_max_index
        # scan_indices = (indptr % self.data.scan_max_index).astype(np.uint16)

        # grouped = intensities.groupby([tof_indices, scan_indices], sort=False)
        # mean_spec = grouped.sum(
        #     engine="numba",
        #     engine_kwargs={"nopython": True, "nogil": False, "parallel": False},
        # ) / (self.data.frame_max_index - 1)
        # # there's a dummy frame
        # mean_spec.index.names = ["tof_indices", "scan_indices"]

        sum_mx = self.data.bin_intensities(
            np.arange(len(self.data)), axis=["mz_values", "mobility_values"]
        )
        if intensity_threshold is not None:
            intensity_cut = (
                self.data.intensity_min_value
                * (self.data.frame_max_index - 1)
                * intensity_threshold
            )
            tof_indices, scan_indices = np.nonzero(sum_mx > intensity_cut)
        else:
            tof_indices, scan_indices = sum_mx.nonzero()

        mean_spec = pd.DataFrame(
            {
                "tof_indices": tof_indices,
                "scan_indices": scan_indices,
                "intensity_values": sum_mx[tof_indices, scan_indices]
                / (self.data.frame_max_index - 1),
            }
        )

        # # filter out low frequent intensities
        # if intensity_threshold is not None:
        #     mean_spec = mean_spec.loc[
        #         lambda x: x >= self.data.intensity_min_value * intensity_threshold
        #     ]  # need improvement

        if as_frame:
            return mean_spec
        else:
            return IntensityArray(
                mean_spec,
                mz_domain=self.data.mz_values,
                mobility_domain=self.data.mobility_values,
            )

    def image(self):
        f, _ = image(self)
        show(f)


class IntensityArray:
    """The class for a frame."""

    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        mz_domain: np.ndarray | None = None,
        mobility_domain: np.ndarray | None = None,
    ):
        """

        :param data: 3 columns for 3 dimensions: mz, mobility, intensity
        :type data: pd.DataFrame | pd.Series
        :param mz_domain: a sorted list of all possible mz values, defaults to None
        :type mz_domain: np.ndarray | None, optional
        :param mobility_domain: a sorted list of all possible 1/K0 values, defaults to None
        :type mobility_domain: np.ndarray | None, optional
        :raises TypeError: domains are missing
        """
        # initialized from a dataframe with mz, mobility and intensity columns
        if isinstance(data, pd.Series):
            data = data.reset_index()
        if isinstance(data, pd.DataFrame):
            try:
                self.data = pd.DataFrame(
                    data[
                        [
                            "tof_indices",
                            "scan_indices",
                            "mz_values",
                            "mobility_values",
                            "intensity_values",
                        ]
                    ]
                )
                self.idx_available = True
            # only values provided
            except KeyError:
                try:
                    self.data = pd.DataFrame(
                        data[
                            [
                                "mz_values",
                                "mobility_values",
                                "intensity_values",
                            ]
                        ]
                    )
                    if mz_domain is not None and mobility_domain is not None:
                        self.data["tof_indices"] = np.searchsorted(
                            mz_domain, data["mz_values"]
                        )
                        self.data["scan_indices"] = np.searchsorted(
                            mobility_domain, data["mobility_values"]
                        )
                        self.idx_available = True
                    else:
                        self.idx_available = False
                # only indices and domain provided
                except KeyError:
                    self.data = pd.DataFrame(
                        data[
                            [
                                "tof_indices",
                                "scan_indices",
                                "intensity_values",
                            ]
                        ]
                    )
                    if mz_domain is not None and mobility_domain is not None:
                        self.data["mz_values"] = mz_domain[data["tof_indices"]]
                        self.data["mobility_values"] = mobility_domain[
                            data["scan_indices"]
                        ]
                        self.idx_available = True
                    else:
                        raise TypeError("Invalid domains")

        # compute resolution(minimium stride) in each dimension
        self.resolution = (
            self.data[["mz_values", "mobility_values"]]
            .apply(lambda x: np.min(np.diff(np.unique(x))))
            .to_numpy()
        )

    # for console display
    def as_series(self, use_index=False, sort=True) -> pd.Series:
        """Represent the frame as a multiindexed pd.Series

        :param use_index: if True the multiindex uses integer indices, otherwise uses values, defaults to False
        :type use_index: bool, optional
        :param sort: sort the Series by multiindex levels, defaults to True
        :type sort: bool, optional
        :return: A pd.Series of intensities, with (mz, mobility) multiindex
        :rtype: pd.Series
        """
        if use_index:
            series = self.data.set_index(["tof_indices", "scan_indices"])[
                "intensity_values"
            ]
        else:
            series = self.data.set_index(["mz_values", "mobility_values"])[
                "intensity_values"
            ]
        if sort == True:
            series = series.groupby(series.index.names).sum()
        return series

    def __repr__(self):
        return self.as_series().__repr__()

    def __len__(self):
        return self.data.shape[0]

    def _build_graph(self, **kwargs):
        return IntensityGraph(self, **kwargs)

    def peakPick(
        self,
        tolerance: Iterable[int | float] | int | float | None = 2,
        metric: Literal["euclidean", "chebyshev"] = "euclidean",
        window_size: Iterable[int] = [17, 7],
        count_thrshold=5,  # at least 5 points for a 3D peak
        sort=False,
        return_labels=False,
        return_extents=False,
    ) -> pd.DataFrame:
        """2D peak-picking on a frame
        First group intensities based on approximity in (mz, mobility) space, then detect local maxima in each group

        :param tolerance: tolerance to determine neighbors, in integer indices, defaults to [2,2]
        :type tolerance: Iterable[int  |  float] | int | float | None, optional
        :param metric: distance metric, defaults to "euclidean"
        :type metric: Literal[&quot;euclidean&quot;, &quot;chebyshev&quot;], optional
        :param window_size: window size of the maximum filter, defaults to [13, 5]
        :type window_size: Iterable[int], optional
        :param count_thrshold: minimum intensity count of a peak, defaults to 5
        :type count_thrshold: int, optional
        :param sort: if True, sort peaks by descending total intensity, defaults to False
        :type sort: bool, optional
        :param return_labels: if True, return group labels, defaults to False
        :type return_labels: bool, optional
        :param return_extents: if True, return extents in each dimension of groups, defaults to False
        :type return_extents: bool, optional
        :return: peak information
        :rtype: pd.DataFrame
        """

        graph = self._build_graph(tolerance=tolerance, metric=metric)
        group_labels = graph.group_nodes(count_thrshold)  # ndarray of (k,)

        intensity_groups = self.data[group_labels > 0].groupby(
            group_labels[group_labels > 0], group_keys=True
        )  # filter, then group

        peak_labels = np.zeros_like(group_labels)
        current_group = 1
        for i, g in intensity_groups:
            dense_mx = g.reset_index().pivot(
                index="scan_indices", columns="tof_indices", values="intensity_values"
            )  # create dense matrix for each group
            # dense_mx = dense_mx.reindex(
            #     index=np.arange(
            #         np.min(dense_mx.index), np.max(dense_mx.index) + 1
            #     )[::-1],
            #     columns=np.arange(
            #         np.min(dense_mx.columns), np.max(dense_mx.columns) + 1
            #     ),
            # )
            dense_mx.fillna(0, inplace=True)
            maxima = maximum_filter(
                dense_mx, size=window_size
            )  # row index is y and col is x
            # find local maxima
            peaks = dense_mx.where((dense_mx == maxima) & dense_mx > 0).stack()
            if peaks.shape[0] > 1:  # isomers
                peaks = peaks.reset_index()
                # is mz resolvable?
                # tof_indices are stored as unsigned integer
                if (
                    np.max(peaks["tof_indices"]) - np.min(peaks["tof_indices"])
                ) <= tolerance:
                    proj = np.sum(dense_mx, axis=1)
                    minima = minimum_filter1d(proj, size=17)
                    # , mode="constant", cval=0)
                    # saddle point
                    split = proj[proj == minima].index.to_numpy()

                    for s in range(len(split) - 1):
                        subgroup = g.loc[
                            lambda df: (df["scan_indices"] >= split[s])
                            & (df["scan_indices"] < split[s + 1])
                        ]
                        peak_labels[subgroup.index.to_numpy()] = current_group
                        current_group += 1
                else:
                    proj = np.sum(dense_mx, axis=0)
                    minima = minimum_filter1d(proj, size=5)
                    split = proj[proj == minima].index.to_numpy()
                    for s in range(len(split) - 1):
                        subgroup = g.loc[
                            lambda df: (df["tof_indices"] >= split[s])
                            & (df["tof_indices"] < split[s + 1])
                        ]
                        peak_labels[subgroup.index.to_numpy()] = current_group
                        current_group += 1
            else:
                peak_labels[g.index.to_numpy()] = current_group
                current_group += 1

        peak_groups = self.data[peak_labels > 0].groupby(
            peak_labels[peak_labels > 0], group_keys=True
        )

        # intensity-weighted mz and mob
        peak_list = peak_groups.apply(
            lambda df: df[["mz_values", "mobility_values"]].apply(
                np.average, weights=df["intensity_values"]
            )
        )
        peak_list["total_intensity"] = peak_groups["intensity_values"].sum()
        if sort:
            peak_list.sort_values("total_intensity", ascending=False, inplace=True)

        results = (peak_list,)
        # include group labels
        if return_labels is True:
            results += (peak_labels,)
        # include extents of each peak
        if return_extents is True:
            peak_extents = peak_groups[["mz_values", "mobility_values"]].agg(
                [np.min, np.max]
            )
            results += (peak_extents,)

        return results

    # plotting methods
    def spectrum(self):
        data1d = self.data.groupby("mz_values")["intensity_values"].sum().reset_index()
        f, _ = spectrum(data1d)
        show(f)

    def mobilogram(self):
        data1d = (
            self.data.groupby("mobility_values")["intensity_values"].sum().reset_index()
        )
        f, _ = mobilogram(data1d)
        show(f)

    def heatmap(self):
        f, _ = heatmap(self)
        show(f)


class IntensityGraph:
    """The class for clustering intensities, where nodes are intensities, edges(connectivity) are determined
    by their distances in the (mz, mobility) space, then cluster them by connectivity.
    """

    def __init__(
        self,
        arr: IntensityArray,
        tolerance: Iterable[int | float] | int | float | None = 2,
        metric: Literal["euclidean", "chebyshev"] = "euclidean",
    ):
        """Build a graph from IntensityArray

        :param arr: a frame to cluster
        :type arr: IntensityArray
        :param tolerance: Distance tolerances for connectivity at each dimension, defaults to 2
        :type tolerance: Iterable[int  |  float] | int | float | None, optional
        :param metric: distance metric, defaults to "euclidean"
        :type metric: Literal[&quot;euclidean&quot;, &quot;chebyshev&quot;], optional
        """

        if arr.idx_available is True:
            coords = arr.data[["tof_indices", "scan_indices"]].to_numpy(
                dtype=np.float64, copy=True
            )
            # normalize coordinates, decision boundary ellipse->circle
            if tolerance is not None:
                coords /= tolerance

        self.nodes = arr.data["intensity_values"]

        tree = KDTree(coords)
        # get the connection list, where i-th is a list of all neighbors represented in INDEX
        if metric == "euclidean":
            self.conn_list = tree.query_ball_point(coords, r=1, p=2.0)

        elif metric == "chebyshev":  # rectangular window
            self.conn_list = tree.query_ball_point(coords, r=1, p=np.inf)

    def __len__(self):
        return self.nodes.shape[0]

    def _dfs(self, graph: set[int]) -> set[int]:
        """Tranverse a graph from a start node using depth-first search,
        return visited node indices

        :param graph: a set of node indices
        :type graph: set[int]
        :return: the node set of a subgraph
        :rtype: set[int]
        """
        # problem: larger radius make it much slower
        start = graph.pop()  # graph is modified, bug-prone
        stack = [start]
        visited = set()
        while stack:
            node = stack.pop()
            if node not in visited:
                # there are repeated nodes in the stack if a cycle exists
                visited.add(node)
                appr = self.conn_list[node]
                for i in appr:
                    if i not in visited:
                        stack.append(i)
        return visited

    # breath-first search
    def _bfs(self, graph: set) -> set:
        """Tranverse a graph from a start node using breadth-first search,
        more efficient when node degree is higher

        :param graph: a set of node indices
        :type graph: set[int]
        :return: the node set of a subgraph
        :rtype: set[int]
        """
        start = graph.pop()
        queue = deque([start])
        visited = set()
        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                appr = self.conn_list[node]
                for i in appr:
                    if i not in visited:
                        queue.append(i)
        return visited

    def group_nodes(self, count_thrshold=5, breadth_first=True) -> np.ndarray:
        """Group nodes by connectivity

        :param count_thrshold: groups smaller than the threshold would be ignored and labeled as group 0, defaults to 5
        :type count_thrshold: int, optional
        :return: an array of group labels
        :rtype: np.ndarray
        """
        graph = set(np.arange(len(self)))
        group_labels = np.zeros(len(self), dtype=int)
        current_group = 1

        if breadth_first:
            search_func = self._bfs
        else:
            search_func = self._dfs
        # grouped_coords = {}

        while graph:
            subgraph = search_func(graph)

            graph -= subgraph

            if len(subgraph) >= count_thrshold:  # filter out all fragments
                # grouped_coords[current_group] = subgraph
                group_labels[list(subgraph)] = current_group
                current_group += 1

        return group_labels
