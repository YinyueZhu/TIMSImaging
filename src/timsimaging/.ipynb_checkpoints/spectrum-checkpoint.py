import alphatims.utils
import alphatims.bruker

import struct
import numpy as np
import pandas as pd
import sqlite3
import os

from ripser import ripser
from scipy.ndimage import maximum_filter, minimum_filter1d
from typing import List, Iterable, Literal, Dict
from pyimzml.ImzMLWriter import ImzMLWriter
from pyimzml.compression import NoCompression, ZlibCompression


from bokeh.plotting import show
from .utils import CoordsGraph
from .plotting import spectrum, mobilogram, heatmap, image, _visualize
__all__=["MSIDataset", "Frame", "export_imzML"]

class MSIDataset:
    """The class for a raw MSI dataset"""

    def __init__(self, path: str):
        """

        :param path: path of the .d directory
        :type path: str
        """
        self.data = alphatims.bruker.TimsTOF(path, use_hdf_if_available=False)
        # not inherit TimsTOF directly for future detachment

        # parse .tdf SQL file
        with sqlite3.connect(os.path.join(path, "analysis.tdf")) as con:
            # read pixel coordinates
            self.pos = pd.read_sql("SELECT * FROM MaldiFrameInfo", con)[
                ["Frame", "XIndexPos", "YIndexPos"]
            ]
            # imaging resolution in μm
            img_res = pd.read_sql("SELECT * FROM MaldiFrameLaserInfo", con)["SpotSize"][0]
            # mass and ion mobility calibration info
            self.cali_info = pd.read_sql("SELECT * FROM CalibrationInfo", con).set_index("KeyName")

        mz_res = (self.data.mz_max_value - self.data.mz_min_value) / len(self.data.mz_values)
        mob_res = (self.data.mobility_max_value - self.data.mobility_min_value) / len(
            self.data.mobility_values
        )
        self.resolution = {"xy": img_res, "mz": mz_res, "1/K0": mob_res}
        # calibration info

    # get one frame
    def __getitem__(self, index):
        return Frame(self.data[index])

    def ccs_calibrator(self):
        """Generate a CCS calibrater, mapping 1/K_0 -> CCS

        :return: calibrator
        :rtype: CCS_calibration
        """
        from .calibration import CCS_calibration

        polarity = self.cali_info.loc[0, "KeyPolarity"]
        # calibrants in chemical formula
        calibrants = self.cali_info.at["ReferenceMobilityPeakNames", "Value"].decode()
        calibrants = calibrants.split("\x00")[:-1]
        # x
        raw_mob = struct.unpack(
            f"{len(calibrants)}d",
            self.cali_info.at["MobilitiesPreviousCalibration", "Value"],
        )
        # model x->y
        calibrator = CCS_calibration(calibrants, raw_mob, polarity)
        return calibrator

    def tic(self) -> pd.Series:
        # is already summarized in .tdf
        total_intensities = self.data.frames["SummedIntensities"][1:]
        return total_intensities

    def mean_spectrum(
        self,
        frame_indices=None,
        sampling_ratio: float = 1.0,
        intensity_threshold: float = 0.05,
        as_frame=False,
    ):
        """compute mean spectra over the whole dataset

        :param as_frame: if True, return a pd.DataFrame, otherwise an Frame, defaults to False
        :type as_frame: bool, optional
        :param intensity_threshold: Filter out intensities that appear in little fraction of pixels, defaults to 0.05
        :type intensity_threshold: float, optional
        :return: _description_
        :rtype: _type_
        """

        if sampling_ratio == 1:
            intensity_indices = np.arange(len(self.data))
            n_frame = self.data.frame_max_index - 1
        # randomly pick frames out of n
        elif sampling_ratio < 1:
            n_frame = int(self.data.frame_max_index * sampling_ratio)
            frame_indices = np.random.choice(
                np.arange(1, self.data.frame_max_index),
                size=n_frame,
            )
            intensity_indices = self.data[frame_indices, "raw"]

        sum_mx = self.data.bin_intensities(intensity_indices, axis=["mz_values", "mobility_values"])
        if intensity_threshold is not None:
            intensity_cut = self.data.intensity_min_value * n_frame * intensity_threshold
            tof_indices, scan_indices = np.nonzero(sum_mx > intensity_cut)
        else:
            tof_indices, scan_indices = sum_mx.nonzero()

        mean_spec = pd.DataFrame(
            {
                "tof_indices": tof_indices,
                "scan_indices": scan_indices,
                "intensity_values": sum_mx[tof_indices, scan_indices] / n_frame,
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
            return Frame(
                mean_spec,
                mz_domain=self.data.mz_values,
                mobility_domain=self.data.mobility_values,
            )

    def process(self, sampling_ratio=0.1, intensity_threshold=0.05, visualize=False, 
    ccs_calibration=False, verbose=False, **kwargs) -> Dict:
        """Process the dataset to peak picked and aligned data cube

        :param sampling_ratio: ratio for computing the mean spectrum, defaults to 0.1
        :type sampling_ratio: float, optional
        :param intensity_threshold: intensity threshold relative to lowest intensity in a single frame for filtering the mean spectrum, defaults to 0.05
        :type intensity_threshold: float, optional
        :return: a dictionary with intensity array, peak list and coordinates
        :rtype: Dict
        """
        # peak picking
        mean_spec = self.mean_spectrum(
            sampling_ratio=sampling_ratio, intensity_threshold=intensity_threshold
        )
        peak_list, peak_extents = mean_spec.peakPick(return_extents=True, **kwargs)
        n_peak = peak_list.shape[0]

        # use dataframe for missing values
        intensity_array = pd.DataFrame(
            None,
            index=range(1, self.data.frame_max_index),
            columns=range(1, n_peak + 1),
        )
        for i in range(n_peak):
            mz_min, mz_max, mob_min, mob_max = peak_extents.iloc[i]
            # all data for i-th peak
            image_data = self.data[:, mob_min:mob_max, 0, mz_min:mz_max]
            intensity_array[i + 1] = image_data.groupby("frame_indices")["intensity_values"].sum()

        intensity_array.fillna(0.0, inplace=True)

        results = {
            "coords": self.pos,
            "peak_list": peak_list,
            "intensity_array": intensity_array,
        }  # 3 dataframes
        if visualize is True:
            app = _visualize(self, mean_spec, peak_list, peak_extents)
            results["viz"]=app

        if ccs_calibration is True:
            calibrator = self.ccs_calibrator()
            # assume charge is 1, true for most ions in MALDI
            # add isotope envelope decection for more accurate computation
            ccs_values = calibrator.transform(
                peak_list["mz_values"], peak_list["mobility_values"], charge=1
            )
            results["peak_list"]["ccs_values"] = ccs_values
            results["ccs_calibrator"] = calibrator
            
        return results

    def image(self):
        f, _ = image(self)
        show(f)


class Frame:
    """The class for a frame."""

    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        mz_domain: np.ndarray | None = None,
        mobility_domain: np.ndarray | None = None,
        coords: List[int] = None,
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
                        self.data["tof_indices"] = np.searchsorted(mz_domain, data["mz_values"])
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
                        self.data["mobility_values"] = mobility_domain[data["scan_indices"]]
                        self.idx_available = True
                    else:
                        raise TypeError("Invalid domains")

        self.coords = coords
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
            series = self.data.set_index(["tof_indices", "scan_indices"])["intensity_values"]
        else:
            series = self.data.set_index(["mz_values", "mobility_values"])["intensity_values"]
        if sort == True:
            series = series.groupby(series.index.names).sum()
        return series

    def __repr__(self):
        return self.as_series().__repr__()

    def __len__(self):
        return self.data.shape[0]

    @property
    def mz(self):
        return self.data["mz_values"].to_numpy()

    @property
    def mobility(self):
        return self.data["mobility_values"].to_numpy()

    @property
    def intensity(self):
        return self.data["intensity_values"].to_numpy()

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

        if self.idx_available is True:
            coords = self.data[["tof_indices", "scan_indices"]].to_numpy(
                dtype=np.float64, copy=True
            )
        graph = CoordsGraph(coordinates=coords, tolerance=tolerance, metric=metric)
        group_labels = graph.group_nodes(count_thrshold)  # ndarray of (k,)
        # filter off intensities with group label=0
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
            maxima = maximum_filter(dense_mx, size=window_size)  # row index is y and col is x
            # find local maxima
            peaks = dense_mx.where((dense_mx == maxima) & dense_mx > 0).stack()
            if peaks.shape[0] > 1:  # isomers
                peaks = peaks.reset_index()
                # is mz resolvable?
                # tof_indices are stored as unsigned integer
                if (np.max(peaks["tof_indices"]) - np.min(peaks["tof_indices"])) <= tolerance:
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
                # separate peaks by mz values
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
            peak_extents = peak_groups[["mz_values", "mobility_values"]].agg(["min", "max"])
            results += (peak_extents,)

        return results

    # plotting methods
    def spectrum(self, plotting=True):
        data1d = self.data.groupby("mz_values")["intensity_values"].sum().reset_index()
        if plotting is True:
            f, _ = spectrum(data1d)
            show(f)
        else:
            return data1d

    def mobilogram(self, plotting=True):
        data1d = self.data.groupby("mobility_values")["intensity_values"].sum().reset_index()
        if plotting is True:
            f, _ = mobilogram(data1d)
            show(f)
        else:
            return data1d

    def heatmap(self, plotting=True):
        f, _ = heatmap(self)
        show(f)

    def lower_star_filter(self):
        pass



def export_imzML(
    dataset: MSIDataset,
    path: str,
    peaks: Dict = None,
    mode="centroid",
    imzml_mode="continuous",
):

    key_polarity = dataset.cali_info["KeyPolarity"].iloc[0]
    if key_polarity == "+":
        polarity = "positive"
    elif key_polarity == "-":
        polarity = "negative"
    compression_object = NoCompression()
    # create imzML and ibd files
    writer = ImzMLWriter(
        path,
        polarity=polarity,
        mode=imzml_mode,
        spec_type=mode,
        mz_dtype=np.float64,
        intensity_dtype=np.float64,
        mobility_dtype=np.float64,
        mz_compression=compression_object,
        intensity_compression=compression_object,
        mobility_compression=compression_object,
        include_mobility=True,
    )
    if peaks is None:
        mean_spec = dataset.mean_spectra(intensity_threshold=0.05)
        peak_list, peak_extents = mean_spec.peakPick(
            return_extents=True,
        )
        # get peak picked frames

        intensity_arrays = pd.DataFrame(index=np.arange(1, dataset.data.frame_max_index))
        # can I use pure Numpy here?
        for i in range(peak_extents.shape[0]):
            mz_min, mz_max, mob_min, mob_max = peak_extents.iloc[i]
            # all data for i-th peak
            image_data = dataset.data[:, mob_min:mob_max, 0, mz_min:mz_max]
            intensity_arrays[i + 1] = image_data.groupby("frame_indices")["intensity_values"].sum()
        intensity_array = intensity_arrays.copy().fillna(0)

    else:
        peak_list = peaks["peak_list"]
        intensity_array = peaks["intensity_array"]
    pos = dataset.pos.set_index("Frame")

    # write files
    for frame in np.arange(1, dataset.data.frame_max_index):
        writer.addSpectrum(
            mzs=peak_list["mz_values"],
            intensities=intensity_array.loc[frame],
            mobilities=peak_list["mobility_values"],
            coords=pos.loc[frame],
        )
    writer.close()
