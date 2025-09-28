import alphatims.utils
import alphatims.bruker

import struct
import numpy as np
import pandas as pd
import sqlite3
import os
from tqdm import tqdm

# from ripser import ripser
from scipy.sparse import coo_matrix
from scipy.ndimage import maximum_filter, minimum_filter1d
from typing import List, Iterable, Literal, Dict
from pyimzml.ImzMLWriter import ImzMLWriter
from pyimzml.compression import NoCompression, ZlibCompression


from bokeh.plotting import show
from .utils import CoordsGraph, local_maxima
from .plotting import spectrum, mobilogram, heatmap, image, _visualize

__all__ = ["MSIDataset", "Frame", "export_imzML"]


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
            # self.pos = pd.read_sql("SELECT * FROM MaldiFrameInfo", con)[
            #     ["Frame", "XIndexPos", "YIndexPos"]
            # ]
            self.pos = pd.read_sql("SELECT * FROM MaldiFrameInfo", con, index_col="Frame")[
                ["XIndexPos", "YIndexPos"]
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
        self.rois = {}

    def __repr__(self):
        return f"{self.__class__.__name__} with {self.data.frame_max_index-1} pixels\n\
        mz range: {self.data.mz_min_value:.3f}-{self.data.mz_max_value:.3f}\n\
        mobility range: {self.data.mobility_min_value:.3f}-{self.data.mobility_max_value:.3f}\n\
        "

    # get one frame
    def __getitem__(self, index):
        return Frame(
            self.data[index],
            mz_domain=self.data.mz_values,
            mobility_domain=self.data.mobility_values,
        )

    def ccs_calibrator(self, method: Literal["linear", "internal"] = "linear"):
        """Generate a CCS calibrater, mapping 1/K_0 -> CCS, using either a linear model or Bruker's internal method.
        The linear model is based on refernce CCS from Calibrants(not Mason-Shamp equation) recorded in the raw data, and Bruker's method is from the dll that takes 1/K, charge and m/z as inputs

        :return: a calibrator with a `transform` method.
        :rtype: _type_
        """
        from .calibration import CCS_calibration, CCS_Bruker_Calibration

        if method=="linear":
            polarity = self.cali_info["KeyPolarity"].iloc[0]
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
        elif method=="internal":
            calibrator = CCS_Bruker_Calibration()
        else:
            raise NotImplementedError("Method not implemented!")
        return calibrator

    def tic(self) -> pd.Series:
        """TIC

        :return: _description_
        :rtype: pd.Series
        """
        # is already summarized in .tdf
        intensities = self.data.frames["SummedIntensities"][1:]
        intensities.name = "intensity_values"
        return intensities

    def rms(self):
        rmss = []
        for i in range(1, self.data.frame_max_index):
            intensities = self.data.intensity_values[self.data[i, "raw"]]
            scale = np.sqrt(np.mean(np.square(intensities)))
            rmss.append(scale)
        return rmss

    def slice_image(self, mz: slice = None, mobility: slice = None) -> np.ndarray:
        """Get slice image data, used for ion images.
        Example: slice_image(mz=slice(499, 500), mobility=slice(1.0, 1.1))

        :param mz: mz range in slice, defaults to None
        :type mz: slice, optional
        :param mobility: mobility range in slice, defaults to None
        :type mobility: slice, optional
        :return: an intensity array
        :rtype: np.ndarray
        """
        if mz is not None or mobility is not None:
            indices = self.data[:, mobility, 0, mz, "raw"]
            intensities = self.data.bin_intensities(indices, axis=["rt_values"])[1:]
        else:
            intensities = self.tic().to_numpy()
        return intensities

    def set_ROI(self, name, xmin=None, xmax=None, ymin=None, ymax=None):
        filt1 = self.pos["XIndexPos"] > xmin if xmin is not None else True
        filt2 = self.pos["XIndexPos"] < xmax if xmax is not None else True
        filt3 = self.pos["YIndexPos"] > ymin if ymin is not None else True
        filt4 = self.pos["YIndexPos"] > ymax if ymax is not None else True
        self.rois[name] = self.pos.loc[filt1 & filt2 & filt3 & filt4].index.to_numpy()

    def mean_spectrum(
        self,
        frame_indices: np.ndarray = None,
        sampling_ratio: float = 1.0,
        frequency_threshold: float = 0.05,
        as_frame=False,
        seed=42,
    ):
        """compute mean spectra over the whole dataset

        :param as_frame: if True, return a pd.DataFrame, otherwise an Frame, defaults to False
        :type as_frame: bool, optional
        :param frequency_threshold: Filter out intensities that appear in little fraction of pixels, defaults to 0.05
        :type frequency_threshold: float, optional
        :return: _description_
        :rtype: _type_
        """

        if seed is not None:
            np.random.seed(seed)
        if frame_indices is None:
            frame_indices = np.arange(1, self.data.frame_max_index)
            n_frame = self.data.frame_max_index - 1
        # if sampling_ratio == 1:
        #     intensity_indices = np.arange(len(self.data))
        #     n_frame = self.data.frame_max_index - 1

        # randomly pick frames out of n
        if sampling_ratio < 1:
            n_frame = int(frame_indices.shape[0] * sampling_ratio)
            frame_indices = np.random.choice(
                frame_indices,
                size=n_frame,
            )
        elif sampling_ratio == 1:
            n_frame = frame_indices.shape[0]
        intensity_indices = self.data[frame_indices, "raw"]

        sum_mx = self.data.bin_intensities(intensity_indices, axis=["mz_values", "mobility_values"])
        if frequency_threshold is not None:
            intensity_cut = self.data.intensity_min_value * n_frame * frequency_threshold
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
        # if frequency_threshold is not None:
        #     mean_spec = mean_spec.loc[
        #         lambda x: x >= self.data.intensity_min_value * frequency_threshold
        #     ]  # need improvement

        if as_frame:
            return mean_spec
        else:
            return Frame(
                mean_spec,
                mz_domain=self.data.mz_values,
                mobility_domain=self.data.mobility_values,
            )

    def integrate_intensity(self, peak_list: pd.DataFrame, peak_extents: pd.DataFrame):
        n_peak = peak_list.shape[0]
        frame_indices = np.arange(1, self.data.frame_max_index)
        # if isinstance(intensity_threshold, float):
        # np.max(peak_list["total_intensity"]) * intensity_threshold
        # use dataframe for missing values
        intensity_array = pd.DataFrame(
            None,
            index=frame_indices,
            columns=np.arange(1, n_peak + 1),
        )  # (n_pixel, n_peak)
        intensity_array.index.name = "Pixel index"
        intensity_array.columns.name = "Feature index"
        for i in tqdm(range(n_peak)):
            tof_min, tof_max, scan_min, scan_max = peak_extents.iloc[i][
                ["tof_indices", "scan_indices"]
            ].astype(int)
            indices = self.data[
                :, scan_min : (scan_max + 1), 0, tof_min : (tof_max + 1), "raw"
            ]  # all data points of a peak
            intensity_array[i + 1] = self.data.bin_intensities(indices, axis=["rt_values"])[
                frame_indices
            ]  # JIT function
        return intensity_array

    def process(
        self,
        sampling_ratio=0.1,
        frequency_threshold=0.05,
        intensity_threshold=None,
        roi=None,  # what if there are multiple ROIs?
        visualize=False,
        ccs_calibration=False,
        **kwargs,
    ) -> Dict:
        """Process the dataset to peak picked and aligned data cube

        :param sampling_ratio: ratio for computing the mean spectrum, defaults to 0.1
        :type sampling_ratio: float, optional
        :param frequency_threshold: intensity threshold relative to lowest intensity in a single frame for filtering the mean spectrum, defaults to 0.05
        :type frequency_threshold: float, optional
        :return: a dictionary with intensity array, peak list and coordinates
        :rtype: Dict
        """
        # peak picking
        if roi is not None:
            assert roi in self.rois
            frame_indices = self.rois[roi]
        else:
            frame_indices = np.arange(1, self.data.frame_max_index)

        print("Computing mean spectrum...")
        mean_spec = self.mean_spectrum(
            sampling_ratio=sampling_ratio,
            frequency_threshold=frequency_threshold,
            frame_indices=frame_indices,
        )

        peak_list, peak_extents = mean_spec.peakPick(return_extents=True, **kwargs).values()
        if isinstance(intensity_threshold, float):
            intensity_cut = np.max(peak_list["total_intensity"]) * intensity_threshold
            indices = peak_list["total_intensity"] > intensity_cut
            peak_list = peak_list.loc[indices]
            peak_extents = peak_extents.loc[indices]
        intensity_array = self.integrate_intensity(peak_list, peak_extents)

        # intensity_array.fillna(0.0, inplace=True)

        results = {
            "coords": self.pos.loc[frame_indices],
            "peak_list": peak_list,
            "intensity_array": intensity_array,
        }  # 3 dataframes
        if visualize is True:
            app = _visualize(self, mean_spec, peak_list, peak_extents)
            results["viz"] = app

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

    def image(self, mz: slice = None, mobility: slice = None):
        """Show slice image of a dataset, TIC image by default

        :param mz: mz range, defaults to None
        :type mz: slice, optional
        :param mobility: mobility range, defaults to None
        :type mobility: slice, optional
        """
        f, _ = image(self, mz=mz, mobility=mobility)
        show(f)


class Frame:
    """The class for a frame."""

    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        mz_domain: np.ndarray | None = None,
        mobility_domain: np.ndarray | None = None,
        coords: List[int] = None,
        # dataset: MSIDataset = None,
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
        self.mz_domain = mz_domain
        self.mobility_domain = mobility_domain
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

    def to_dense_array(self):
        df = self.data
        X = self.mz_domain
        Y = self.mobility_domain

        intensity_mx = coo_matrix(
            (df.intensity_values, (df.scan_indices, df.tof_indices)), shape=(len(Y), len(X))
        )
        return intensity_mx.toarray()

    def peakPick(
        self,
        tolerance: Iterable[int | float] | int | float | None = 2,
        metric: Literal["euclidean", "chebyshev"] = "euclidean",
        count_thrshold=5,  # at least 5 points for a 3D peak
        window_size: Iterable[int] = [17, 7],
        adaptive_window=False,
        subdivide=True,
        sort=False,
        return_labels=False,
        return_extents=False,
        return_apex=False,
    ) -> Dict:
        """
        2D peak-picking on a frame
        1)group intensities based on approximity in (mz, mobility) space,
        2)detect local maxima in each group as peaks,
        3)summarize m/z, ion mobility and intensity for each peak

        :param tolerance: [mobility, mz] tolerance in integer indices for step 1, two data points within the distance determined by the tolerance are neighbors, defaults to [2,2]
        :type tolerance: Iterable[int  |  float] | int | float | None, optional
        :param metric: distance metric, defaults to "euclidean"
        :type metric: Literal[&quot;euclidean&quot;, &quot;chebyshev&quot;], optional
        :param count_thrshold: minimum count of data points for a peak, defaults to 5
        :type count_thrshold: int, optional
        :param window_size: [mobility, mz] window size of the maximum filter for step 2, defaults to [17, 7]
        :type window_size: Iterable[int], optional
        :param adaptive_window: if True, the function would determine the maximum filter size automatically and override `window_size`, defaults to False
        :type adaptive_window: bool, optional
        :param subdivide: whether to detect saddle points in a group to divide it into multiple peaks, defaults to True
        :type subdivide: bool, optional
        :param sort: if True, sort peaks by descending total intensity, defaults to False
        :type sort: bool, optional
        :param return_labels: return group labels, defaults to False
        :type return_labels: bool, optional
        :param return_extents: return the m/z and ion mobility ranges for each peak, defaults to False
        :type return_extents: bool, optional
        :param return_apex: return apexes m/z and ion mobility values for each peak, where peak list uses average weighted values to represent peaks, defaults to False
        :type return_apex: bool, optional
        :return: a dicitonary contains `peak_list`(a dataframe) and other optional results
        :rtype: Dict
        """

        if self.idx_available is True:
            coords = self.data[["tof_indices", "scan_indices"]].to_numpy(
                dtype=np.float32, copy=True
            )  # coerce to float type to avoid overflow

        graph = CoordsGraph(coordinates=coords, tolerance=tolerance, metric=metric)

        print("Traversing graph...")
        group_labels = graph.group_nodes(count_thrshold=count_thrshold)  # ndarray of (k,)
        # filter off intensities with group label=0
        intensity_groups = self.data[group_labels > 0].groupby(
            group_labels[group_labels > 0], group_keys=True
        )  # filter, then group

        print("Finding local maxima...")
        raw_apexes = []
        peak_labels = np.zeros_like(group_labels)
        current_group = 1
        for i, g in intensity_groups:
            # create dense matrix(in DataFrame) for each group
            dense_mx = g.reset_index().pivot(
                index="scan_indices", columns="tof_indices", values="intensity_values"
            )
            dense_mx.fillna(0, inplace=True)
            # find local maxima
            if adaptive_window:
                h, w = dense_mx.shape  # row index is y and col is x
                window_size = [max(h // 2 + 1, window_size[0]), max(w // 2 + 1, window_size[1])]
            peaks = local_maxima(dense_mx, window_size=window_size)
            raw_apexes.append(peaks)

            # divide a group with multiple local maxima by saddle points
            if (peaks.shape[0] > 1) and subdivide:
                peaks = peaks.reset_index()  # (scan, tof, intensity)
                # is mz resolvable?
                # tof_indices are stored as unsigned integer
                if (np.max(peaks["tof_indices"]) - np.min(peaks["tof_indices"])) <= tolerance:
                    proj = np.sum(dense_mx, axis=1)

                    minima = minimum_filter1d(proj, size=window_size[0] // 2)
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
                    minima = minimum_filter1d(proj, size=window_size[1] // 2)
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
        print("Summarizing...")
        # intensity-weighted mz and mob
        peak_list = peak_groups.apply(
            lambda df: df[["mz_values", "mobility_values"]].apply(
                np.average, weights=df["intensity_values"]
            )
        )
        peak_list["total_intensity"] = peak_groups["intensity_values"].sum()
        # peak_list = peak_list.reset_index()
        if sort:
            peak_list.sort_values("total_intensity", ascending=False, inplace=True)

        results = {"peak_list": peak_list}
        # include group labels
        if return_labels is True:
            results["peak_labels"] = peak_labels
        # include extents of each peak
        if return_extents is True:
            peak_extents = peak_groups[
                ["tof_indices", "scan_indices", "mz_values", "mobility_values"]
            ].agg(["min", "max"])
            # peak_extents = peak_extents.reset_index()
            results["peak_extents"] = peak_extents

        if return_apex is True:
            apex_list = pd.concat(raw_apexes).reset_index()
            apex_list["mz_values"] = self.mz_domain[apex_list["tof_indices"]]
            apex_list["mobility_values"] = self.mobility_domain[apex_list["scan_indices"]]
            results["peak_apex"] = apex_list
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
    mode: Literal["centroid", "profile"] = "centroid",
    imzml_mode: Literal["continuous", "processed"] = "continuous",
):
    """Export processed data as imzML format with ion mobility

    :param dataset: the original dataset, contains necessary metadata
    :type dataset: MSIDataset
    :param path: path of the output
    :type path: str
    :param peaks: processing results from MSIDataset.process(), defaults to None
    :type peaks: Dict, optional
    :param mode: , defaults to "centroid"
    :type mode: Literal[&quot;centroid&quot;, &quot;profile&quot;], optional
    :param imzml_mode: mode of arrays in the imzML file, defaults to "continuous"
    :type imzml_mode: Literal[&quot;continuous&quot;, &quot;processed&quot;], optional
    """
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
        mean_spec = dataset.mean_spectra(frequency_threshold=0.05)
        peak_list, peak_extents = mean_spec.peakPick(
            return_extents=True,
        ).values()
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
    # pos = dataset.pos.set_index("Frame")

    # indices_sorted = peak_list.sort_values("mz_values").index
    indices_sorted = np.argsort(peak_list["mz_values"])

    mz_array = peak_list["mz_values"].to_numpy()[indices_sorted]
    mobility_array = peak_list["mobility_values"].to_numpy()[indices_sorted]
    intensity_array = intensity_array.iloc[:, indices_sorted]
    # write files
    for frame in tqdm(intensity_array.index):
        # or I can do adhoc extraction here?

        writer.addSpectrum(
            mzs=mz_array,
            intensities=intensity_array.loc[frame].to_numpy(),
            mobilities=mobility_array,
            coords=dataset.pos.loc[frame],
        )
    writer.close()
