import re
from tqdm import tqdm
import numpy as np
from typing import Tuple, Dict, Literal
from pyimzml.compression import NoCompression
from pyimzml.ImzMLWriter import ImzMLWriter
from .spectrum import MSIDataset

# Type alias for a CV term triple
CVTerm = Tuple[str, str, str]  # (cvRef, accession, name)

# ---------------------------------------------------------------------------
# CV term constants
# Verified against psi-ms.obo data-version 4.1.244 (2026-03-13)
# https://raw.githubusercontent.com/hupo-psi/psi-ms-cv/master/psi-ms.obo
# ---------------------------------------------------------------------------

# --- Instrument model (sits directly on <instrumentConfiguration>) ---
#
# For a more specific Bruker timsTOF model (e.g. timsTOF fleX), pass the
# appropriate term via the `instrument_model` constructor kwarg.  The model
# terms live under MS:1000122 (Bruker Daltonics instrument model) in the OBO.
CV_INSTRUMENT_MODEL_DEFAULT: CVTerm = (
    "MS",
    "MS:1003970",
    "ion mobility quadrupole time-of-flight instrument",
)

# --- Source ---
CV_MALDI_SOURCE: CVTerm = ("MS", "MS:1000075", "matrix-assisted laser desorption ionization")

# --- Analyzer ---
# There is currently no component-level OBO term for TIMS as a standalone
# analyzer stage; the instrument model term above encodes its presence.
CV_TOF_ANALYZER: CVTerm = ("MS", "MS:1000084", "time-of-flight")

# --- Detector ---
# MS:1000114  microchannel plate detector  [confirmed]
# is_a: MS:1000345 (array detector) — correct component-level term.
# Used by SCiLS for the same instrument class; appropriate for timsTOF.
CV_MCP_DETECTOR: CVTerm = ("MS", "MS:1000114", "microchannel plate detector")

# --- Scan settings: pixel size ---
# IMS:1000046 / IMS:1000047 live in the IMS CV, not psi-ms.obo.
# cvRef must be "IMS".
CV_PIXEL_SIZE_X: CVTerm = ("IMS", "IMS:1000046", "pixel size x")
CV_PIXEL_SIZE_Y: CVTerm = ("IMS", "IMS:1000047", "pixel size y")

# --- Software / data processing ---
CV_ANALYSIS_SOFTWARE: CVTerm = ("MS", "MS:1000799", "custom unreleased software tool")
CV_PEAK_PICKING: CVTerm = ("MS", "MS:1000035", "peak picking")
CV_FORMAT_CONVERSION: CVTerm = ("MS", "MS:1000530", "file format conversion")



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cv_param(cv_ref: str, accession: str, name: str, value: str = "") -> str:
    """Return a <cvParam .../> element (no leading indent)."""
    if value:
        return (
            f'<cvParam cvRef="{cv_ref}" accession="{accession}" ' f'name="{name}" value="{value}"/>'
        )
    return f'<cvParam cvRef="{cv_ref}" accession="{accession}" name="{name}"/>'


# ---------------------------------------------------------------------------
# Subclass
# ---------------------------------------------------------------------------


class TimsImagingImzMLWriter(ImzMLWriter):
    """ImzMLWriter subclass that fills in MALDI-TIMS-TOF instrument metadata.

    pyimzml leaves several imzML sections empty or hardcoded.  This subclass
    patches them after the parent renders the XML:

    - ``instrumentConfigurationList`` — instrument model CV term, plus a
      three-component list (MALDI source, TOF analyzer, MCP detector).
    - ``scanSettingsList`` — pixel size x/y in µm injected into the existing
      block produced by the parent.
    - ``softwareList`` — timsimaging credited as the authoring software.
    - ``dataProcessingList`` — single processing block recording peak picking
      followed by imzML export, both attributed to timsimaging.

    Parameters
    ----------
    path : str
        Output path for the ``.imzML`` file (passed through to
        ``ImzMLWriter``).
    pixel_size_um : float
        Pixel (raster) size in micrometres.  Pass
        ``MSIDataset.resolution["xy"]`` directly.
    instrument_model : CVTerm, optional
        A ``(cvRef, accession, name)`` triple for the instrument model term,
        placed as a ``cvParam`` directly on ``<instrumentConfiguration>``.
        Defaults to ``CV_INSTRUMENT_MODEL_DEFAULT`` (MS:1003970,
        *ion mobility quadrupole time-of-flight instrument*).  Override with
        a more specific Bruker timsTOF model term if known, e.g.::

            instrument_model=('MS', 'MS:XXXXXXX', 'Bruker timsTOF fleX')

        Specific Bruker model terms live under MS:1000122 in the OBO.
    **kwargs
        All remaining keyword arguments are forwarded to ``ImzMLWriter``
        (e.g. ``polarity``, ``mode``, ``spec_type``, dtype and compression
        arguments).  ``include_mobility`` defaults to ``True`` and does not
        need to be passed explicitly.
    """

    _SOFTWARE_ID = "timsimaging"
    _SOFTWARE_VERSION = "0.1.0"

    def __init__(
        self,
        path,
        *,
        pixel_size_um: float,
        instrument_model: CVTerm = CV_INSTRUMENT_MODEL_DEFAULT,
        **kwargs,
    ):
        self._pixel_size_um = pixel_size_um
        self._instrument_model = instrument_model
        # TIMS data always carries a mobility dimension
        kwargs.setdefault("include_mobility", True)
        super().__init__(path, **kwargs)

    # ------------------------------------------------------------------
    # XML snippet builders
    # ------------------------------------------------------------------

    def _instrument_config_xml(self) -> str:
        """Return the full ``<instrumentConfigurationList>`` XML block.

        Places the instrument model CV term directly on
        ``<instrumentConfiguration>``, then lists three components in order:
        MALDI source (MS:1000075), TOF analyzer (MS:1000084), and microchannel
        plate detector (MS:1000114).  There is currently no component-level OBO
        term for TIMS as a standalone analyzer stage; its presence is encoded
        by the instrument model term.
        """
        return "\n".join(
            [
                '  <instrumentConfigurationList count="1">',
                '    <instrumentConfiguration id="IC1">',
                f"      {_cv_param(*self._instrument_model)}",
                '      <componentList count="3">',
                '        <source order="1">',
                f"          {_cv_param(*CV_MALDI_SOURCE)}",
                "        </source>",
                '        <analyzer order="2">',
                f"          {_cv_param(*CV_TOF_ANALYZER)}",
                "        </analyzer>",
                '        <detector order="3">',
                f"          {_cv_param(*CV_MCP_DETECTOR)}",
                "        </detector>",
                "      </componentList>",
                "    </instrumentConfiguration>",
                "  </instrumentConfigurationList>",
            ]
        )

    def _software_list_xml(self) -> str:
        """Return the ``<softwareList>`` XML block.

        Credits timsimaging (``_SOFTWARE_ID``, ``_SOFTWARE_VERSION``) as a
        single software entry using CV term MS:1000799 (custom unreleased
        software tool).
        """
        base = _cv_param(*CV_ANALYSIS_SOFTWARE)
        tims_cv = base[:-2] + f' value="{self._SOFTWARE_ID}"/>'
        return "\n".join(
            [
                '  <softwareList count="1">',
                f'    <software id="software0" version="{self._SOFTWARE_VERSION}">',
                f"      {tims_cv}",
                "    </software>",
                "  </softwareList>",
            ]
        )

    def _data_processing_list_xml(self) -> str:
        """Return the ``<dataProcessingList>`` XML block.

        Records a single ``<dataProcessing>`` block with one
        ``<processingMethod>`` attributed to timsimaging, containing two
        CV terms in processing order: peak picking (MS:1000035, order 0)
        followed by file format conversion to imzML (MS:1000530, order 1).
        """
        peakpicking_cv = _cv_param(*CV_PEAK_PICKING)[:-2] + f' value="2D peak picking"/>'
        export_cv = _cv_param(*CV_FORMAT_CONVERSION)[:-2] + f' value="Export to imzML with ion mobility"/>'

        return "\n".join(
            [
                '  <dataProcessingList count="1">',
                f'    <dataProcessing id="dataProcessing0">',
                f'      <processingMethod order="0" softwareRef="software0">',
                f"        {peakpicking_cv}",
                f"        {export_cv}",
                "      </processingMethod>",
                "    </dataProcessing>",
                "  </dataProcessingList>",
            ]
        )

    def _patch_scan_settings(self, xml: str) -> str:
        """Inject pixel size cvParams just before </scanSettings>."""
        px = str(self._pixel_size_um)
        insert = (
            f"\n      {_cv_param(*CV_PIXEL_SIZE_X, value=px)}"
            f"\n      {_cv_param(*CV_PIXEL_SIZE_Y, value=px)}"
        )
        return xml.replace("    </scanSettings>", insert + "\n    </scanSettings>", 1)

    # ------------------------------------------------------------------
    # Override _write_xml
    # ------------------------------------------------------------------

    def _write_xml(self):
        """Render the imzML XML and inject TIMS-specific metadata.

        Calls the parent ``_write_xml()`` to produce the standard imzML
        output, then reads the file back and applies four regex patches:
        instrument configuration, software list, data processing list, and
        pixel size inside the scan settings block.
        """
        # 1. Let pyimzml render the full XML to disk as usual
        super()._write_xml()

        # 2. Read it back, patch the four sections, write it back
        self.xml.flush()
        with open(self.filename, "r", encoding="ISO-8859-1") as f:
            xml = f.read()

        xml = re.sub(
            r'[ \t]*<instrumentConfigurationList count="1">.*?</instrumentConfigurationList>',
            self._instrument_config_xml(),
            xml,
            count=1,
            flags=re.DOTALL,
        )
        xml = re.sub(
            r'[ \t]*<softwareList count="1">.*?</softwareList>',
            self._software_list_xml(),
            xml,
            count=1,
            flags=re.DOTALL,
        )
        xml = re.sub(
            r'[ \t]*<dataProcessingList count="1">.*?</dataProcessingList>',
            self._data_processing_list_xml(),
            xml,
            count=1,
            flags=re.DOTALL,
        )
        xml = self._patch_scan_settings(xml)

        with open(self.filename, "w", encoding="ISO-8859-1") as f:
            f.write(xml)

def export_imzML(
    dataset: MSIDataset,
    path: str,
    peaks: Dict = None,
    mode: Literal["centroid", "profile"] = "centroid",
    imzml_mode: Literal["continuous", "processed"] = "continuous",
):
    """Export peak-picked MSI data as an imzML file with ion mobility arrays.

    Writes ``.imzML`` and ``.ibd`` files using :class:`TimsImagingImzMLWriter`,
    which adds instrument configuration, pixel size, software credit, and data
    processing metadata that pyimzml otherwise leaves empty.

    Peaks are sorted by ascending m/z before writing so that the imzML arrays
    are in the order expected by continuous-mode readers.

    :param dataset: source dataset, used for pixel coordinates, polarity, and
        spatial resolution metadata.
    :type dataset: MSIDataset
    :param path: output path for the ``.imzML`` file; the ``.ibd`` binary file
        is written to the same directory with the same stem.
    :type path: str
    :param peaks: pre-computed processing results dict as returned by
        :meth:`MSIDataset.process`.  Must contain ``"peak_list"`` (a DataFrame
        with ``mz_values`` and ``mobility_values`` columns) and
        ``"intensity_array"`` (a per-pixel intensity DataFrame).  If ``None``,
        a mean spectrum is computed on the fly and peak picking is run with
        default parameters before export.
    :type peaks: Dict, optional
    :param mode: spectral representation written per pixel — ``"centroid"`` for
        peak-picked data (the normal case) or ``"profile"`` for raw spectra.
        Defaults to ``"centroid"``.
    :type mode: Literal["centroid", "profile"], optional
    :param imzml_mode: array storage layout — ``"continuous"`` writes shared m/z
        and mobility arrays once (suitable when all pixels share the same peak
        list), ``"processed"`` writes per-pixel arrays.  Defaults to
        ``"continuous"``.
    :type imzml_mode: Literal["continuous", "processed"], optional
    """
    key_polarity = dataset.cali_info["KeyPolarity"].iloc[0]
    if key_polarity == "+":
        polarity = "positive"
    elif key_polarity == "-":
        polarity = "negative"
    compression_object = NoCompression()
    # create imzML and ibd files
    writer = TimsImagingImzMLWriter(
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
        pixel_size_um=dataset.resolution["xy"]
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