"""Peak-pick selected Bruker MSI ROIs and export CSV plus imzML/IBD.

CCS calibration is enabled by default. The generated peak-list CSV therefore
contains ``mz_values``, ``mobility_values``, ``total_intensity``, and
``ccs_values`` without a separate post-processing step.
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def add_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        "process",
        description=__doc__,
        help="Peak-pick selected Bruker MSI ROIs and export CSV plus imzML/IBD.",
    )
    parser.add_argument("dataset", type=Path, help="Bruker .d dataset directory")
    parser.add_argument(
        "--regions",
        nargs="+",
        metavar="ROI",
        help="ROI keys to process, for example: --regions r0 r2 "
        "(default: process the whole dataset as a single region)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for roi_summary.csv, peak_list[_<roi>].csv, and timsimaging.log",
    )
    parser.add_argument(
        "--imzml-output-dir",
        type=Path,
        help="Directory for imzML/IBD; defaults to --output-dir",
    )
    parser.add_argument("--sampling-ratio", type=float, default=0.1)
    parser.add_argument("--frequency-threshold", type=float, default=0.05)
    parser.add_argument("--tolerance", type=float, default=3)
    parser.add_argument(
        "--adaptive-window",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use adaptive peak-picking windows (default: enabled)",
    )
    parser.add_argument(
        "--ccs-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add ccs_values using the TDF calibration (default: enabled)",
    )
    parser.add_argument(
        "--export-imzml",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export one imzML/IBD pair per ROI (default: enabled)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow existing output files to be replaced",
    )
    parser.set_defaults(func=run)
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if not args.dataset.is_dir():
        parser.error(f"dataset is not a directory: {args.dataset}")
    if args.dataset.suffix.lower() != ".d":
        parser.error(f"dataset must be a Bruker .d directory: {args.dataset}")
    if not (args.dataset / "analysis.tdf").is_file():
        parser.error(f"analysis.tdf not found under: {args.dataset}")
    if not math.isfinite(args.sampling_ratio) or not 0 < args.sampling_ratio <= 1:
        parser.error("--sampling-ratio must be greater than 0 and at most 1")
    if not math.isfinite(args.frequency_threshold) or args.frequency_threshold < 0:
        parser.error("--frequency-threshold must be nonnegative")
    if not math.isfinite(args.tolerance) or args.tolerance <= 0:
        parser.error("--tolerance must be greater than 0")
    if args.regions is not None and len(args.regions) != len(set(args.regions)):
        parser.error("--regions contains duplicate ROI keys")


def output_paths(
    args: argparse.Namespace,
) -> tuple[Path, Path, dict[str | None, list[Path]]]:
    output_dir = args.output_dir.resolve()
    imzml_dir = (args.imzml_output_dir or output_dir).resolve()
    stem = args.dataset.name[: -len(".d")]
    # a single `None` region means "the whole dataset", written without a
    # `_<roi>` suffix so single-region output looks like a plain export
    regions = args.regions if args.regions is not None else [None]
    paths: dict[str | None, list[Path]] = {}
    for roi in regions:
        suffix = f"_{roi}" if roi is not None else ""
        roi_paths = [output_dir / f"peak_list{suffix}.csv"]
        if args.export_imzml:
            roi_paths.extend(
                [
                    imzml_dir / f"{stem}{suffix}.imzML",
                    imzml_dir / f"{stem}{suffix}.ibd",
                ]
            )
        paths[roi] = roi_paths
    return output_dir, imzml_dir, paths


def validate_ccs_calibration(dataset) -> None:
    required_rows = {"ReferenceMobilityPeakNames", "MobilitiesPreviousCalibration"}
    missing_rows = sorted(required_rows.difference(dataset.cali_info.index))
    if "KeyPolarity" not in dataset.cali_info.columns:
        raise RuntimeError("CalibrationInfo is missing the KeyPolarity column")
    if missing_rows:
        raise RuntimeError(
            "CalibrationInfo is missing required key(s): " + ", ".join(missing_rows)
        )
    try:
        dataset.ccs_calibrator()
    except Exception as error:
        raise RuntimeError(f"unable to construct CCS calibrator: {error}") from error


def validate_processing_results(results: dict, require_ccs: bool) -> None:
    missing_results = [
        key for key in ("coords", "peak_list", "intensity_array") if key not in results
    ]
    if missing_results:
        raise RuntimeError("processing result is missing: " + ", ".join(missing_results))

    peak_list = results["peak_list"]
    required_columns = ["mz_values", "mobility_values", "total_intensity"]
    if require_ccs:
        required_columns.append("ccs_values")
    missing_columns = [column for column in required_columns if column not in peak_list.columns]
    if missing_columns:
        raise RuntimeError(
            "processed peak list is missing required column(s): "
            + ", ".join(missing_columns)
        )
    if peak_list.empty:
        raise RuntimeError("processing produced an empty peak list")
    if not np.isfinite(peak_list[required_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("processed peak list contains non-finite values")

    intensity_array = results["intensity_array"]
    if intensity_array.shape[1] != len(peak_list):
        raise RuntimeError("intensity matrix columns do not match the peak list")
    if intensity_array.shape[0] != len(results["coords"]):
        raise RuntimeError("intensity matrix rows do not match the ROI coordinates")


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    validate_args(parser, args)
    output_dir, imzml_dir, paths = output_paths(args)

    planned = [output_dir / "roi_summary.csv"]
    planned.extend(path for roi_paths in paths.values() for path in roi_paths)
    existing = [path for path in planned if path.exists()]
    if existing and not args.overwrite:
        parser.error(
            "output files already exist; use --overwrite to replace them:\n  "
            + "\n  ".join(str(path) for path in existing)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if args.export_imzml:
        imzml_dir.mkdir(parents=True, exist_ok=True)

    # capture this run's log messages (from this module and the rest of
    # timsimaging, e.g. spectrum.py) alongside the CSV/imzML outputs
    log_path = output_dir / "timsimaging.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    package_logger = logging.getLogger("timsimaging")
    package_logger.addHandler(file_handler)
    try:
        import alphatims.utils
        from timsimaging.io import export_imzML
        from timsimaging.spectrum import MSIDataset

        alphatims.utils.set_progress_callback(True)
        dataset_path = args.dataset.resolve()
        logger.info("== loading %s", dataset_path)
        started = time.time()
        dataset = MSIDataset(str(dataset_path))
        logger.info("== loaded in %.0fs", time.time() - started)
        if args.ccs_calibration:
            validate_ccs_calibration(dataset)
            logger.info("== CCS calibration metadata validated (charge +1)")

        summary = dataset.list_rois()
        logger.info("%s", summary.to_string(index=False))
        available = set(summary["key"].astype(str))
        if args.regions is not None:
            unknown = [roi for roi in args.regions if roi not in available]
            if unknown:
                parser.error(
                    f"unknown ROI key(s): {', '.join(unknown)}; available: "
                    + ", ".join(sorted(available))
                )
        summary_path = output_dir / "roi_summary.csv"
        summary.to_csv(summary_path, index=False)

        process_kwargs = {
            "sampling_ratio": args.sampling_ratio,
            "frequency_threshold": args.frequency_threshold,
            "tolerance": args.tolerance,
            "adaptive_window": args.adaptive_window,
            "ccs_calibration": args.ccs_calibration,
        }

        regions = args.regions if args.regions is not None else [None]
        for roi in regions:
            roi_label = roi if roi is not None else "whole dataset"
            logger.info("\n== processing %s", roi_label)
            started = time.time()
            results = dataset.process(roi=roi, **process_kwargs)
            validate_processing_results(results, require_ccs=args.ccs_calibration)
            peak_list = results["peak_list"]

            peak_csv = paths[roi][0]
            peak_list.to_csv(peak_csv)
            if not peak_csv.is_file() or peak_csv.stat().st_size == 0:
                raise RuntimeError(f"peak-list output was not written correctly: {peak_csv}")
            written = [peak_csv]
            if args.export_imzml:
                imzml_path = paths[roi][1]
                export_imzML(dataset, str(imzml_path), peaks=results)
                imzml_outputs = [imzml_path, paths[roi][2]]
                invalid = [
                    path
                    for path in imzml_outputs
                    if not path.is_file() or path.stat().st_size == 0
                ]
                if invalid:
                    raise RuntimeError(
                        "imzML export did not create nonempty output(s): "
                        + ", ".join(map(str, invalid))
                    )
                written.extend(imzml_outputs)

            n_pixels, n_peaks = results["intensity_array"].shape
            logger.info(
                "== %s done: %d peaks, %d pixels, %.0fs",
                roi_label,
                n_peaks,
                n_pixels,
                time.time() - started,
            )
            for path in written:
                logger.info("   %s", path)
            del results
            gc.collect()

        logger.info("\n== all regions complete; ROI summary: %s", summary_path)
        return 0
    finally:
        package_logger.removeHandler(file_handler)
        file_handler.close()
