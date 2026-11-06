"""Isotope-pattern extraction for picked TIMS imaging spectra."""

from __future__ import annotations

from copy import copy
from numbers import Integral, Real
import re
from typing import Iterable

import numpy as np
import pandas as pd
from pyteomics.mass import nist_mass

from .isotope_ownership import resolve_paths

__all__ = ["Deisotoper", "normalize_isotope_elements"]


PATTERN_COLUMNS = [
    "pattern_id", "precursor_peak_id", "precursor_mz",
    "precursor_mobility", "apex_scan", "apex_mobility", "charge",
    "isotope_count", "flags", "sirius_ready",
]
ISOTOPE_COLUMNS = [
    "pattern_id", "isotope_index", "peak_id", "mz_values",
    "mobility_values", "mobility_delta", "total_intensity",
    "relative_intensity", "apex_scan", "apex_mobility",
    "apex_scan_delta",
]
MAPPING_COLUMNS = ["peak_id", "pattern_id", "role", "status"]
HYPOTHESIS_COLUMNS = ["pattern_id", "peak_id", "isotope_index"]
DECISION_COLUMNS = [
    "parent_pattern_id", "child_pattern_id", "shared_peak_id",
    "decision", "reason", "junction_intensity_ratio",
    "max_step_mobility_delta", "total_mobility_delta",
    "parent_before", "child_before", "parent_after", "child_after",
]
LINK_COLUMNS = [
    "pattern_id", "source_peak_id", "target_peak_id",
    "source_isotope_index", "target_isotope_index", "link_type",
    "observed_mass_shift", "expected_mass_shift_lower",
    "expected_mass_shift_upper", "mass_tolerance", "mass_margin",
    "mass_agrees", "mobility_delta", "mobility_tolerance",
    "mobility_margin", "mobility_agreement", "mobility_agrees",
    "accepted",
]
RICH_COLUMNS = [
    "pattern_id", "precursor_peak_id", "peak_id", "isotope_index",
    "role", "status", "mz_values", "mobility_values", "mobility_delta",
    "total_intensity", "relative_intensity", "apex_scan",
    "apex_mobility", "apex_scan_delta", "observed_mass_shift",
    "expected_mass_shift_lower", "expected_mass_shift_upper",
    "mass_tolerance", "mass_margin", "mass_agrees",
    "best_predecessor_peak_id", "supporting_link_count",
    "step_mobility_delta", "mobility_tolerance", "mobility_margin",
    "mobility_agreement", "mobility_agrees", "candidate_count_at_order",
    "active_owner_count", "competing_partner", "shared_peak",
    "ownership_decision_count", "ownership_resolved",
    "ownership_unresolved", "pattern_flags", "sirius_ready",
]


def normalize_isotope_elements(elements: str | Iterable[str]) -> tuple[str, ...]:
    """Return validated isotope element symbols in caller-supplied order."""
    if isinstance(elements, str):
        compact = re.sub(r"[\s,;]+", "", elements)
        symbols = re.findall(r"[A-Z][a-z]?", compact)
        if not compact or "".join(symbols) != compact:
            raise ValueError(
                "elements must contain valid, case-sensitive element symbols"
            )
    else:
        try:
            symbols = list(elements)
        except TypeError as error:
            raise ValueError("elements must be a string or iterable of symbols") from error
        if not symbols or any(
            not isinstance(symbol, str)
            or re.fullmatch(r"[A-Z][a-z]?", symbol) is None
            for symbol in symbols
        ):
            raise ValueError(
                "elements must contain valid, case-sensitive element symbols"
            )

    symbols = list(dict.fromkeys(symbols))
    invalid = [symbol for symbol in symbols if symbol not in nist_mass]
    if invalid:
        raise ValueError(f"Unknown isotope elements: {invalid}")

    has_m_plus_one = False
    for symbol in symbols:
        natural = [
            isotope
            for isotope, (_, abundance) in nist_mass[symbol].items()
            if isotope > 0 and abundance > 0
        ]
        if not natural:
            raise ValueError(f"Element {symbol} has no natural-isotope data")
        light = min(natural)
        has_m_plus_one |= any(isotope - light == 1 for isotope in natural)
    if not has_m_plus_one:
        raise ValueError(
            "elements must include a naturally occurring M+1 isotope contributor"
        )
    return tuple(symbols)


def _agreement(delta: float, tolerance: float) -> tuple[float, float]:
    margin = tolerance - abs(delta)
    if tolerance == 0:
        return margin, 1.0 if delta == 0 else 0.0
    return margin, max(0.0, margin / tolerance)


class Deisotoper:
    """Extract provisional singly charged isotope patterns from picked peaks.

    Configuration belongs to the extractor, while :meth:`extract` receives a
    Frame-like object and optional peak-picking results. The object does not
    retain extraction state and can be reused across frames.
    """

    def __init__(
        self,
        *,
        polarity: int,
        mass_tolerance_ppm: float = 5.0,
        mass_tolerance_da: float = 0.001,
        mobility_tolerance: float = 0.01,
        mobility_step_tolerance: float | None = None,
        resolve_shared_intensity: bool = False,
        intensity_split_ratio: float = 1.5,
        max_isotope_order: int = 5,
        elements: str | Iterable[str] = "CHNOPS",
    ):
        self.elements = normalize_isotope_elements(elements)
        if mobility_step_tolerance is not None and (
            isinstance(mobility_step_tolerance, (bool, np.bool_))
            or not isinstance(mobility_step_tolerance, Real)
            or not np.isfinite(mobility_step_tolerance)
            or mobility_step_tolerance < 0
        ):
            raise ValueError("mobility_step_tolerance must be finite and nonnegative")
        if (
            isinstance(intensity_split_ratio, (bool, np.bool_))
            or not isinstance(intensity_split_ratio, Real)
            or not np.isfinite(intensity_split_ratio)
            or intensity_split_ratio <= 1
        ):
            raise ValueError("intensity_split_ratio must be finite and greater than one")
        if resolve_shared_intensity and mobility_step_tolerance is None:
            raise ValueError("Intensity resolution requires mobility_step_tolerance")
        if isinstance(polarity, (bool, np.bool_)) or polarity not in (-1, 1):
            raise ValueError("polarity must be +1 or -1")
        for name, value in (
            ("mass_tolerance_ppm", mass_tolerance_ppm),
            ("mass_tolerance_da", mass_tolerance_da),
            ("mobility_tolerance", mobility_tolerance),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite nonnegative number")
        if (
            isinstance(max_isotope_order, (bool, np.bool_))
            or not isinstance(max_isotope_order, Integral)
            or max_isotope_order < 1
        ):
            raise ValueError("max_isotope_order must be an integer >= 1")

        self.polarity = int(polarity)
        self.mass_tolerance_ppm = float(mass_tolerance_ppm)
        self.mass_tolerance_da = float(mass_tolerance_da)
        self.mobility_tolerance = float(mobility_tolerance)
        self.mobility_step_tolerance = (
            None if mobility_step_tolerance is None else float(mobility_step_tolerance)
        )
        self.resolve_shared_intensity = bool(resolve_shared_intensity)
        self.intensity_split_ratio = float(intensity_split_ratio)
        self.max_isotope_order = int(max_isotope_order)
        self.lower, self.upper = self._isotope_intervals()

    def _isotope_intervals(self) -> tuple[np.ndarray, np.ndarray]:
        shifts = []
        for element in self.elements:
            natural = [
                (a, mass)
                for a, (mass, abundance) in nist_mass[element].items()
                if a > 0 and abundance > 0
            ]
            light_a, light_mass = min(natural)
            shifts.extend(
                (a - light_a, mass - light_mass)
                for a, mass in natural
                if a > light_a
            )
        lower = np.full(self.max_isotope_order + 1, np.inf)
        upper = np.full(self.max_isotope_order + 1, -np.inf)
        lower[0] = upper[0] = 0.0
        for order in range(1, self.max_isotope_order + 1):
            for step, shift in shifts:
                if step <= order:
                    lower[order] = min(lower[order], lower[order - step] + shift)
                    upper[order] = max(upper[order], upper[order - step] + shift)
        return lower, upper

    def _settings(self, selected: set[int] | None) -> dict:
        return {
            "polarity": self.polarity,
            "mass_tolerance_ppm": self.mass_tolerance_ppm,
            "mass_tolerance_da": self.mass_tolerance_da,
            "mobility_tolerance": self.mobility_tolerance,
            "mobility_step_tolerance": self.mobility_step_tolerance,
            "resolve_shared_intensity": self.resolve_shared_intensity,
            "intensity_split_ratio": self.intensity_split_ratio,
            "max_isotope_order": self.max_isotope_order,
            "elements": "".join(self.elements),
            "isotope_shift_intervals": list(
                zip(self.lower[1:].tolist(), self.upper[1:].tolist())
            ),
            "peak_ids": None if selected is None else sorted(selected),
        }

    @staticmethod
    def _empty_table(columns):
        table = pd.DataFrame(columns=columns)
        boolean_columns = {
            "sirius_ready", "mass_agrees", "mobility_agrees", "accepted",
            "competing_partner", "shared_peak", "ownership_resolved",
            "ownership_unresolved",
        }
        integer_columns = {
            "pattern_id", "precursor_peak_id", "peak_id", "isotope_index",
            "source_peak_id", "target_peak_id", "source_isotope_index",
            "target_isotope_index", "best_predecessor_peak_id", "apex_scan",
            "apex_scan_delta", "charge", "isotope_count",
            "supporting_link_count", "candidate_count_at_order",
            "active_owner_count", "ownership_decision_count",
            "parent_pattern_id", "child_pattern_id", "shared_peak_id",
        }
        object_columns = {
            "flags", "pattern_flags", "role", "status", "link_type",
            "decision", "reason", "parent_before", "child_before",
            "parent_after", "child_after",
        }
        for column in columns:
            if column in boolean_columns:
                table[column] = table[column].astype(bool)
            elif column in integer_columns:
                table[column] = table[column].astype("int64")
            elif column not in object_columns:
                table[column] = table[column].astype("float64")
        return table

    @classmethod
    def _table(cls, rows, columns):
        return pd.DataFrame(rows, columns=columns) if len(rows) else cls._empty_table(columns)

    def _make_result(
        self,
        settings,
        pattern_rows=(),
        isotope_rows=(),
        mapping_rows=(),
        hypotheses=None,
        decisions=(),
        links=(),
    ):
        patterns = self._table(pattern_rows, PATTERN_COLUMNS)
        patterns["sirius_ready"] = patterns["sirius_ready"].astype(bool)
        result = {
            "patterns": patterns,
            "isotope_peaks": self._table(isotope_rows, ISOTOPE_COLUMNS),
            "seed_mapping": self._table(mapping_rows, MAPPING_COLUMNS),
            "settings": settings,
            "original_hypotheses": (
                self._empty_table(HYPOTHESIS_COLUMNS)
                if hypotheses is None
                else hypotheses
            ),
            "ownership_decisions": self._table(decisions, DECISION_COLUMNS),
            "links": self._table(links, LINK_COLUMNS),
        }
        result["rich_peak_list"] = self._empty_table(RICH_COLUMNS)
        return result

    def extract(
        self,
        frame,
        *,
        peak_ids=None,
        peak_pick_result=None,
        peak_pick_kwargs=None,
    ) -> dict:
        """Extract isotope patterns and measured assignment/link evidence."""
        if peak_pick_result is not None and peak_pick_kwargs is not None:
            raise ValueError("Supply peak_pick_result or peak_pick_kwargs, not both")
        selected = None
        if peak_ids is not None:
            selected = set()
            for peak_id in peak_ids:
                if (
                    isinstance(peak_id, (bool, np.bool_))
                    or not isinstance(peak_id, Integral)
                    or peak_id <= 0
                ):
                    raise ValueError("peak_ids must contain positive integer peak IDs")
                selected.add(int(peak_id))
        settings = self._settings(selected)

        points = frame.data.reset_index(drop=True).copy()
        required = {
            "scan_indices", "tof_indices", "mz_values",
            "mobility_values", "intensity_values",
        }
        if not required.issubset(points.columns):
            raise ValueError("Isotope extraction requires indexed Frame data")
        values = points[list(required)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Frame coordinates and intensities must be finite")
        if (points["mz_values"] <= 0).any() or (points["intensity_values"] < 0).any():
            raise ValueError("Frame m/z must be positive and intensities nonnegative")
        for column in ("scan_indices", "tof_indices"):
            indices = points[column].to_numpy()
            if (indices < 0).any() or (indices != np.floor(indices)).any():
                raise ValueError(f"{column} must contain nonnegative integer indices")
        if peak_pick_result is None:
            if points.empty:
                if selected:
                    raise ValueError("peak_ids are absent from the empty frame")
                return self._make_result(settings)
            temporary_frame = copy(frame)
            temporary_frame.data = points
            kwargs = dict(peak_pick_kwargs or {})
            kwargs.update(return_labels=True, return_extents=True, return_apex=False)
            peak_pick_result = temporary_frame.peakPick(**kwargs)
        if not {"peak_list", "peak_labels"}.issubset(peak_pick_result):
            raise ValueError("peak_pick_result requires peak_list and peak_labels")
        peaks = peak_pick_result["peak_list"].copy()
        labels = np.asarray(peak_pick_result["peak_labels"])
        if (
            labels.shape != (len(points),)
            or not np.issubdtype(labels.dtype, np.integer)
            or (labels < 0).any()
        ):
            raise ValueError("peak_labels must be nonnegative integers aligned to Frame rows")
        if (
            not peaks.index.is_unique
            or any(
                isinstance(i, (bool, np.bool_))
                or not isinstance(i, Integral)
                or i <= 0
                for i in peaks.index
            )
            or set(peaks.index) != set(labels[labels > 0])
        ):
            raise ValueError("peak_list IDs must uniquely match the positive peak_labels")
        if selected is not None and not selected.issubset(peaks.index):
            raise ValueError(f"Unknown peak_ids: {sorted(selected - set(peaks.index))}")
        if peaks.empty:
            return self._make_result(settings)
        summary_columns = ["mz_values", "mobility_values", "total_intensity"]
        if not set(summary_columns).issubset(peaks.columns):
            raise ValueError("peak_list requires m/z, mobility, and total_intensity summaries")
        if (
            not np.isfinite(peaks[summary_columns].to_numpy(dtype=float)).all()
            or (peaks["total_intensity"] <= 0).any()
        ):
            raise ValueError("Picked peaks must have finite summaries and positive intensities")

        points["_peak_id"] = labels
        apex_scans, apex_mobilities, scan_bounds = {}, {}, {}
        for peak_id, group in points.loc[labels > 0].groupby("_peak_id", sort=True):
            weights = group["intensity_values"].to_numpy()
            total = weights.sum()
            if total <= 0:
                raise ValueError(f"Peak {peak_id} has no positive labeled intensity")
            measured = [
                np.average(group[column], weights=weights)
                for column in ("mz_values", "mobility_values")
            ] + [total]
            if not np.allclose(
                peaks.loc[peak_id, summary_columns].to_numpy(dtype=float),
                measured,
                rtol=1e-7,
                atol=1e-10,
            ):
                raise ValueError(f"Peak {peak_id} summaries do not match Frame labels")
            scans = group.groupby("scan_indices", sort=True)["intensity_values"].sum()
            apex_scan = int(scans.idxmax())
            apex_points = group.loc[group["scan_indices"] == apex_scan]
            apex_scans[peak_id] = apex_scan
            apex_mobilities[peak_id] = float(
                np.average(
                    apex_points["mobility_values"],
                    weights=apex_points["intensity_values"],
                )
            )
            scan_bounds[peak_id] = (int(scans.index.min()), int(scans.index.max()))

        peaks = peaks.assign(_id=peaks.index).sort_values(["mz_values", "_id"])
        ids = peaks.index.to_numpy()
        masses = peaks["mz_values"].to_numpy(dtype=float)
        mobilities = peaks["mobility_values"].to_numpy(dtype=float)
        intensities = peaks["total_intensity"].to_numpy(dtype=float)
        apex = np.array([apex_scans[i] for i in ids], dtype=np.int64)
        hypotheses = []
        for root, mass in enumerate(masses):
            assignments = {root: 0}
            for order in range(1, self.max_isotope_order + 1):
                tolerance = max(
                    self.mass_tolerance_da,
                    (mass + self.upper[order]) * self.mass_tolerance_ppm * 1e-6,
                )
                left = np.searchsorted(
                    masses, mass + self.lower[order] - tolerance, side="left"
                )
                right = np.searchsorted(
                    masses, mass + self.upper[order] + tolerance, side="right"
                )
                for candidate in range(max(root + 1, left), right):
                    if self.mobility_step_tolerance is None:
                        compatible = (
                            abs(mobilities[candidate] - mobilities[root])
                            <= self.mobility_tolerance
                        )
                    else:
                        compatible = any(
                            previous_order == order - 1
                            and abs(mobilities[candidate] - mobilities[previous])
                            <= self.mobility_step_tolerance
                            and self.lower[1]
                            - max(
                                self.mass_tolerance_da,
                                (masses[previous] + self.upper[1])
                                * self.mass_tolerance_ppm
                                * 1e-6,
                            )
                            <= masses[candidate] - masses[previous]
                            <= self.upper[1]
                            + max(
                                self.mass_tolerance_da,
                                (masses[previous] + self.upper[1])
                                * self.mass_tolerance_ppm
                                * 1e-6,
                            )
                            for previous, previous_order in assignments.items()
                        )
                    if compatible:
                        if candidate in assignments:
                            raise ValueError("Mass tolerances produce overlapping isotope orders")
                        assignments[candidate] = order
            hypotheses.append(assignments)

        original_hypotheses = pd.DataFrame(
            [
                (int(ids[root]), int(ids[position]), order)
                for root, assignments in enumerate(hypotheses)
                for position, order in assignments.items()
            ],
            columns=HYPOTHESIS_COLUMNS,
        )
        suppressed, decisions = set(), []
        if self.resolve_shared_intensity:
            hypotheses, suppressed, decisions = resolve_paths(
                hypotheses,
                ids,
                masses,
                mobilities,
                intensities,
                settings["isotope_shift_intervals"],
                self.mobility_step_tolerance,
                self.mass_tolerance_ppm,
                self.mass_tolerance_da,
                self.intensity_split_ratio,
            )

        owners = [[] for _ in ids]
        for root, assignments in enumerate(hypotheses):
            if root not in suppressed:
                for position in assignments:
                    owners[position].append(root)
        retained = []
        for root, assignments in enumerate(hypotheses):
            if root in suppressed:
                continue
            redundant = False
            for parent in owners[root]:
                if parent >= root:
                    continue
                parent_assignments = hypotheses[parent]
                if len(assignments) == 1:
                    redundant = True
                    break
                if self.mobility_step_tolerance is not None:
                    continue
                if (
                    len(set(assignments.values())) != len(assignments)
                    or len(set(parent_assignments.values()))
                    != len(parent_assignments)
                ):
                    continue
                offset = parent_assignments[root]
                parent_tail = {
                    position: order - offset
                    for position, order in parent_assignments.items()
                    if order >= offset
                }
                if assignments == parent_tail:
                    redundant = True
                    break
            if not redundant:
                retained.append(root)
        active_owners = [[] for _ in ids]
        for root in retained:
            for candidate in hypotheses[root]:
                active_owners[candidate].append(root)

        mz_domain = None if frame.mz_domain is None else np.asarray(frame.mz_domain)
        mobility_domain = (
            None if frame.mobility_domain is None else np.asarray(frame.mobility_domain)
        )
        known_bounds = (
            mz_domain is not None
            and mobility_domain is not None
            and mz_domain.size > 0
            and mobility_domain.size > 0
            and np.isfinite(mz_domain).all()
            and np.isfinite(mobility_domain).all()
        )
        pattern_rows, isotope_rows, mapping_rows, link_rows = [], [], [], []
        assignment_metadata = {}
        for root in retained:
            assignments = hypotheses[root]
            if selected is not None and not any(
                int(ids[position]) in selected for position in assignments
            ):
                continue
            pattern_id = int(ids[root])
            orders = set(assignments.values())
            flags = []
            if len(orders) == 1:
                flags.append("singleton")
            if orders != set(range(max(orders) + 1)):
                flags.append("missing_isotope")
            competing = len(orders) != len(assignments)
            shared = any(len(active_owners[position]) > 1 for position in assignments)
            if competing:
                flags.append("competing_partners")
            if shared:
                flags.append("shared_peak")
            if not known_bounds:
                flags.append("unknown_acquisition_bounds")
            else:
                tolerance = max(
                    self.mass_tolerance_da,
                    (masses[root] + self.upper[-1])
                    * self.mass_tolerance_ppm
                    * 1e-6,
                )
                if (
                    masses[root] - tolerance <= mz_domain.min()
                    or masses[root] + self.upper[-1] + tolerance >= mz_domain.max()
                    or any(
                        scan_bounds[ids[position]][0] <= 0
                        or scan_bounds[ids[position]][1] >= len(mobility_domain) - 1
                        for position in assignments
                    )
                ):
                    flags.append("acquisition_boundary")
            pattern_rows.append([
                pattern_id,
                pattern_id,
                masses[root],
                mobilities[root],
                int(apex[root]),
                apex_mobilities[ids[root]],
                self.polarity,
                len(orders),
                tuple(flags),
                not flags,
            ])
            largest = max(intensities[position] for position in assignments)
            order_counts = pd.Series(list(assignments.values())).value_counts().to_dict()
            for candidate, order in sorted(
                assignments.items(), key=lambda item: (item[1], item[0])
            ):
                peak_id = int(ids[candidate])
                isotope_rows.append([
                    pattern_id,
                    order,
                    peak_id,
                    masses[candidate],
                    mobilities[candidate],
                    mobilities[candidate] - mobilities[root],
                    intensities[candidate],
                    intensities[candidate] / largest,
                    int(apex[candidate]),
                    apex_mobilities[peak_id],
                    int(apex[candidate]) - int(apex[root]),
                ])
                status = (
                    "ambiguous"
                    if competing or shared
                    else "unmatched"
                    if len(orders) == 1
                    else "matched"
                )
                mapping_rows.append([
                    peak_id,
                    pattern_id,
                    "precursor" if order == 0 else "isotope",
                    status,
                ])
                assignment_metadata[(pattern_id, peak_id)] = {
                    "candidate_count_at_order": int(order_counts[order]),
                    "active_owner_count": len(active_owners[candidate]),
                }
            link_rows.extend(
                self._links_for_pattern(
                    pattern_id, root, assignments, ids, masses, mobilities
                )
            )

        result = self._make_result(
            settings,
            pattern_rows,
            isotope_rows,
            mapping_rows,
            original_hypotheses,
            decisions,
            link_rows,
        )
        result["rich_peak_list"] = self._build_rich_peak_list(
            result, assignment_metadata
        )
        return result

    def _links_for_pattern(
        self, pattern_id, root, assignments, ids, masses, mobilities
    ):
        rows = []
        for target, target_order in sorted(
            assignments.items(), key=lambda item: (item[1], item[0])
        ):
            if target_order == 0:
                continue
            if self.mobility_step_tolerance is None:
                sources = [(root, 0)]
                expected_order = target_order
                link_type = "m0_relative"
                mobility_tolerance = self.mobility_tolerance
            else:
                sources = [
                    (source, source_order)
                    for source, source_order in assignments.items()
                    if source_order == target_order - 1
                ]
                expected_order = 1
                link_type = "adjacent"
                mobility_tolerance = self.mobility_step_tolerance
            for source, source_order in sources:
                low = self.lower[expected_order]
                high = self.upper[expected_order]
                mass_tolerance = max(
                    self.mass_tolerance_da,
                    (masses[source] + high) * self.mass_tolerance_ppm * 1e-6,
                )
                mass_shift = masses[target] - masses[source]
                mass_margin = min(
                    mass_shift - (low - mass_tolerance),
                    high + mass_tolerance - mass_shift,
                )
                mobility_delta = mobilities[target] - mobilities[source]
                mobility_margin, mobility_agreement = _agreement(
                    mobility_delta, mobility_tolerance
                )
                mass_agrees = mass_margin >= -1e-12
                mobility_agrees = mobility_margin >= -1e-12
                rows.append([
                    pattern_id,
                    int(ids[source]),
                    int(ids[target]),
                    int(source_order),
                    int(target_order),
                    link_type,
                    mass_shift,
                    low,
                    high,
                    mass_tolerance,
                    mass_margin,
                    mass_agrees,
                    mobility_delta,
                    mobility_tolerance,
                    mobility_margin,
                    mobility_agreement,
                    mobility_agrees,
                    mass_agrees and mobility_agrees,
                ])
        return rows

    def _build_rich_peak_list(self, result, assignment_metadata):
        isotopes = result["isotope_peaks"]
        if isotopes.empty:
            return self._empty_table(RICH_COLUMNS)
        rich = isotopes.merge(
            result["seed_mapping"],
            on=["pattern_id", "peak_id"],
            how="left",
            validate="one_to_one",
        ).merge(
            result["patterns"][
                ["pattern_id", "precursor_peak_id", "precursor_mz", "flags", "sirius_ready"]
            ],
            on="pattern_id",
            how="left",
            validate="many_to_one",
        )
        intervals = result["settings"]["isotope_shift_intervals"]
        mass_evidence = []
        for row in rich.itertuples():
            if row.isotope_index == 0:
                mass_evidence.append((0.0, np.nan, np.nan, np.nan, np.nan, pd.NA))
                continue
            low, high = intervals[int(row.isotope_index) - 1]
            tolerance = max(
                self.mass_tolerance_da,
                (row.precursor_mz + high) * self.mass_tolerance_ppm * 1e-6,
            )
            shift = row.mz_values - row.precursor_mz
            margin = min(shift - (low - tolerance), high + tolerance - shift)
            mass_evidence.append((shift, low, high, tolerance, margin, margin >= -1e-12))
        for index, name in enumerate([
            "observed_mass_shift", "expected_mass_shift_lower",
            "expected_mass_shift_upper", "mass_tolerance", "mass_margin",
            "mass_agrees",
        ]):
            rich[name] = [values[index] for values in mass_evidence]

        accepted = result["links"].loc[result["links"]["accepted"]].copy()
        best_links = {}
        support_counts = {}
        if not accepted.empty:
            accepted = accepted.sort_values(
                ["pattern_id", "target_peak_id", "mobility_agreement", "source_peak_id"],
                ascending=[True, True, False, True],
            )
            for key, group in accepted.groupby(
                ["pattern_id", "target_peak_id"], sort=False
            ):
                best_links[key] = group.iloc[0]
                support_counts[key] = len(group)

        decisions_by_pattern = {}
        for decision in result["ownership_decisions"].itertuples():
            for pattern_id in (decision.parent_pattern_id, decision.child_pattern_id):
                decisions_by_pattern.setdefault(int(pattern_id), []).append(decision.decision)

        evidence_rows = []
        for row in rich.itertuples():
            key = (int(row.pattern_id), int(row.peak_id))
            link = best_links.get(key)
            metadata = assignment_metadata[key]
            outcomes = decisions_by_pattern.get(int(row.pattern_id), [])
            evidence_rows.append({
                "best_predecessor_peak_id": (
                    pd.NA if link is None else int(link.source_peak_id)
                ),
                "supporting_link_count": support_counts.get(key, 0),
                "step_mobility_delta": np.nan if link is None else link.mobility_delta,
                "mobility_tolerance": np.nan if link is None else link.mobility_tolerance,
                "mobility_margin": np.nan if link is None else link.mobility_margin,
                "mobility_agreement": np.nan if link is None else link.mobility_agreement,
                "mobility_agrees": pd.NA if link is None else bool(link.mobility_agrees),
                **metadata,
                "competing_partner": metadata["candidate_count_at_order"] > 1,
                "shared_peak": metadata["active_owner_count"] > 1,
                "ownership_decision_count": len(outcomes),
                "ownership_resolved": any(value in {"merge", "split"} for value in outcomes),
                "ownership_unresolved": "unresolved" in outcomes,
            })
        evidence = pd.DataFrame(evidence_rows)
        rich = pd.concat([rich.reset_index(drop=True), evidence], axis=1)
        rich = rich.rename(columns={"flags": "pattern_flags"})
        return rich[RICH_COLUMNS]

    def rich_peak_list(self, result: dict) -> pd.DataFrame:
        """Return a copy of the assignment-level measured evidence table."""
        if "rich_peak_list" not in result:
            raise ValueError("result was not produced by Deisotoper.extract")
        return result["rich_peak_list"].copy()
