"""Conservative, auditable ownership decisions for adjacent isotope paths."""

import json
import numpy as np


def resolve_paths(hypotheses, ids, masses, mobilities, intensities, intervals,
                  step_tolerance, ppm, da, split_ratio=1.5):
    """Merge descending paths or split at a >= split_ratio intensity reset.

    These are experimental heuristics, not chemical identification. Branches,
    incompatible assignments, and weak intensity evidence remain unresolved.
    Input hypotheses are never mutated; every attempted junction is recorded.
    """
    paths = [dict(path) for path in hypotheses]
    suppressed = set()
    decisions = []
    def encoded(path):
        return json.dumps([[int(ids[p]), int(k)] for p, k in sorted(path.items(), key=lambda x: x[1])])
    for parent in range(len(paths)):
        if parent in suppressed:
            continue
        for child in sorted(list(paths[parent])):
            if child == parent or child in suppressed or len(paths[child]) < 2:
                continue
            a, b = paths[parent], paths[child]
            if child not in a:
                continue
            offset = a[child]
            record = {
                "parent_pattern_id": int(ids[parent]),
                "child_pattern_id": int(ids[child]),
                "shared_peak_id": int(ids[child]),
                "decision": "unresolved",
                "reason": "branching_or_inconsistent_orders",
                "junction_intensity_ratio": None,
                "max_step_mobility_delta": None,
                "total_mobility_delta": None,
                "parent_before": encoded(a),
                "child_before": encoded(b),
                "parent_after": encoded(a),
                "child_after": encoded(b),
            }
            decisions.append(record)
            if (len(set(a.values())) != len(a)
                    or len(set(b.values())) != len(b)):
                continue
            shifted = {p: k + offset for p, k in b.items()}
            if any(p in a and a[p] != k for p, k in shifted.items()):
                continue
            union = {**a, **shifted}
            if (len(set(union.values())) != len(union)
                    or set(union.values()) != set(range(max(union.values()) + 1))):
                continue
            ordered = sorted(union, key=union.get)
            previous = ordered[offset - 1]
            ratio = float(intensities[child] / intensities[previous])
            record["junction_intensity_ratio"] = ratio
            steps = np.diff(mobilities[ordered])
            record["max_step_mobility_delta"] = float(np.abs(steps).max())
            record["total_mobility_delta"] = float(mobilities[ordered[-1]] - mobilities[parent])
            # A strong new maximum followed by a descending child envelope
            # supports exclusive reassignment of the entire tail to the child.
            child_ordered = sorted(b, key=b.get)
            if (ratio >= split_ratio
                    and np.all(np.diff(intensities[child_ordered]) < 0)):
                if not {p for p, k in a.items() if k >= offset}.issubset(b):
                    record['reason'] = 'split_would_discard_parent_tail'
                    continue
                paths[parent] = {p: k for p, k in a.items() if k < offset}
                record.update(decision="split", reason="intensity_reset_and_descending_child")
                record["parent_after"] = encoded(paths[parent])
                continue
            if max(union.values()) > len(intervals):
                record["reason"] = "merged_order_exceeds_search_limit"
                continue
            if np.any(np.abs(steps) > step_tolerance):
                record["reason"] = "adjacent_mobility_failed"
                continue
            valid_mass = True
            for p in ordered[1:]:
                low, high = intervals[union[p] - 1]
                tolerance = max(da, (masses[parent] + high) * ppm * 1e-6)
                shift = masses[p] - masses[parent]
                if not low - tolerance <= shift <= high + tolerance:
                    valid_mass = False
                    break
            if not valid_mass:
                record["reason"] = "M0_relative_mass_failed"
                continue
            if np.all(np.diff(intensities[ordered]) < 0):
                paths[parent] = union
                suppressed.add(child)
                record.update(decision="merge", reason="descending_intensity_continuation")
                record.update(parent_after=encoded(union), child_after="[]")
            else:
                record["reason"] = "insufficient_intensity_evidence"
    return paths, suppressed, decisions
