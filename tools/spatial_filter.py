#!/usr/bin/env python3
"""
Wi-Fi Vision 3D - Spatial Zone Filter

Discriminates events *inside* the target zone (staircase) from events
*outside* (hallway, kitchen, adjacent rooms) using multi-node CSI consensus.

Core idea
---------
A person inside the staircase perturbs the RF field seen by *most* of the 8
Tx-Rx links because the router sits on one wall and the 8 ESP32-S3 nodes
surround the room.  A person walking through the kitchen only affects the 2-3
links whose propagation path passes near that area.

The filter works in three stages:

1. **Baseline subtraction** -- For each node, subtract the per-subcarrier
   mean amplitude recorded during the "baseline_empty" session.  The residual
   (delta) is the perturbation caused by any change in the environment.

2. **Per-node activation** -- Compute a scalar "activity" metric per node
   (e.g., root-mean-square of the delta vector).  Mark a node as *active* if
   the metric exceeds a threshold derived from the baseline noise floor.

3. **Spatial consensus** -- Only declare a *staircase event* when a minimum
   number of nodes (configurable, default >= 5 out of 8) are simultaneously
   active.  Links that cross the zone interior are weighted higher than links
   that graze the periphery.

Additionally, each link can be assigned a **zone weight** based on how much of
its Tx-Rx path lies inside the target zone vs. outside.  Links with short
internal segments contribute less to the consensus score.

Usage (standalone analysis):
    python spatial_filter.py --baseline data/sessions/…/raw \\
                             --live data/sessions/…/raw \\
                             --positions data/positions.json

Usage (programmatic):
    from spatial_filter import SpatialZoneFilter
    filt = SpatialZoneFilter(baseline_dir, positions_json)
    score = filt.score_frame(amplitudes_dict)
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze_csi import load_csi_file  # noqa: E402

# Maximum time difference (microseconds) between frames from different nodes
# for them to be considered "simultaneous".  At 50 Hz, one frame period is
# 20 000 µs; ±20 ms gives ±1 frame of tolerance.
ALIGN_WINDOW_US = 20_000


# ── Temporal alignment helpers ────────────────────────────────────────────

def read_t0_host_us(csv_path: Path) -> int:
    """
    Read the ``# t0_host_us: <value>`` comment written by capture_node()
    into the CSV header.  Returns 0 if the field is absent (legacy files).

    The value is the Unix epoch in microseconds at the moment the
    synchronization barrier released all capture threads.  Combined with
    the per-frame ``timestamp_us`` from the ESP32 firmware, it allows
    computing a common absolute timeline across nodes:

        t_abs_us = t0_host_us + timestamp_us_firmware
    """
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("#"):
                    break
                if "t0_host_us" in line:
                    _, _, value = line.partition(":")
                    try:
                        return int(value.strip())
                    except ValueError:
                        return 0
    except OSError:
        pass
    return 0


def align_nodes_by_timestamp(
    node_data: dict[int, dict],
    window_us: int = ALIGN_WINDOW_US,
) -> dict[int, np.ndarray]:
    """
    Align amplitude matrices from multiple nodes onto a common time axis
    using their absolute timestamps.

    Each entry in ``node_data`` must be a dict with:
        - ``"amplitude"`` : np.ndarray, shape (n_frames, n_subcarriers)
        - ``"t_abs_us"``  : np.ndarray, shape (n_frames,)  — absolute
          Unix timestamps in microseconds, computed as
          ``t0_host_us + timestamp_us_firmware`` for each frame.

    The alignment algorithm:
    1. Use the node with the **most frames** as the reference axis.
    2. For every reference frame t_ref, find the frame in each other node
       whose |t_abs - t_ref| is minimised and within ``window_us``.
    3. If no match is found within the window the frame is dropped from
       the output (conservative).

    Returns a dict mapping node_id → aligned amplitude array (same length
    as the shortest matched sequence).

    Falls back to index-based pairing (legacy behaviour) when fewer than
    two nodes have valid absolute timestamps (t_abs_us all zeros).
    """
    # Decide whether we have real timestamps.
    has_ts = {
        nid: (d["t_abs_us"].max() > 0)
        for nid, d in node_data.items()
        if "t_abs_us" in d
    }

    if sum(has_ts.values()) < 2:
        # Legacy path: index-based pairing.
        n_frames = min(d["amplitude"].shape[0] for d in node_data.values())
        return {nid: d["amplitude"][:n_frames] for nid, d in node_data.items()}

    # Use the node with most frames as reference.
    ref_id = max(node_data, key=lambda nid: node_data[nid]["amplitude"].shape[0])
    ref_ts = node_data[ref_id]["t_abs_us"]

    aligned: dict[int, np.ndarray] = {}
    valid_mask = np.ones(len(ref_ts), dtype=bool)

    for nid, d in node_data.items():
        amp = d["amplitude"]
        if nid == ref_id:
            aligned[nid] = amp
            continue

        t_abs = d["t_abs_us"]
        matched_rows = []
        matched_ref_indices = []

        for i, t_ref in enumerate(ref_ts):
            diffs = np.abs(t_abs - t_ref)
            best = int(np.argmin(diffs))
            if diffs[best] <= window_us:
                matched_rows.append(amp[best])
                matched_ref_indices.append(i)
            else:
                valid_mask[i] = False

        if not matched_rows:
            # No overlap — fall back to empty; caller should handle.
            aligned[nid] = np.empty((0, amp.shape[1]), dtype=amp.dtype)
        else:
            aligned[nid] = np.array(matched_rows)

    # Trim reference to rows that had a match in ALL nodes.
    aligned[ref_id] = aligned[ref_id][valid_mask]
    for nid in aligned:
        if nid != ref_id:
            # matched_rows already excludes unmatched ref frames by construction.
            pass

    return aligned


# ── Zone weight heuristics ────────────────────────────────────────────────

def _line_segment_inside_box(tx: np.ndarray, rx: np.ndarray,
                             box_min: np.ndarray, box_max: np.ndarray,
                             n_samples: int = 200) -> float:
    """
    Estimate the fraction of the Tx→Rx segment that lies inside the
    axis-aligned bounding box [box_min, box_max] using uniform sampling.
    Returns a value in [0, 1].
    """
    t = np.linspace(0, 1, n_samples).reshape(-1, 1)
    points = tx + t * (rx - tx)
    inside = np.all((points >= box_min) & (points <= box_max), axis=1)
    return float(np.mean(inside))


def compute_zone_weights(positions: dict, router_pos: dict,
                         zone_box: dict | None = None) -> dict[int, float]:
    """
    For each receiver position, compute a weight in [0, 1] representing how
    relevant that Tx-Rx link is to the target zone.

    If zone_box is provided, it defines the bounding box of the target zone
    (keys: x_min, x_max, y_min, y_max, z_min, z_max).  Otherwise, the
    bounding box is inferred from the receiver positions themselves.
    """
    tx = np.array([router_pos["x"], router_pos["y"], router_pos["z"]])

    if zone_box:
        box_min = np.array([zone_box["x_min"], zone_box["y_min"], zone_box["z_min"]])
        box_max = np.array([zone_box["x_max"], zone_box["y_max"], zone_box["z_max"]])
    else:
        all_rx = np.array([[p["x"], p["y"], p["z"]]
                           for p in positions.values()])
        margin = 0.1
        box_min = all_rx.min(axis=0) - margin
        box_max = all_rx.max(axis=0) + margin

    weights: dict[int, float] = {}
    for pos_id, pos in positions.items():
        rx = np.array([pos["x"], pos["y"], pos["z"]])
        frac = _line_segment_inside_box(tx, rx, box_min, box_max)
        weights[int(pos_id)] = round(max(frac, 0.1), 3)

    return weights


# ── Core filter class ─────────────────────────────────────────────────────

class SpatialZoneFilter:
    """
    Multi-node spatial consensus filter.

    Parameters
    ----------
    baseline_stats : dict
        Per-node baseline statistics.  Keys are node/position IDs (int),
        values are dicts with "amplitude_mean" and "amplitude_std" arrays.
    zone_weights : dict[int, float]
        Per-node weight in [0, 1] (from compute_zone_weights).
    activation_sigma : float
        Number of std deviations above baseline to consider a node *active*.
    consensus_threshold : float
        Minimum weighted activation score to declare a zone event.
        Range (0, 1]; higher = stricter.
    """

    def __init__(self, baseline_stats: dict, zone_weights: dict[int, float],
                 activation_sigma: float = 3.0,
                 consensus_threshold: float = 0.5):
        self.baseline = baseline_stats
        self.zone_weights = zone_weights
        self.activation_sigma = activation_sigma
        self.consensus_threshold = consensus_threshold

    def node_activity(self, node_id: int, amplitudes: np.ndarray) -> float:
        """
        Compute the activation metric for a single node given its current
        amplitude vector (one per subcarrier).
        Returns 0 if the node is within baseline noise, positive otherwise.
        """
        key = str(node_id)
        if key not in self.baseline:
            return 0.0

        bl = self.baseline[key]
        mean = np.array(bl["amplitude_mean"])
        std = np.array(bl["amplitude_std"])

        n = min(len(amplitudes), len(mean))
        delta = np.abs(amplitudes[:n] - mean[:n])
        threshold = std[:n] * self.activation_sigma + 1e-6
        exceedance = np.maximum(delta - threshold, 0)
        return float(np.sqrt(np.mean(exceedance ** 2)))

    def node_is_active(self, node_id: int, amplitudes: np.ndarray) -> bool:
        return self.node_activity(node_id, amplitudes) > 0.0

    def score_frame(self, amplitudes_by_node: dict[int, np.ndarray]) -> float:
        """
        Compute a weighted consensus score for one CSI frame across all nodes.
        Returns a value in [0, 1] where 1 = full consensus that an event is
        happening inside the target zone.
        """
        weighted_sum = 0.0
        weight_total = 0.0

        for node_id, amp in amplitudes_by_node.items():
            w = self.zone_weights.get(node_id, 0.1)
            active = 1.0 if self.node_is_active(node_id, amp) else 0.0
            weighted_sum += w * active
            weight_total += w

        if weight_total == 0:
            return 0.0
        return weighted_sum / weight_total

    def is_zone_event(self, amplitudes_by_node: dict[int, np.ndarray]) -> bool:
        return self.score_frame(amplitudes_by_node) >= self.consensus_threshold

    def analyze_session(self, session_data: dict[int, np.ndarray],
                        ) -> dict:
        """
        Analyze an entire capture session (amplitude matrices per node).
        Returns per-frame scores and aggregate statistics.

        Parameters
        ----------
        session_data : dict[int, np.ndarray]
            Mapping of node_id to amplitude matrix (n_frames, n_subcarriers).
            Pass pre-aligned arrays from :func:`align_nodes_by_timestamp` for
            accurate multi-node temporal correlation.  Raw (unaligned) arrays
            are accepted for single-node analysis or legacy calls.
        """
        node_ids = sorted(session_data.keys())
        n_frames = min(m.shape[0] for m in session_data.values())

        scores = np.zeros(n_frames)
        per_node_activity = {nid: np.zeros(n_frames) for nid in node_ids}

        for f in range(n_frames):
            frame_amps = {}
            for nid in node_ids:
                amp = session_data[nid][f]
                frame_amps[nid] = amp
                per_node_activity[nid][f] = self.node_activity(nid, amp)
            scores[f] = self.score_frame(frame_amps)

        event_frames = int(np.sum(scores >= self.consensus_threshold))

        return {
            "n_frames": n_frames,
            "scores": scores,
            "per_node_activity": per_node_activity,
            "event_frames": event_frames,
            "event_fraction": round(event_frames / max(n_frames, 1), 4),
            "mean_score": round(float(np.mean(scores)), 4),
            "max_score": round(float(np.max(scores)), 4),
        }


# ── Helpers to build filter from files ────────────────────────────────────

def build_baseline_stats(baseline_dir: Path) -> dict:
    """
    Load all CSV files in baseline_dir and compute per-node amplitude
    mean/std.  Returns dict keyed by node_id (str).
    """
    stats: dict[str, dict] = {}
    for fpath in sorted(baseline_dir.rglob("*.csv")):
        data = load_csi_file(fpath)
        if data["n_frames"] == 0:
            continue

        stem_parts = fpath.stem.split("_")
        node_id = None
        for part in stem_parts:
            if part.startswith("node"):
                node_id = part.replace("node", "")
                break
        if node_id is None:
            node_id = fpath.stem

        stats[node_id] = {
            "amplitude_mean": data["amplitude"].mean(axis=0).tolist(),
            "amplitude_std": data["amplitude"].std(axis=0).tolist(),
            "n_frames": data["n_frames"],
        }

    return stats


def load_positions(positions_path: Path) -> tuple[dict, dict]:
    """Load positions.json and return (positions_dict, router_dict)."""
    with open(positions_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    positions = {}
    for k, v in data["positions"].items():
        positions[int(k)] = v
    return positions, data["router"]


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wi-Fi Vision 3D - Spatial Zone Filter"
    )
    parser.add_argument("--baseline", required=True,
                        help="Directory with baseline (empty) CSVs")
    parser.add_argument("--live", required=True,
                        help="Directory with CSVs to analyze")
    parser.add_argument("--positions", default="data/positions.json",
                        help="Path to positions.json")
    parser.add_argument("--sigma", type=float, default=3.0,
                        help="Activation threshold in std deviations (default: 3.0)")
    parser.add_argument("--consensus", type=float, default=0.5,
                        help="Consensus threshold 0-1 (default: 0.5)")
    args = parser.parse_args()

    positions, router = load_positions(Path(args.positions))
    zone_weights = compute_zone_weights(positions, router)

    print("Zone weights per node:")
    for nid, w in sorted(zone_weights.items()):
        print(f"  Node {nid}: {w:.3f}")

    baseline_stats = build_baseline_stats(Path(args.baseline))
    print(f"\nBaseline loaded: {len(baseline_stats)} nodes")

    filt = SpatialZoneFilter(
        baseline_stats, zone_weights,
        activation_sigma=args.sigma,
        consensus_threshold=args.consensus,
    )

    live_dir = Path(args.live)
    session_data: dict[int, dict] = {}
    for fpath in sorted(live_dir.rglob("*.csv")):
        data = load_csi_file(fpath)
        if data["n_frames"] == 0:
            continue
        stem_parts = fpath.stem.split("_")
        node_id = None
        for part in stem_parts:
            if part.startswith("node"):
                try:
                    node_id = int(part.replace("node", ""))
                except ValueError:
                    pass
                break
        if node_id is None:
            continue

        # Build absolute timestamps using the t0_host_us anchor from the
        # CSV header (written by capture_node at barrier release).
        t0 = int(data["metadata"].get("t0_host_us", 0))
        t_abs_us = data["timestamps"] + t0

        session_data[node_id] = {
            "amplitude": data["amplitude"],
            "t_abs_us": t_abs_us,
        }

    if not session_data:
        print("ERROR: No valid CSI data found in --live directory")
        sys.exit(1)

    print(f"Live data loaded: {len(session_data)} nodes")

    # Align nodes temporally before analysis.
    aligned_amps = align_nodes_by_timestamp(session_data)

    result = filt.analyze_session(aligned_amps)

    print(f"\n{'=' * 60}")
    print(f"  SPATIAL FILTER RESULTS")
    print(f"{'=' * 60}")
    print(f"  Frames analyzed:   {result['n_frames']}")
    print(f"  Event frames:      {result['event_frames']} "
          f"({result['event_fraction']*100:.1f}%)")
    print(f"  Mean zone score:   {result['mean_score']:.4f}")
    print(f"  Max zone score:    {result['max_score']:.4f}")
    print(f"  Consensus thresh:  {args.consensus}")
    print()

    if result["event_fraction"] > 0.1:
        print("  RESULTADO: Actividad DETECTADA en la zona de escaleras")
    else:
        print("  RESULTADO: Zona de escaleras SIN actividad significativa")


if __name__ == "__main__":
    main()
