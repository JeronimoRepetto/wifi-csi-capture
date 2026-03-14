#!/usr/bin/env python3
"""
Wi-Fi Vision 3D - CSI Data Analysis Tool

Analyzes captured CSI data to:
- Validate signal quality per node/position
- Compare empty room vs. human presence signatures
- Generate the electromagnetic baseline map for digital twin calibration
- Export statistics for Sionna parameter tuning

Usage:
    python analyze_csi.py --data-dir data/
    python analyze_csi.py --file data/pos01_node01_20260313.csv
    python analyze_csi.py --compare data/round1_empty data/round1_person
"""

import argparse
import csv
import json
import sys
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


CSI_LINE_PREFIX = "CSI_DATA,"
CSI_HEADER_FIELDS = 18


def load_csi_file(filepath: Path) -> dict:
    """Load a CSI CSV file and return structured data."""
    amplitudes = []
    phases = []
    rssis = []
    timestamps = []
    metadata = {}

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line.startswith("# "):
                if ":" in line:
                    key, _, val = line[2:].partition(":")
                    metadata[key.strip()] = val.strip()
                continue

            if line.startswith("timestamp_us"):
                continue

            parts = line.split(",")
            if len(parts) < CSI_HEADER_FIELDS + 1:
                continue

            try:
                ts = int(parts[0])
                rssi = int(parts[2])

                csi_str = parts[CSI_HEADER_FIELDS]
                csi_vals = [int(v) for v in csi_str.strip().split() if v.strip()]

                if len(csi_vals) < 4:
                    continue

                n_pairs = len(csi_vals) // 2
                amp = np.zeros(n_pairs)
                phase = np.zeros(n_pairs)

                for i in range(n_pairs):
                    imag = csi_vals[2 * i]
                    real = csi_vals[2 * i + 1]
                    amp[i] = np.sqrt(real ** 2 + imag ** 2)
                    phase[i] = np.arctan2(imag, real)

                amplitudes.append(amp)
                phases.append(np.unwrap(phase))
                rssis.append(rssi)
                timestamps.append(ts)

            except (ValueError, IndexError):
                continue

    if not amplitudes:
        return {"error": f"No valid CSI frames in {filepath}", "n_frames": 0}

    max_len = max(len(a) for a in amplitudes)
    amp_matrix = np.zeros((len(amplitudes), max_len))
    phase_matrix = np.zeros((len(phases), max_len))
    for i in range(len(amplitudes)):
        amp_matrix[i, :len(amplitudes[i])] = amplitudes[i]
        phase_matrix[i, :len(phases[i])] = phases[i]

    ts_arr = np.array(timestamps)
    if len(ts_arr) > 1:
        intervals = np.diff(ts_arr) / 1e6
        avg_hz = 1.0 / np.mean(intervals) if np.mean(intervals) > 0 else 0
    else:
        avg_hz = 0

    return {
        "filepath": str(filepath),
        "metadata": metadata,
        "n_frames": len(amplitudes),
        "n_subcarriers": max_len,
        "amplitude": amp_matrix,
        "phase": phase_matrix,
        "rssi": np.array(rssis),
        "timestamps": ts_arr,
        "avg_hz": avg_hz,
    }


def analyze_single(data: dict):
    """Print analysis of a single capture file."""
    print(f"\n{'=' * 60}")
    print(f"  File: {data['filepath']}")
    print(f"{'=' * 60}")

    if data["n_frames"] == 0:
        print(f"  ERROR: {data.get('error', 'No frames')}")
        return

    for k, v in data.get("metadata", {}).items():
        print(f"  {k}: {v}")

    print(f"\n  Frames:        {data['n_frames']}")
    print(f"  Subcarriers:   {data['n_subcarriers']}")
    print(f"  Avg rate:      {data['avg_hz']:.1f} Hz")
    print(f"  RSSI mean:     {np.mean(data['rssi']):.1f} dBm")
    print(f"  RSSI std:      {np.std(data['rssi']):.2f} dBm")
    print(f"  RSSI range:    [{np.min(data['rssi'])}, {np.max(data['rssi'])}] dBm")

    amp = data["amplitude"]
    print(f"\n  Amplitude stats (across all frames):")
    print(f"    Mean:    {np.mean(amp):.2f}")
    print(f"    Std:     {np.std(amp):.2f}")
    print(f"    Max:     {np.max(amp):.2f}")

    amp_variance_per_sc = np.var(amp, axis=0)
    top_variance = np.argsort(amp_variance_per_sc)[-5:][::-1]
    print(f"\n  Most dynamic subcarriers (highest variance):")
    for idx in top_variance:
        print(f"    SC[{idx:3d}]: variance = {amp_variance_per_sc[idx]:.2f}")


def compare_captures(empty_data: dict, person_data: dict):
    """Compare empty room vs human presence CSI data."""
    print(f"\n{'=' * 60}")
    print(f"  COMPARISON: Empty vs. Human Presence")
    print(f"{'=' * 60}")

    if empty_data["n_frames"] == 0 or person_data["n_frames"] == 0:
        print("  ERROR: Cannot compare - one or both datasets are empty")
        return

    n_sc = min(empty_data["n_subcarriers"], person_data["n_subcarriers"])

    empty_amp = empty_data["amplitude"][:, :n_sc]
    person_amp = person_data["amplitude"][:, :n_sc]

    empty_mean = np.mean(empty_amp, axis=0)
    person_mean = np.mean(person_amp, axis=0)
    empty_var = np.var(empty_amp, axis=0)
    person_var = np.var(person_amp, axis=0)

    amplitude_delta = np.abs(person_mean - empty_mean)
    variance_ratio = person_var / (empty_var + 1e-10)

    print(f"\n  Empty room:  {empty_data['n_frames']} frames, "
          f"RSSI={np.mean(empty_data['rssi']):.1f} dBm")
    print(f"  With person: {person_data['n_frames']} frames, "
          f"RSSI={np.mean(person_data['rssi']):.1f} dBm")

    rssi_change = np.mean(person_data["rssi"]) - np.mean(empty_data["rssi"])
    print(f"\n  RSSI change: {rssi_change:+.1f} dBm")

    print(f"\n  Mean amplitude delta (person - empty):")
    print(f"    Average: {np.mean(amplitude_delta):.2f}")
    print(f"    Max:     {np.max(amplitude_delta):.2f}")
    print(f"    Min:     {np.min(amplitude_delta):.2f}")

    significant = np.sum(amplitude_delta > 2.0)
    print(f"\n  Subcarriers with significant change (delta > 2.0): "
          f"{significant}/{n_sc} ({100*significant/n_sc:.1f}%)")

    print(f"\n  Variance ratio (person/empty):")
    print(f"    Mean: {np.mean(variance_ratio):.2f}x")
    print(f"    Max:  {np.max(variance_ratio):.2f}x")

    high_variance = np.sum(variance_ratio > 2.0)
    print(f"    Subcarriers with >2x variance increase: "
          f"{high_variance}/{n_sc} ({100*high_variance/n_sc:.1f}%)")

    if np.mean(amplitude_delta) > 1.0 or high_variance > n_sc * 0.2:
        print(f"\n  RESULTADO: La presencia humana ES detectable en CSI")
    else:
        print(f"\n  RESULTADO: Señal debil - verificar posiciones de nodos")

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle("CSI Comparison: Empty Room vs. Human Presence",
                 fontsize=14, fontweight="bold")

    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(empty_mean, label="Empty", alpha=0.8, linewidth=0.8)
    ax1.plot(person_mean, label="Person", alpha=0.8, linewidth=0.8)
    ax1.set_title("Mean Amplitude per Subcarrier")
    ax1.set_xlabel("Subcarrier Index")
    ax1.set_ylabel("Amplitude")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(range(n_sc), amplitude_delta, width=1.0, alpha=0.7, color="orange")
    ax2.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="Threshold")
    ax2.set_title("Absolute Amplitude Difference")
    ax2.set_xlabel("Subcarrier Index")
    ax2.set_ylabel("|Delta|")
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(range(n_sc), empty_var, width=1.0, alpha=0.6, label="Empty")
    ax3.bar(range(n_sc), person_var, width=1.0, alpha=0.4, label="Person")
    ax3.set_title("Amplitude Variance per Subcarrier")
    ax3.set_xlabel("Subcarrier Index")
    ax3.set_ylabel("Variance")
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.bar(range(n_sc), variance_ratio, width=1.0, alpha=0.7, color="green")
    ax4.axhline(y=2.0, color="red", linestyle="--", alpha=0.5, label="2x threshold")
    ax4.set_title("Variance Ratio (Person / Empty)")
    ax4.set_xlabel("Subcarrier Index")
    ax4.set_ylabel("Ratio")
    ax4.set_ylim(0, min(np.max(variance_ratio) * 1.2, 20))
    ax4.legend()

    ax5 = fig.add_subplot(gs[2, 0])
    t_max = min(200, empty_amp.shape[0])
    ax5.imshow(empty_amp[:t_max, :].T, aspect="auto", cmap="viridis", origin="lower")
    ax5.set_title(f"Empty Room Spectrogram (first {t_max} frames)")
    ax5.set_xlabel("Frame")
    ax5.set_ylabel("Subcarrier")

    ax6 = fig.add_subplot(gs[2, 1])
    t_max = min(200, person_amp.shape[0])
    ax6.imshow(person_amp[:t_max, :].T, aspect="auto", cmap="viridis", origin="lower")
    ax6.set_title(f"Human Presence Spectrogram (first {t_max} frames)")
    ax6.set_xlabel("Frame")
    ax6.set_ylabel("Subcarrier")

    plt.savefig("csi_comparison.png", dpi=150, bbox_inches="tight")
    print(f"\n  Grafico guardado: csi_comparison.png")
    plt.show()


def export_baseline(data_dir: Path, output_path: Path):
    """Export the electromagnetic baseline for digital twin calibration."""
    baseline = {}
    csv_files = sorted(data_dir.rglob("*.csv"))

    for fpath in csv_files:
        if "empty" not in str(fpath).lower() and "round" not in str(fpath).lower():
            continue

        data = load_csi_file(fpath)
        if data["n_frames"] == 0:
            continue

        node_id = fpath.stem.split("_")[1] if "_" in fpath.stem else fpath.stem
        baseline[node_id] = {
            "file": str(fpath),
            "n_frames": data["n_frames"],
            "n_subcarriers": data["n_subcarriers"],
            "avg_hz": data["avg_hz"],
            "rssi_mean": float(np.mean(data["rssi"])),
            "rssi_std": float(np.std(data["rssi"])),
            "amplitude_mean": data["amplitude"].mean(axis=0).tolist(),
            "amplitude_std": data["amplitude"].std(axis=0).tolist(),
            "phase_mean": data["phase"].mean(axis=0).tolist(),
            "phase_std": data["phase"].std(axis=0).tolist(),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)

    print(f"\nBaseline exported to: {output_path}")
    print(f"Nodes processed: {len(baseline)}")


def spatial_analysis(baseline_dir: Path, live_dir: Path,
                     positions_path: Path, sigma: float = 3.0,
                     consensus: float = 0.5):
    """
    Run the spatial zone filter over a live session, comparing against a
    baseline.  Prints per-node activity and overall zone score.
    """
    from spatial_filter import (
        SpatialZoneFilter, build_baseline_stats,
        load_positions, compute_zone_weights,
    )

    positions, router = load_positions(positions_path)
    zone_weights = compute_zone_weights(positions, router)
    baseline_stats = build_baseline_stats(baseline_dir)

    filt = SpatialZoneFilter(
        baseline_stats, zone_weights,
        activation_sigma=sigma,
        consensus_threshold=consensus,
    )

    session_data: dict[int, np.ndarray] = {}
    for fpath in sorted(live_dir.rglob("*.csv")):
        data = load_csi_file(fpath)
        if data["n_frames"] == 0:
            continue
        for part in fpath.stem.split("_"):
            if part.startswith("node"):
                try:
                    nid = int(part.replace("node", ""))
                    session_data[nid] = data["amplitude"]
                except ValueError:
                    pass
                break

    if not session_data:
        print("ERROR: No valid CSI data in live directory")
        return

    result = filt.analyze_session(session_data)

    print(f"\n{'=' * 60}")
    print(f"  SPATIAL ZONE ANALYSIS")
    print(f"{'=' * 60}")
    print(f"  Baseline nodes:    {len(baseline_stats)}")
    print(f"  Live nodes:        {len(session_data)}")
    print(f"  Frames analyzed:   {result['n_frames']}")
    print(f"  Event frames:      {result['event_frames']} "
          f"({result['event_fraction']*100:.1f}%)")
    print(f"  Mean zone score:   {result['mean_score']:.4f}")
    print(f"  Max zone score:    {result['max_score']:.4f}")

    print(f"\n  Per-node peak activity:")
    for nid in sorted(result["per_node_activity"].keys()):
        peak = float(np.max(result["per_node_activity"][nid]))
        w = zone_weights.get(nid, 0)
        print(f"    Node {nid:>2d}  (weight {w:.2f}):  peak = {peak:.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Wi-Fi Vision 3D - CSI Data Analysis Tool"
    )
    parser.add_argument("--file", help="Analyze a single CSV file")
    parser.add_argument("--data-dir", help="Analyze all CSV files in a directory")
    parser.add_argument("--compare", nargs=2, metavar=("EMPTY_DIR", "PERSON_DIR"),
                        help="Compare empty vs person capture directories")
    parser.add_argument("--export-baseline", default=None,
                        help="Export baseline JSON for digital twin calibration")
    parser.add_argument("--spatial", nargs=2, metavar=("BASELINE_DIR", "LIVE_DIR"),
                        help="Run spatial zone filter (baseline vs live)")
    parser.add_argument("--positions", default="data/positions.json",
                        help="Path to positions.json (for --spatial)")
    args = parser.parse_args()

    if args.file:
        data = load_csi_file(Path(args.file))
        analyze_single(data)

    elif args.data_dir:
        data_path = Path(args.data_dir)
        csv_files = sorted(data_path.rglob("*.csv"))
        if not csv_files:
            print(f"No CSV files found in {data_path}")
            sys.exit(1)

        print(f"Found {len(csv_files)} CSV files in {data_path}\n")
        for fpath in csv_files:
            data = load_csi_file(fpath)
            analyze_single(data)

        if args.export_baseline:
            export_baseline(data_path, Path(args.export_baseline))

    elif args.compare:
        empty_dir = Path(args.compare[0])
        person_dir = Path(args.compare[1])

        empty_files = sorted(empty_dir.rglob("*.csv"))
        person_files = sorted(person_dir.rglob("*.csv"))

        if not empty_files or not person_files:
            print("ERROR: No CSV files found in one or both directories")
            sys.exit(1)

        empty_data = load_csi_file(empty_files[0])
        person_data = load_csi_file(person_files[0])

        analyze_single(empty_data)
        analyze_single(person_data)
        compare_captures(empty_data, person_data)

    elif args.spatial:
        spatial_analysis(
            Path(args.spatial[0]), Path(args.spatial[1]),
            Path(args.positions),
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
