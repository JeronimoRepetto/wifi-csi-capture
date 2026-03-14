#!/usr/bin/env python3
"""
Wi-Fi Vision 3D - Session Recorder

One command to record CSI data from all connected ESP32-S3 nodes for a fixed
duration.  Creates a timestamped session directory with raw CSVs and a JSON
manifest that makes the recording fully traceable and reusable for analysis,
baseline export, and digital twin calibration.

Typical workflow (2 nodes, sequential rounds):
    python record_session.py --scenario baseline_empty --duration 300 --round 1
    python record_session.py --scenario baseline_empty --duration 300 --round 2
    python record_session.py --scenario baseline_empty --duration 300 --round 3
    python record_session.py --scenario baseline_empty --duration 300 --round 4

Future workflow (8 nodes, parallel):
    python record_session.py --scenario baseline_empty --duration 300

Scenarios:
    baseline_empty   Room with no people (electromagnetic baseline)
    stairs_walk      Person walking up/down the stairs
    stairs_still     Person standing still on the stairs
    custom           Free-form label for any other test
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import serial
    import serial.tools.list_ports
except ImportError:
    print("ERROR: pyserial is required.  Install with: pip install pyserial")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent))
from capture_csi import launch_parallel_capture, BAUD_RATE  # noqa: E402

SCENARIOS = ["baseline_empty", "stairs_walk", "stairs_still", "custom"]

ESP32_KNOWN_VIDS = {0x303A, 0x10C4, 0x1A86}

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_DATASET_LABEL = "capture"


# ── Input helpers ────────────────────────────────────────────────────────────

def sanitize_dataset_label(label: str, max_len: int = 48) -> str:
    """Sanitize a user-provided label so it is safe for Windows file names."""
    normalized = label.strip().lower().replace(" ", "_")
    safe = re.sub(r"[^a-z0-9_-]+", "", normalized)
    safe = re.sub(r"_+", "_", safe).strip("_-")
    if not safe:
        raise ValueError("Dataset label cannot be empty after sanitization.")
    return safe[:max_len]


def parse_duration_seconds(raw_value: str | float | int) -> float:
    """
    Parse duration accepting:
      - seconds as numeric: 300, 120.5
      - suffixed strings: 300s, 5m, 2.5min
    """
    if isinstance(raw_value, (int, float)):
        value = float(raw_value)
        if value <= 0:
            raise ValueError("Duration must be greater than zero.")
        return value

    text = str(raw_value).strip().lower()
    if not text:
        raise ValueError("Duration cannot be empty.")

    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([a-z]*)", text)
    if not m:
        raise ValueError(f"Invalid duration format: {raw_value}")

    amount = float(m.group(1))
    unit = m.group(2) or "s"
    if amount <= 0:
        raise ValueError("Duration must be greater than zero.")

    if unit in {"s", "sec", "secs", "second", "seconds"}:
        return amount
    if unit in {"m", "min", "mins", "minute", "minutes"}:
        return amount * 60.0
    raise ValueError(f"Unsupported duration unit: {unit}")


def build_session_name(scenario: str, dataset_label: str,
                       round_id: int | None,
                       now_fn=datetime.now) -> str:
    """Build deterministic dataset-ready session names."""
    ts = now_fn().strftime("%Y%m%d_%H%M%S")
    round_tag = f"_round{round_id}" if round_id else ""
    return f"{ts}_{scenario}_{dataset_label}{round_tag}"


def ensure_unique_session_dir(sessions_root: Path, session_name: str) -> Path:
    """Return a unique session directory path by adding numeric suffixes."""
    candidate = sessions_root / session_name
    idx = 1
    while candidate.exists():
        candidate = sessions_root / f"{session_name}_{idx:02d}"
        idx += 1
    return candidate


def _prompt_with_default(prompt: str, default: str,
                         input_fn=input) -> str:
    answer = input_fn(f"{prompt} [{default}]: ").strip()
    return answer or default


def resolve_runtime_inputs(args: argparse.Namespace, input_fn=input):
    """
    Hybrid mode:
      - If critical fields were passed via CLI, run non-interactive.
      - Otherwise ask only for missing values.
    """
    scenario = args.scenario
    duration_s = args.duration
    data_root = args.data_root
    dataset_label = args.dataset_label

    # Critical fields for a beginner-friendly, dataset-ready run.
    capture_mode = "cli_flags"
    if not (scenario and duration_s and data_root and dataset_label):
        capture_mode = "interactive"
        print("\n[Interactive setup] Complete the missing fields.\n")

        if not scenario:
            print(f"Available scenarios: {', '.join(SCENARIOS)}")
            while True:
                scenario_input = _prompt_with_default(
                    "Scenario", "baseline_empty", input_fn=input_fn
                )
                if scenario_input in SCENARIOS:
                    scenario = scenario_input
                    break
                print(f"Invalid scenario. Choose one of: {', '.join(SCENARIOS)}")

        if not duration_s:
            while True:
                duration_input = _prompt_with_default(
                    "Duration (e.g. 300, 120s, 5m)", "300", input_fn=input_fn
                )
                try:
                    duration_s = parse_duration_seconds(duration_input)
                    break
                except ValueError as e:
                    print(f"Invalid duration: {e}")
        else:
            duration_s = parse_duration_seconds(duration_s)

        if not data_root:
            data_root = _prompt_with_default(
                "Data root folder", str(DEFAULT_DATA_ROOT), input_fn=input_fn
            )

        if not dataset_label:
            while True:
                raw_label = _prompt_with_default(
                    "Dataset label (human-readable)", DEFAULT_DATASET_LABEL,
                    input_fn=input_fn
                )
                try:
                    dataset_label = sanitize_dataset_label(raw_label)
                    break
                except ValueError as e:
                    print(f"Invalid label: {e}")
    else:
        duration_s = parse_duration_seconds(duration_s)
        dataset_label = sanitize_dataset_label(dataset_label)

    return {
        "scenario": scenario,
        "duration_s": duration_s,
        "data_root": Path(data_root).expanduser(),
        "dataset_label": dataset_label,
        "capture_mode": capture_mode,
    }


# ── Port auto-detection ────────────────────────────────────────────────────

def detect_esp32_ports(expected: int | None = None) -> list[str]:
    """
    Scan serial ports and return those that look like ESP32-S3 boards.
    Filters by known USB VID values (Espressif, SiLabs CP2102, CH340).
    Falls back to listing all ports if no VID match is found.
    """
    ports = serial.tools.list_ports.comports()
    matched: list[str] = []

    for p in ports:
        if p.vid and p.vid in ESP32_KNOWN_VIDS:
            matched.append(p.device)

    if not matched:
        candidates = [
            p.device for p in ports
            if p.description and ("USB" in p.description.upper()
                                  or "SERIAL" in p.description.upper())
        ]
        if candidates:
            print(f"[Auto-detect] No ports matched ESP32 VIDs.  "
                  f"Falling back to USB/serial ports: {candidates}")
            matched = candidates

    matched.sort()

    if expected is not None and len(matched) != expected:
        print(f"[WARNING] Expected {expected} ESP32 ports but detected "
              f"{len(matched)}: {matched}")

    return matched


# ── Round → position mapping (reuses measurement_protocol.py layout) ──────

ROUND_POSITIONS = {
    1: [1, 2],
    2: [3, 4],
    3: [5, 6],
    4: [7, 8],
}


def build_port_node_map(ports: list[str],
                        round_id: int | None,
                        manual_positions: list[int] | None
                        ) -> list[tuple[str, int, int]]:
    """
    Map detected ports to (port, node_id, position_id) tuples.

    Priority:
      1) --positions flag (explicit list)
      2) --round flag (uses ROUND_POSITIONS lookup)
      3) auto-assign sequential positions starting at 1
    """
    if manual_positions:
        positions = manual_positions
    elif round_id and round_id in ROUND_POSITIONS:
        positions = ROUND_POSITIONS[round_id]
    else:
        positions = list(range(1, len(ports) + 1))

    if len(ports) > len(positions):
        print(f"[WARNING] More ports ({len(ports)}) than positions "
              f"({len(positions)}).  Extra ports will be ignored.")
        ports = ports[:len(positions)]
    elif len(ports) < len(positions):
        print(f"[WARNING] Fewer ports ({len(ports)}) than positions "
              f"({len(positions)}).  Some positions will be skipped.")
        positions = positions[:len(ports)]

    mapping = []
    for port, pos_id in zip(ports, positions):
        node_id = pos_id
        mapping.append((port, node_id, pos_id))

    return mapping


# ── Session manifest ──────────────────────────────────────────────────────

def write_manifest(session_dir: Path, scenario: str, duration: float,
                   round_id: int | None, port_node_map, results,
                   notes: str, dataset_label: str = "",
                   capture_mode: str = "cli_flags",
                   data_root_resolved: str = "",
                   operator: str = ""):
    """
    Write session_manifest.json with full traceability metadata.
    """
    manifest = {
        "session_id": session_dir.name,
        "scenario": scenario,
        "dataset_label": dataset_label,
        "capture_mode": capture_mode,
        "round": round_id,
        "duration_requested_s": duration,
        "start_utc": datetime.now(tz=timezone.utc).isoformat(),
        "data_root_resolved": data_root_resolved,
        "operator": operator,
        "notes": notes,
        "nodes": [],
    }

    for res in results:
        manifest["nodes"].append(res.to_dict())

    total_frames = sum(r.frame_count for r in results)
    ok_nodes = sum(1 for r in results if r.success)

    manifest["summary"] = {
        "total_nodes": len(results),
        "successful_nodes": ok_nodes,
        "total_frames": total_frames,
    }

    manifest_path = session_dir / "session_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nManifest saved: {manifest_path}")
    return manifest


# ── CLI ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Wi-Fi Vision 3D - Session Recorder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Record 5-min baseline with 2 ESP32 (round 1 = positions 1,2)\n"
            "  python record_session.py --scenario baseline_empty --duration 300 --round 1\n\n"
            "  # Record person walking for 2 min (round 1)\n"
            "  python record_session.py --scenario stairs_walk --duration 120 --round 1\n\n"
            "  # Record with explicit ports and positions\n"
            "  python record_session.py --scenario baseline_empty --duration 300 "
            "--ports COM3,COM5 --positions 5,6\n\n"
            "  # Record all 8 nodes at once (auto-detect)\n"
            "  python record_session.py --scenario baseline_empty --duration 300\n\n"
            "  # Save data to a different drive (e.g. D: with more space)\n"
            "  python record_session.py --scenario baseline_empty --duration 300 "
            "--round 1 --data-root D:\\csi_data\n"
        )
    )
    parser.add_argument("--scenario", default=None, choices=SCENARIOS,
                        help="Capture scenario label")
    parser.add_argument("--duration", default=None,
                        help="Recording duration. Accepts seconds or suffixes "
                             "(e.g. 300, 120s, 5m)")
    parser.add_argument("--round", type=int, choices=[1, 2, 3, 4], default=None,
                        help="Round ID (1-4) for sequential 2-node capture")
    parser.add_argument("--ports", default=None,
                        help="Comma-separated serial ports (e.g. COM3,COM4). "
                             "If omitted, auto-detects ESP32-S3 ports.")
    parser.add_argument("--positions", default=None,
                        help="Comma-separated position IDs matching --ports "
                             "(e.g. 5,6).  Overrides --round mapping.")
    parser.add_argument("--expected-nodes", type=int, default=None,
                        help="Expected number of ESP32 nodes.  Warns if mismatch.")
    parser.add_argument("--data-root", default=None,
                        help="Root directory for session data "
                             "(default: data).  Use to save on another drive, "
                             "e.g. --data-root D:\\csi_data")
    parser.add_argument("--dataset-label", "--label", dest="dataset_label",
                        default=None,
                        help="Human-readable label used in dataset-ready "
                             "session names (sanitized)")
    parser.add_argument("--operator", default="",
                        help="Optional operator/user identifier for manifest")
    parser.add_argument("--notes", default="",
                        help="Free-text notes stored in the session manifest")
    args = parser.parse_args()

    runtime = resolve_runtime_inputs(args)
    scenario = runtime["scenario"]
    duration_s = runtime["duration_s"]
    data_root = runtime["data_root"]
    dataset_label = runtime["dataset_label"]
    capture_mode = runtime["capture_mode"]

    sessions_root = data_root / "sessions"
    try:
        sessions_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"ERROR: Cannot create data root '{sessions_root}': {e}")
        sys.exit(1)

    # ── Resolve ports ─────────────────────────────────────────────────
    if args.ports:
        ports = [p.strip() for p in args.ports.split(",")]
    else:
        print("Scanning for ESP32-S3 serial ports...")
        ports = detect_esp32_ports(expected=args.expected_nodes)

    if not ports:
        print("ERROR: No serial ports detected.  "
              "Connect ESP32-S3 boards or use --ports.")
        sys.exit(1)

    print(f"Detected ports: {ports}")

    # ── Resolve positions ─────────────────────────────────────────────
    manual_pos = None
    if args.positions:
        manual_pos = [int(p.strip()) for p in args.positions.split(",")]

    port_node_map = build_port_node_map(ports, args.round, manual_pos)

    # ── Create session directory ──────────────────────────────────────
    sessions_root = Path(args.data_root) / "sessions"
    session_name = build_session_name(scenario, dataset_label, args.round)
    session_dir = ensure_unique_session_dir(sessions_root, session_name)
    raw_dir = session_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # ── Print summary ─────────────────────────────────────────────────
    print()
    print("=" * 64)
    print("  Wi-Fi Vision 3D - Session Recorder")
    print("=" * 64)
    print(f"  Session:   {session_name}")
    print(f"  Scenario:  {scenario}")
    print(f"  Label:     {dataset_label}")
    print(f"  Duration:  {duration_s}s ({duration_s/60:.1f} min)")
    print(f"  Mode:      {capture_mode}")
    if args.round:
        print(f"  Round:     {args.round} (positions "
              f"{ROUND_POSITIONS.get(args.round, '?')})")
    print(f"  Nodes:     {len(port_node_map)}")
    for port, nid, pid in port_node_map:
        print(f"    Node {nid:>2d}  pos {pid:>2d}  ->  {port}")
    print(f"  Output:    {raw_dir}")
    print("=" * 64)
    confirm = input("Proceed with this capture? [Y/n]: ").strip().lower()
    if confirm in {"n", "no"}:
        print("Capture aborted by user.")
        sys.exit(0)
    print()

    # ── Countdown ─────────────────────────────────────────────────────
    print("Starting in 3 seconds... (Ctrl+C to abort)")
    for i in range(3, 0, -1):
        print(f"  {i}...")
        time.sleep(1)
    print()

    # ── Capture ───────────────────────────────────────────────────────
    results = launch_parallel_capture(
        port_node_map, raw_dir, duration_s, scenario=scenario
    )

    # ── Write manifest ────────────────────────────────────────────────
    manifest = write_manifest(
        session_dir, scenario, duration_s,
        args.round, port_node_map, results, args.notes,
        dataset_label=dataset_label,
        capture_mode=capture_mode,
        data_root_resolved=str(data_root.resolve()),
        operator=args.operator,
    )

    manifest_path = session_dir / "session_manifest.json"
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not manifest_path.exists():
        print("ERROR: manifest file was not created.")
    if not csv_files:
        print("WARNING: No CSV files were generated in raw/.")

    # ── Final summary ─────────────────────────────────────────────────
    ok = sum(1 for r in results if r.success)
    total_frames = sum(r.frame_count for r in results)
    print()
    print("=" * 64)
    print(f"  SESSION COMPLETE")
    print(f"  Nodes OK:     {ok}/{len(results)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Session dir:  {session_dir}")
    print(f"  Raw CSVs:     {len(csv_files)}")
    print(f"  Manifest:     {manifest_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
