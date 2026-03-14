#!/usr/bin/env python3
"""
Wi-Fi Vision 3D - CSI Data Capture Tool

Captures CSI data from 1 or more ESP32-S3 nodes simultaneously via serial USB.
Stores raw CSV data with position metadata for later analysis and digital twin
calibration.

Usage:
    python capture_csi.py --port1 COM3 --port2 COM4 --position 1 --duration 300
    python capture_csi.py --port1 COM3 --position 1 --duration 300  # single node
"""

import argparse
import csv
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

try:
    import serial
except ImportError:
    print("ERROR: pyserial is required. Install with: pip install pyserial")
    sys.exit(1)

BAUD_RATE = 921600
CSI_LINE_PREFIX = "CSI_DATA,"

CSI_HEADER = [
    "timestamp_us", "mac", "rssi", "rate", "sig_mode", "mcs", "cwb",
    "smoothing", "not_sounding", "aggregation", "stbc", "fec", "sgi",
    "channel", "secondary_channel", "rx_seq", "len", "first_word_invalid"
]


def parse_csi_line(line: str) -> dict | None:
    """Parse a CSI_DATA line from the ESP32-S3 serial output."""
    if not line.startswith(CSI_LINE_PREFIX):
        return None

    parts = line[len(CSI_LINE_PREFIX):].strip().split(",")
    if len(parts) < len(CSI_HEADER):
        return None

    result = {}
    for i, key in enumerate(CSI_HEADER):
        result[key] = parts[i]

    result["csi_raw"] = parts[len(CSI_HEADER):]
    return result


class CaptureResult:
    """Stores statistics from a single node capture thread."""

    def __init__(self, node_id: int, port: str):
        self.node_id = node_id
        self.port = port
        self.filepath: Path | None = None
        self.frame_count = 0
        self.error_count = 0
        self.duration_actual = 0.0
        self.avg_hz = 0.0
        self.success = False

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "port": self.port,
            "filepath": str(self.filepath) if self.filepath else None,
            "frame_count": self.frame_count,
            "error_count": self.error_count,
            "duration_actual": round(self.duration_actual, 2),
            "avg_hz": round(self.avg_hz, 2),
            "success": self.success,
        }


def capture_node(port: str, node_id: int, position_id: int,
                 output_dir: Path, duration: float,
                 stop_event: threading.Event,
                 scenario: str = "",
                 baud_rate: int = BAUD_RATE) -> CaptureResult:
    """
    Capture CSI data from one serial port. Returns a CaptureResult with
    statistics. This is the reusable core used by both capture_csi.py
    and record_session.py.
    """
    result = CaptureResult(node_id, port)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{scenario}_" if scenario else ""
    filename = f"{prefix}pos{position_id:02d}_node{node_id:02d}_{timestamp_str}.csv"
    filepath = output_dir / filename
    result.filepath = filepath

    start_time = time.time()

    print(f"[Node {node_id}] Opening {port} at {baud_rate} baud...")

    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
    except serial.SerialException as e:
        print(f"[Node {node_id}] ERROR: Cannot open {port}: {e}")
        return result

    print(f"[Node {node_id}] Saving to: {filepath}")

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        meta_header = [
            f"# Wi-Fi Vision 3D - CSI Capture",
            f"# Node ID: {node_id}",
            f"# Position ID: {position_id}",
            f"# Port: {port}",
            f"# Scenario: {scenario or 'unspecified'}",
            f"# Start: {datetime.now().isoformat()}",
            f"# Duration: {duration}s",
        ]
        for meta_line in meta_header:
            f.write(meta_line + "\n")

        writer.writerow(CSI_HEADER + ["csi_data"])

        while not stop_event.is_set():
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            try:
                raw = ser.readline()
                if not raw:
                    continue

                line = raw.decode("utf-8", errors="replace").strip()

                if not line.startswith(CSI_LINE_PREFIX):
                    continue

                parsed = parse_csi_line(line)
                if parsed is None:
                    result.error_count += 1
                    continue

                row = [parsed[k] for k in CSI_HEADER]
                row.append(" ".join(parsed["csi_raw"]))
                writer.writerow(row)

                result.frame_count += 1

                if result.frame_count % 100 == 0:
                    remaining = duration - elapsed
                    hz = result.frame_count / elapsed if elapsed > 0 else 0
                    print(f"[Node {node_id}] {result.frame_count} frames | "
                          f"{hz:.1f} Hz | {remaining:.0f}s remaining | "
                          f"{result.error_count} errors")
                    f.flush()

            except (serial.SerialException, UnicodeDecodeError) as e:
                result.error_count += 1
                if result.error_count % 50 == 0:
                    print(f"[Node {node_id}] Serial error #{result.error_count}: {e}")

    ser.close()
    result.duration_actual = time.time() - start_time
    result.avg_hz = (result.frame_count / result.duration_actual
                     if result.duration_actual > 0 else 0)
    result.success = result.frame_count > 0

    print(f"\n[Node {node_id}] Capture complete:")
    print(f"  Frames: {result.frame_count}")
    print(f"  Duration: {result.duration_actual:.1f}s")
    print(f"  Average rate: {result.avg_hz:.1f} Hz")
    print(f"  Errors: {result.error_count}")
    print(f"  File: {filepath}")

    return result


def launch_parallel_capture(port_node_map: list[tuple[str, int, int]],
                            output_dir: Path, duration: float,
                            scenario: str = "") -> list[CaptureResult]:
    """
    Launch capture threads for multiple nodes in parallel.
    port_node_map is a list of (port, node_id, position_id) tuples.
    Returns a list of CaptureResult objects once all threads finish.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    stop_event = threading.Event()
    results: list[CaptureResult] = []
    threads: list[threading.Thread] = []

    for port, node_id, position_id in port_node_map:
        res = CaptureResult(node_id, port)
        results.append(res)

        def _run(p=port, nid=node_id, pid=position_id, idx=len(results) - 1):
            results[idx] = capture_node(
                p, nid, pid, output_dir, duration, stop_event, scenario
            )

        t = threading.Thread(target=_run, daemon=True)
        threads.append(t)

    for t in threads:
        t.start()

    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\n\nStopping capture (Ctrl+C)...")
        stop_event.set()
        for t in threads:
            t.join(timeout=3)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Wi-Fi Vision 3D - CSI Data Capture Tool"
    )
    parser.add_argument("--port1", required=True, help="Serial port for ESP32-S3 node 1 (e.g., COM3)")
    parser.add_argument("--port2", default=None, help="Serial port for ESP32-S3 node 2 (e.g., COM4)")
    parser.add_argument("--position", type=int, required=True, help="Position ID (1-8) for this capture round")
    parser.add_argument("--duration", type=float, default=300, help="Capture duration in seconds (default: 300)")
    parser.add_argument("--output", default="data", help="Output directory (default: data)")
    args = parser.parse_args()

    node1_id = (args.position - 1) * 2 + 1
    node2_id = (args.position - 1) * 2 + 2

    print("=" * 60)
    print("  Wi-Fi Vision 3D - CSI Capture")
    print("=" * 60)
    print(f"  Position: {args.position}")
    print(f"  Node 1 (ID {node1_id}): {args.port1}")
    if args.port2:
        print(f"  Node 2 (ID {node2_id}): {args.port2}")
    print(f"  Duration: {args.duration}s")
    print(f"  Output: {args.output}")
    print("=" * 60)
    print()

    port_map = [(args.port1, node1_id, args.position)]
    if args.port2:
        port_map.append((args.port2, node2_id, args.position))

    print("Starting capture... Press Ctrl+C to stop early.\n")
    results = launch_parallel_capture(
        port_map, Path(args.output), args.duration
    )

    ok = sum(1 for r in results if r.success)
    print(f"\nAll captures complete. {ok}/{len(results)} nodes successful.")


if __name__ == "__main__":
    main()
