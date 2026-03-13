#!/usr/bin/env python3
"""
Wi-Fi Vision 3D - CSI Data Capture Tool

Captures CSI data from 1 or 2 ESP32-S3 nodes simultaneously via serial USB.
Stores raw CSV data with position metadata for later analysis and digital twin
calibration.

Usage:
    python capture_csi.py --port1 COM3 --port2 COM4 --position 1 --duration 300
    python capture_csi.py --port1 COM3 --position 1 --duration 300  # single node
"""

import argparse
import csv
import os
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


def capture_serial(port: str, node_id: int, position_id: int,
                   output_dir: Path, duration: float, stop_event: threading.Event):
    """Capture CSI data from one serial port."""

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pos{position_id:02d}_node{node_id:02d}_{timestamp_str}.csv"
    filepath = output_dir / filename

    frame_count = 0
    error_count = 0
    start_time = time.time()

    print(f"[Node {node_id}] Opening {port} at {BAUD_RATE} baud...")

    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=1)
    except serial.SerialException as e:
        print(f"[Node {node_id}] ERROR: Cannot open {port}: {e}")
        return

    print(f"[Node {node_id}] Saving to: {filepath}")

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        meta_header = [
            f"# Wi-Fi Vision 3D - CSI Capture",
            f"# Node ID: {node_id}",
            f"# Position ID: {position_id}",
            f"# Port: {port}",
            f"# Start: {datetime.now().isoformat()}",
            f"# Duration: {duration}s",
        ]
        for line in meta_header:
            f.write(line + "\n")

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
                    if line and not line.startswith("I ") and not line.startswith("W "):
                        pass  # skip ESP log lines
                    continue

                parsed = parse_csi_line(line)
                if parsed is None:
                    error_count += 1
                    continue

                row = [parsed[k] for k in CSI_HEADER]
                row.append(" ".join(parsed["csi_raw"]))
                writer.writerow(row)

                frame_count += 1

                if frame_count % 100 == 0:
                    remaining = duration - elapsed
                    hz = frame_count / elapsed if elapsed > 0 else 0
                    print(f"[Node {node_id}] {frame_count} frames | "
                          f"{hz:.1f} Hz | {remaining:.0f}s remaining | "
                          f"{error_count} errors")
                    f.flush()

            except (serial.SerialException, UnicodeDecodeError) as e:
                error_count += 1
                if error_count % 50 == 0:
                    print(f"[Node {node_id}] Serial error #{error_count}: {e}")

    ser.close()
    total_time = time.time() - start_time
    avg_hz = frame_count / total_time if total_time > 0 else 0

    print(f"\n[Node {node_id}] Capture complete:")
    print(f"  Frames: {frame_count}")
    print(f"  Duration: {total_time:.1f}s")
    print(f"  Average rate: {avg_hz:.1f} Hz")
    print(f"  Errors: {error_count}")
    print(f"  File: {filepath}")


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

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    stop_event = threading.Event()

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
    print(f"  Output: {output_dir}")
    print("=" * 60)
    print()

    threads = []

    t1 = threading.Thread(
        target=capture_serial,
        args=(args.port1, node1_id, args.position, output_dir, args.duration, stop_event),
        daemon=True
    )
    threads.append(t1)

    if args.port2:
        t2 = threading.Thread(
            target=capture_serial,
            args=(args.port2, node2_id, args.position, output_dir, args.duration, stop_event),
            daemon=True
        )
        threads.append(t2)

    print("Starting capture... Press Ctrl+C to stop early.\n")

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

    print("\nAll captures complete.")


if __name__ == "__main__":
    main()
