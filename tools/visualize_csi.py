#!/usr/bin/env python3
"""
Wi-Fi Vision 3D - CSI Real-Time Visualizer

Connects to an ESP32-S3 via serial and displays live CSI data:
- Subcarrier amplitude spectrum (bar chart)
- Phase across subcarriers (unwrapped)
- Amplitude heatmap over time (spectrogram)

Usage:
    python visualize_csi.py --port COM3
    python visualize_csi.py --file data/pos01_node01_20260313.csv
"""

import argparse
import sys
import time
import numpy as np
from collections import deque
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
except ImportError:
    print("ERROR: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)

try:
    import serial
except ImportError:
    serial = None

CSI_LINE_PREFIX = "CSI_DATA,"
HISTORY_LENGTH = 200
MAX_SUBCARRIERS = 128


def parse_csi_complex(raw_values: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Convert raw int8 pairs (imag, real) to amplitude and phase arrays."""
    vals = [int(v) for v in raw_values if v.strip()]
    n_pairs = min(len(vals) // 2, MAX_SUBCARRIERS)

    amplitudes = np.zeros(n_pairs)
    phases = np.zeros(n_pairs)

    for i in range(n_pairs):
        imag = vals[2 * i]
        real = vals[2 * i + 1]
        amplitudes[i] = np.sqrt(real ** 2 + imag ** 2)
        phases[i] = np.arctan2(imag, real)

    return amplitudes, phases


def parse_line(line: str) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Parse CSI_DATA line, return (amplitudes, phases, rssi) or None."""
    if not line.startswith(CSI_LINE_PREFIX):
        return None

    parts = line[len(CSI_LINE_PREFIX):].strip().split(",")
    if len(parts) < 19:
        return None

    try:
        rssi = int(parts[2])
        csi_raw = parts[18:]
        if len(csi_raw) < 4:
            return None

        amplitudes, phases = parse_csi_complex(csi_raw)
        return amplitudes, phases, rssi
    except (ValueError, IndexError):
        return None


class CSIVisualizer:
    def __init__(self, source, is_file=False, baud=921600):
        self.source = source
        self.is_file = is_file
        self.baud = baud
        self.ser = None
        self.debug_lines_shown = 0
        self.max_debug_lines = 10
        self.total_serial_lines = 0
        self.unparsed_lines = 0

        self.amp_history = deque(maxlen=HISTORY_LENGTH)
        self.phase_history = deque(maxlen=HISTORY_LENGTH)
        self.rssi_history = deque(maxlen=HISTORY_LENGTH)
        self.frame_count = 0
        self.n_subcarriers = 0

        self.fig, self.axes = plt.subplots(3, 1, figsize=(14, 9))
        self.fig.suptitle("Wi-Fi Vision 3D - CSI Monitor", fontsize=14, fontweight="bold")
        self.fig.set_facecolor("#1a1a2e")

        for ax in self.axes:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="#a0a0a0")
            ax.xaxis.label.set_color("#d0d0d0")
            ax.yaxis.label.set_color("#d0d0d0")
            ax.title.set_color("#e0e0e0")
            for spine in ax.spines.values():
                spine.set_color("#333366")

        self.ax_amp = self.axes[0]
        self.ax_phase = self.axes[1]
        self.ax_spectrogram = self.axes[2]

        self.ax_amp.set_title("Subcarrier Amplitude (current frame)")
        self.ax_amp.set_xlabel("Subcarrier Index")
        self.ax_amp.set_ylabel("Amplitude")

        self.ax_phase.set_title("Subcarrier Phase (unwrapped)")
        self.ax_phase.set_xlabel("Subcarrier Index")
        self.ax_phase.set_ylabel("Phase (rad)")

        self.ax_spectrogram.set_title("Amplitude Spectrogram (time x subcarrier)")
        self.ax_spectrogram.set_xlabel("Subcarrier Index")
        self.ax_spectrogram.set_ylabel("Time (frames)")

        self.amp_bars = None
        self.phase_line = None
        self.spectrogram_img = None

        self.status_text = self.fig.text(
            0.02, 0.01, "", fontsize=9, color="#00ff88",
            fontfamily="monospace"
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    def open_source(self):
        if self.is_file:
            self.file_handle = open(self.source, "r", encoding="utf-8")
            return True
        else:
            if serial is None:
                print("ERROR: pyserial is required for live mode. pip install pyserial")
                return False
            try:
                self.ser = serial.Serial(self.source, self.baud, timeout=0.1)
                print(f"Conectado a {self.source} @ {self.baud} baud")
                return True
            except serial.SerialException as e:
                print(f"ERROR: Cannot open {self.source}: {e}")
                return False

    def read_next_line(self) -> str | None:
        if self.is_file:
            while True:
                line = self.file_handle.readline()
                if not line:
                    return None
                line = line.strip()
                if line.startswith(CSI_LINE_PREFIX):
                    return line
        else:
            try:
                raw = self.ser.readline()
                if raw:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if line:
                        self.total_serial_lines += 1
                        if not line.startswith(CSI_LINE_PREFIX):
                            self.unparsed_lines += 1
                            if self.debug_lines_shown < self.max_debug_lines:
                                self.debug_lines_shown += 1
                                print(f"  [DEBUG] {line[:150]}", file=sys.stderr)
                    return line
            except (serial.SerialException, UnicodeDecodeError):
                pass
        return None

    def update(self, frame_num):
        lines_read = 0
        max_lines = 5 if not self.is_file else 20

        while lines_read < max_lines:
            line = self.read_next_line()
            if line is None:
                break

            result = parse_line(line)
            if result is None:
                continue

            amplitudes, phases, rssi = result
            self.n_subcarriers = len(amplitudes)

            self.amp_history.append(amplitudes)
            self.phase_history.append(np.unwrap(phases))
            self.rssi_history.append(rssi)
            self.frame_count += 1
            lines_read += 1

        if self.frame_count == 0:
            return []

        artists = []

        current_amp = self.amp_history[-1]
        current_phase = self.phase_history[-1]
        n = len(current_amp)
        x = np.arange(n)

        self.ax_amp.cla()
        self.ax_amp.set_facecolor("#16213e")
        self.ax_amp.set_title("Subcarrier Amplitude (current frame)", color="#e0e0e0")
        self.ax_amp.bar(x, current_amp, color="#00d4ff", alpha=0.8, width=1.0)
        self.ax_amp.set_xlim(-1, n)
        amp_max = max(np.max(current_amp) * 1.2, 1)
        self.ax_amp.set_ylim(0, amp_max)
        self.ax_amp.set_xlabel("Subcarrier Index", color="#d0d0d0")
        self.ax_amp.set_ylabel("Amplitude", color="#d0d0d0")

        self.ax_phase.cla()
        self.ax_phase.set_facecolor("#16213e")
        self.ax_phase.set_title("Subcarrier Phase (unwrapped)", color="#e0e0e0")
        self.ax_phase.plot(x, current_phase, color="#ff6b6b", linewidth=1.2)
        self.ax_phase.set_xlim(-1, n)
        self.ax_phase.set_xlabel("Subcarrier Index", color="#d0d0d0")
        self.ax_phase.set_ylabel("Phase (rad)", color="#d0d0d0")

        if len(self.amp_history) > 1:
            max_sc = max(len(a) for a in self.amp_history)
            spectrogram = np.zeros((len(self.amp_history), max_sc))
            for i, amp in enumerate(self.amp_history):
                spectrogram[i, :len(amp)] = amp

            self.ax_spectrogram.cla()
            self.ax_spectrogram.set_facecolor("#16213e")
            self.ax_spectrogram.set_title("Amplitude Spectrogram (time x subcarrier)", color="#e0e0e0")
            self.ax_spectrogram.imshow(
                spectrogram, aspect="auto", origin="lower",
                cmap="inferno", interpolation="nearest"
            )
            self.ax_spectrogram.set_xlabel("Subcarrier Index", color="#d0d0d0")
            self.ax_spectrogram.set_ylabel("Time (frames)", color="#d0d0d0")

        avg_rssi = np.mean(list(self.rssi_history)[-50:]) if self.rssi_history else 0
        elapsed = time.time() - self.start_time if hasattr(self, "start_time") else 0.001
        hz = self.frame_count / max(elapsed, 0.001)

        status = (
            f"Frames: {self.frame_count}  |  "
            f"Subcarriers: {self.n_subcarriers}  |  "
            f"RSSI: {avg_rssi:.0f} dBm  |  "
            f"Rate: {hz:.1f} Hz"
        )

        if self.frame_count == 0 and elapsed > 5 and not self.is_file:
            if self.total_serial_lines == 0:
                status += "  |  SIN DATOS - verificar baud rate"
                self.status_text.set_color("#ff4444")
            else:
                status += f"  |  {self.total_serial_lines} lineas recibidas, 0 CSI - verificar firmware/WiFi"
                self.status_text.set_color("#ffaa00")
        elif self.frame_count > 0:
            self.status_text.set_color("#00ff88")

        self.status_text.set_text(status)

        return artists

    def run(self):
        if not self.open_source():
            return

        self.start_time = time.time()

        interval = 50 if not self.is_file else 20
        self.anim = animation.FuncAnimation(
            self.fig, self.update, interval=interval, blit=False, cache_frame_data=False
        )

        plt.show()

        if self.ser:
            self.ser.close()
        if hasattr(self, "file_handle"):
            self.file_handle.close()


def main():
    parser = argparse.ArgumentParser(
        description="Wi-Fi Vision 3D - CSI Real-Time Visualizer"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--port", help="Serial port for live visualization (e.g., COM3)")
    group.add_argument("--file", help="CSV file for offline visualization")
    parser.add_argument("--baud", type=int, default=921600,
                        help="Serial baud rate (default: 921600)")
    args = parser.parse_args()

    if args.port:
        viz = CSIVisualizer(args.port, is_file=False, baud=args.baud)
    else:
        if not Path(args.file).exists():
            print(f"ERROR: File not found: {args.file}")
            sys.exit(1)
        viz = CSIVisualizer(args.file, is_file=True)

    viz.run()


if __name__ == "__main__":
    main()
