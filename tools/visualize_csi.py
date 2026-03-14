#!/usr/bin/env python3
"""
Wi-Fi Vision 3D - CSI Real-Time Visualizer

Connects to an ESP32-S3 via serial or reads a CSV capture file and displays:
- Subcarrier amplitude spectrum (bar chart)
- Phase across subcarriers (unwrapped)
- Amplitude heatmap over time (spectrogram)
- Activity indicator (turbulence over time)
- Wi-Fi signal strength (RSSI over time)

Supports two file formats:
- Raw serial: lines prefixed with CSI_DATA,...
- CSV capture: files produced by capture_csi.py / record_session.py

Usage:
    python visualize_csi.py --port COM3
    python visualize_csi.py --file data/sessions/.../raw/baseline_pos01_node01.csv
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


CSV_HEADER_FIELDS = 18

def parse_csv_line(line: str) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Parse a CSV capture line (from capture_csi.py / record_session.py).

    CSV format: timestamp_us,mac,rssi,...,first_word_invalid,<csi_data>
    where <csi_data> is space-separated int8 I/Q pairs in a single field.
    """
    parts = line.strip().split(",")
    if len(parts) < CSV_HEADER_FIELDS + 1:
        return None

    try:
        rssi = int(parts[2])
        csi_field = ",".join(parts[CSV_HEADER_FIELDS:]).strip().strip('"')
        iq_tokens = csi_field.split()
        if len(iq_tokens) < 4:
            return None

        amplitudes, phases = parse_csi_complex(iq_tokens)
        return amplitudes, phases, rssi
    except (ValueError, IndexError):
        return None


def detect_file_format(filepath: str) -> str:
    """Peek at a file to determine if it is raw serial or CSV capture format.

    Returns 'csv' or 'raw'.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        for _ in range(30):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("timestamp_us") or line.startswith('"timestamp_us'):
                return "csv"
            if line.startswith(CSI_LINE_PREFIX):
                return "raw"
    return "raw"


class CSIVisualizer:
    def __init__(self, source, is_file=False, baud=921600, save_fig=None,
                 save_after_frames=80):
        self.source = source
        self.is_file = is_file
        self.baud = baud
        self.save_fig = save_fig
        self.save_after_frames = save_after_frames
        self.ser = None
        self.csv_mode = False
        self.csv_header_skipped = False
        self.debug_lines_shown = 0
        self.max_debug_lines = 10
        self.total_serial_lines = 0
        self.unparsed_lines = 0

        self.amp_history = deque(maxlen=HISTORY_LENGTH)
        self.phase_history = deque(maxlen=HISTORY_LENGTH)
        self.rssi_history = deque(maxlen=HISTORY_LENGTH)
        self.activity_history = deque(maxlen=HISTORY_LENGTH)
        self.frame_count = 0
        self.n_subcarriers = 0

        self.fig, self.axes = plt.subplots(5, 1, figsize=(14, 14))
        self.fig.suptitle("Wi-Fi Vision 3D - CSI Monitor", fontsize=14,
                          fontweight="bold", color="#e0e0e0")
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
        self.ax_activity = self.axes[3]
        self.ax_rssi = self.axes[4]

        self.ax_amp.set_title("Subcarrier Amplitude (current frame)")
        self.ax_amp.set_xlabel("Subcarrier Index")
        self.ax_amp.set_ylabel("Amplitude")

        self.ax_phase.set_title("Subcarrier Phase (unwrapped)")
        self.ax_phase.set_xlabel("Subcarrier Index")
        self.ax_phase.set_ylabel("Phase (rad)")

        self.ax_spectrogram.set_title("Amplitude Spectrogram (time x subcarrier)")
        self.ax_spectrogram.set_xlabel("Subcarrier Index")
        self.ax_spectrogram.set_ylabel("Time (frames)")

        self.ax_activity.set_title("Indicador de Actividad (turbulencia)")
        self.ax_activity.set_xlabel("Frame")
        self.ax_activity.set_ylabel("CV (std/mean)")

        self.ax_rssi.set_title("Senal Wi-Fi (RSSI)")
        self.ax_rssi.set_xlabel("Frame")
        self.ax_rssi.set_ylabel("RSSI (dBm)")

        self.status_text = self.fig.text(
            0.02, 0.005, "", fontsize=9, color="#00ff88",
            fontfamily="monospace"
        )

        plt.tight_layout(rect=[0, 0.025, 1, 0.965])

    def open_source(self):
        if self.is_file:
            fmt = detect_file_format(self.source)
            self.csv_mode = (fmt == "csv")
            if self.csv_mode:
                print(f"Formato detectado: CSV captura ({self.source})")
            else:
                print(f"Formato detectado: raw serial ({self.source})")
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
                if not line or line.startswith("#"):
                    continue
                if self.csv_mode:
                    if line.startswith("timestamp_us") or line.startswith('"timestamp_us'):
                        continue
                    return line
                else:
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

    def _parse(self, line: str):
        """Route to the correct parser based on file format."""
        if self.csv_mode:
            return parse_csv_line(line)
        return parse_line(line)

    def update(self, frame_num):
        lines_read = 0
        max_lines = 5 if not self.is_file else 20

        while lines_read < max_lines:
            line = self.read_next_line()
            if line is None:
                break

            result = self._parse(line)
            if result is None:
                continue

            amplitudes, phases, rssi = result
            self.n_subcarriers = len(amplitudes)

            mean_amp = np.mean(amplitudes)
            cv = np.std(amplitudes) / mean_amp if mean_amp > 0 else 0.0

            self.amp_history.append(amplitudes)
            self.phase_history.append(np.unwrap(phases))
            self.rssi_history.append(rssi)
            self.activity_history.append(cv)
            self.frame_count += 1
            lines_read += 1

        if self.frame_count == 0:
            return []

        artists = []

        current_amp = self.amp_history[-1]
        current_phase = self.phase_history[-1]
        n = len(current_amp)
        x = np.arange(n)

        # Panel 1: Amplitude spectrum
        self.ax_amp.cla()
        self.ax_amp.set_facecolor("#16213e")
        self.ax_amp.set_title("Subcarrier Amplitude (current frame)", color="#e0e0e0")
        self.ax_amp.bar(x, current_amp, color="#00d4ff", alpha=0.8, width=1.0)
        self.ax_amp.set_xlim(-1, n)
        amp_max = max(np.max(current_amp) * 1.2, 1)
        self.ax_amp.set_ylim(0, amp_max)
        self.ax_amp.set_xlabel("Subcarrier Index", color="#d0d0d0")
        self.ax_amp.set_ylabel("Amplitude", color="#d0d0d0")

        # Panel 2: Phase
        self.ax_phase.cla()
        self.ax_phase.set_facecolor("#16213e")
        self.ax_phase.set_title("Subcarrier Phase (unwrapped)", color="#e0e0e0")
        self.ax_phase.plot(x, current_phase, color="#ff6b6b", linewidth=1.2)
        self.ax_phase.set_xlim(-1, n)
        self.ax_phase.set_xlabel("Subcarrier Index", color="#d0d0d0")
        self.ax_phase.set_ylabel("Phase (rad)", color="#d0d0d0")

        # Panel 3: Spectrogram
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

        # Panel 4: Activity indicator (CV turbulence)
        if len(self.activity_history) > 1:
            act = list(self.activity_history)
            t = np.arange(len(act))
            self.ax_activity.cla()
            self.ax_activity.set_facecolor("#16213e")
            self.ax_activity.set_title("Indicador de Actividad (turbulencia)", color="#e0e0e0")
            self.ax_activity.fill_between(t, act, alpha=0.35, color="#4ade80")
            self.ax_activity.plot(t, act, color="#4ade80", linewidth=1.2)
            self.ax_activity.set_xlim(0, max(len(act) - 1, 1))
            self.ax_activity.set_ylim(0, max(max(act) * 1.2, 0.1))
            self.ax_activity.set_xlabel("Frame", color="#d0d0d0")
            self.ax_activity.set_ylabel("CV (std/mean)", color="#d0d0d0")
            self.ax_activity.axhline(
                y=np.mean(act), color="#fbbf24", linewidth=0.8,
                linestyle="--", alpha=0.7
            )

        # Panel 5: RSSI over time
        if len(self.rssi_history) > 1:
            rssi_vals = list(self.rssi_history)
            t = np.arange(len(rssi_vals))
            self.ax_rssi.cla()
            self.ax_rssi.set_facecolor("#16213e")
            self.ax_rssi.set_title("Senal Wi-Fi (RSSI)", color="#e0e0e0")
            self.ax_rssi.plot(t, rssi_vals, color="#38bdf8", linewidth=1.2)
            self.ax_rssi.fill_between(t, rssi_vals, alpha=0.2, color="#38bdf8")
            self.ax_rssi.set_xlim(0, max(len(rssi_vals) - 1, 1))
            r_min = min(rssi_vals)
            r_max = max(rssi_vals)
            margin = max((r_max - r_min) * 0.2, 2)
            self.ax_rssi.set_ylim(r_min - margin, r_max + margin)
            self.ax_rssi.set_xlabel("Frame", color="#d0d0d0")
            self.ax_rssi.set_ylabel("RSSI (dBm)", color="#d0d0d0")
            self.ax_rssi.axhline(
                y=np.mean(rssi_vals), color="#fbbf24", linewidth=0.8,
                linestyle="--", alpha=0.7
            )

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

        if self.save_fig:
            self._run_headless()
        else:
            self._run_interactive()

        if self.ser:
            self.ser.close()
        if hasattr(self, "file_handle"):
            self.file_handle.close()

    def _run_headless(self):
        """Render frames in a loop without GUI, then save the figure."""
        frame = 0
        while self.frame_count < self.save_after_frames:
            self.update(frame)
            frame += 1
            if self.is_file and self.frame_count == 0 and frame > 50:
                print("WARN: no frames parsed after 50 iterations, aborting.")
                return
        self.fig.savefig(self.save_fig, dpi=150,
                         facecolor=self.fig.get_facecolor(),
                         bbox_inches="tight", pad_inches=0.3)
        print(f"Screenshot guardado: {self.save_fig} ({self.frame_count} frames)")
        plt.close(self.fig)

    def _run_interactive(self):
        """Normal animation loop with GUI window."""
        interval = 50 if not self.is_file else 20
        self.anim = animation.FuncAnimation(
            self.fig, self.update, interval=interval, blit=False, cache_frame_data=False
        )
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Wi-Fi Vision 3D - CSI Real-Time Visualizer"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--port", help="Serial port for live visualization (e.g., COM3)")
    group.add_argument("--file", help="CSV file for offline visualization")
    parser.add_argument("--baud", type=int, default=921600,
                        help="Serial baud rate (default: 921600)")
    parser.add_argument("--save-fig", metavar="PATH",
                        help="Save screenshot to PNG after ~80 frames and exit")
    parser.add_argument("--save-after", type=int, default=80,
                        help="Number of frames to render before saving (default: 80)")
    args = parser.parse_args()

    if args.port:
        viz = CSIVisualizer(args.port, is_file=False, baud=args.baud,
                            save_fig=args.save_fig,
                            save_after_frames=args.save_after)
    else:
        if not Path(args.file).exists():
            print(f"ERROR: File not found: {args.file}")
            sys.exit(1)
        viz = CSIVisualizer(args.file, is_file=True,
                            save_fig=args.save_fig,
                            save_after_frames=args.save_after)

    viz.run()


if __name__ == "__main__":
    main()
