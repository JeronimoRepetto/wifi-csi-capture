#!/usr/bin/env python3
"""
Wi-Fi Vision 3D - Full Test Suite

Tests all Python modules without requiring real ESP32-S3 hardware.
Uses synthetic CSI data generated in-memory or in tmp_path fixtures.

Run with:
    pytest tests/test_all.py -v
    pytest tests/test_all.py -v -k spatial   # only spatial filter tests
"""

import csv
import json
import sys
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pytest

# Make tools/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from capture_csi import parse_csi_line, CaptureResult, CSI_HEADER  # noqa: E402
from record_session import (  # noqa: E402
    build_port_node_map,
    write_manifest,
    ROUND_POSITIONS,
    DEFAULT_DATA_ROOT,
    sanitize_dataset_label,
    parse_duration_seconds,
    build_session_name,
    ensure_unique_session_dir,
    resolve_runtime_inputs,
    normalize_mac,
    collect_mac_summary,
    evaluate_mac_summary,
)
from analyze_csi import load_csi_file, export_baseline  # noqa: E402
from spatial_filter import (  # noqa: E402
    _line_segment_inside_box,
    compute_zone_weights,
    SpatialZoneFilter,
    read_t0_host_us,
    align_nodes_by_timestamp,
    ALIGN_WINDOW_US,
)


@pytest.fixture
def tmp_path():
    """Use a workspace-local tmp dir so the sandbox permits file writes."""
    base = Path(__file__).parent / "tmp"
    base.mkdir(exist_ok=True)
    import uuid
    d = base / uuid.uuid4().hex[:8]
    d.mkdir(exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers: synthetic CSI data generators
# ═══════════════════════════════════════════════════════════════════════════

N_SUBCARRIERS = 114


def _make_csi_line(timestamp_us=1000000, mac="aa:bb:cc:dd:ee:ff",
                   rssi=-35, n_subcarriers=N_SUBCARRIERS,
                   amplitude_base=30.0, noise_std=2.0, seed=None):
    """Build a realistic CSI_DATA line as a string."""
    rng = np.random.default_rng(seed)
    parts = [
        str(timestamp_us), mac, str(rssi),
        "11", "1", "7", "1",       # rate, sig_mode, mcs, cwb
        "0", "1", "0",             # smoothing, not_sounding, aggregation
        "0", "0", "0",             # stbc, fec, sgi
        "6", "1",                  # channel, secondary_channel
        "1234", str(n_subcarriers * 2), "0",  # rx_seq, len, first_word_invalid
    ]
    for _ in range(n_subcarriers):
        amp = amplitude_base + rng.normal(0, noise_std)
        phase = rng.uniform(-np.pi, np.pi)
        real = int(np.clip(amp * np.cos(phase), -128, 127))
        imag = int(np.clip(amp * np.sin(phase), -128, 127))
        parts.append(str(imag))
        parts.append(str(real))
    return "CSI_DATA," + ",".join(parts)


def _write_synthetic_csv(filepath: Path, n_frames=100,
                         node_id=1, position_id=1,
                         amplitude_base=30.0, noise_std=2.0,
                         scenario="baseline_empty",
                         t0_host_us=1_741_000_000_000_000):
    """Write a synthetic CSV that load_csi_file() can read."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        f.write(f"# Wi-Fi Vision 3D - CSI Capture\n")
        f.write(f"# Node ID: {node_id}\n")
        f.write(f"# Position ID: {position_id}\n")
        f.write(f"# Port: VIRTUAL\n")
        f.write(f"# Scenario: {scenario}\n")
        f.write(f"# Start: 2026-03-14T15:00:00\n")
        f.write(f"# Duration: 10s\n")
        f.write(f"# t0_host_us: {t0_host_us}\n")

        writer = csv.writer(f)
        writer.writerow(CSI_HEADER + ["csi_data"])

        rng = np.random.default_rng(42 + node_id)
        for i in range(n_frames):
            ts_us = 1000000 + i * 20000  # ~50 Hz
            csi_vals = []
            for _ in range(N_SUBCARRIERS):
                amp = amplitude_base + rng.normal(0, noise_std)
                phase = rng.uniform(-np.pi, np.pi)
                real = int(np.clip(amp * np.cos(phase), -128, 127))
                imag = int(np.clip(amp * np.sin(phase), -128, 127))
                csi_vals.extend([str(imag), str(real)])

            row = [
                str(ts_us), "aa:bb:cc:dd:ee:ff", "-35",
                "11", "1", "7", "1",
                "0", "1", "0",
                "0", "0", "0",
                "6", "1",
                str(1000 + i), str(N_SUBCARRIERS * 2), "0",
            ]
            row.append(" ".join(csi_vals))
            writer.writerow(row)


def _make_positions_and_router():
    """Return the standard 8-node positions dict and router dict."""
    positions = {
        1: {"label": "Techo FL", "x": 0.0, "y": 0.0, "z": 2.5},
        2: {"label": "Techo FR", "x": 3.0, "y": 0.0, "z": 2.5},
        3: {"label": "Techo BL", "x": 0.0, "y": 4.0, "z": 2.5},
        4: {"label": "Techo BR", "x": 3.0, "y": 4.0, "z": 2.5},
        5: {"label": "Suelo FL", "x": 0.0, "y": 0.0, "z": 0.15},
        6: {"label": "Suelo FR", "x": 3.0, "y": 0.0, "z": 0.15},
        7: {"label": "Suelo BL", "x": 0.0, "y": 4.0, "z": 0.15},
        8: {"label": "Suelo BR", "x": 3.0, "y": 4.0, "z": 0.15},
    }
    router = {"label": "Router", "x": 1.5, "y": 2.0, "z": 1.5}
    return positions, router


def _make_baseline_stats(n_nodes=8, amplitude_base=30.0, noise_std=2.0):
    """Build baseline_stats dict as SpatialZoneFilter expects."""
    rng = np.random.default_rng(99)
    stats = {}
    for nid in range(1, n_nodes + 1):
        mean = np.full(N_SUBCARRIERS, amplitude_base) + rng.normal(0, 1, N_SUBCARRIERS)
        std = np.full(N_SUBCARRIERS, noise_std) + rng.uniform(0, 0.5, N_SUBCARRIERS)
        stats[str(nid)] = {
            "amplitude_mean": mean.tolist(),
            "amplitude_std": std.tolist(),
        }
    return stats


# ═══════════════════════════════════════════════════════════════════════════
#  Tests: capture_csi.py
# ═══════════════════════════════════════════════════════════════════════════

class TestParseCsiLine:
    def test_valid_line(self):
        line = _make_csi_line(seed=0)
        result = parse_csi_line(line)
        assert result is not None
        assert "timestamp_us" in result
        assert "rssi" in result
        assert "csi_raw" in result
        assert len(result["csi_raw"]) == N_SUBCARRIERS * 2
        for key in CSI_HEADER:
            assert key in result

    def test_non_csi_line_returns_none(self):
        assert parse_csi_line("I CSI_CAPTURE: Connected") is None
        assert parse_csi_line("W (wifi): something") is None
        assert parse_csi_line("") is None

    def test_truncated_line_returns_none(self):
        assert parse_csi_line("CSI_DATA,123,ab:cd") is None

    def test_different_rssi_values(self):
        for rssi in [-20, -45, -70]:
            line = _make_csi_line(rssi=rssi, seed=1)
            result = parse_csi_line(line)
            assert result is not None
            assert result["rssi"] == str(rssi)


class TestCaptureResult:
    def test_to_dict_default(self):
        r = CaptureResult(node_id=3, port="COM7")
        d = r.to_dict()
        assert d["node_id"] == 3
        assert d["port"] == "COM7"
        assert d["frame_count"] == 0
        assert d["success"] is False
        assert d["filepath"] is None

    def test_to_dict_with_data(self):
        r = CaptureResult(node_id=1, port="COM3")
        r.frame_count = 5000
        r.error_count = 3
        r.duration_actual = 100.5
        r.avg_hz = 49.75
        r.success = True
        r.filepath = Path("data/test.csv")
        d = r.to_dict()
        assert d["frame_count"] == 5000
        assert d["success"] is True
        assert d["avg_hz"] == 49.75
        assert "test.csv" in d["filepath"]


# ═══════════════════════════════════════════════════════════════════════════
#  Tests: record_session.py
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildPortNodeMap:
    def test_round_mapping(self):
        mapping = build_port_node_map(["COM3", "COM4"], round_id=1, manual_positions=None)
        assert len(mapping) == 2
        assert mapping[0] == ("COM3", 1, 1)
        assert mapping[1] == ("COM4", 2, 2)

    def test_round_2_mapping(self):
        mapping = build_port_node_map(["COM3", "COM4"], round_id=2, manual_positions=None)
        assert mapping[0] == ("COM3", 3, 3)
        assert mapping[1] == ("COM4", 4, 4)

    def test_round_3_mapping(self):
        mapping = build_port_node_map(["COM3", "COM4"], round_id=3, manual_positions=None)
        assert mapping[0] == ("COM3", 5, 5)
        assert mapping[1] == ("COM4", 6, 6)

    def test_round_4_mapping(self):
        mapping = build_port_node_map(["COM3", "COM4"], round_id=4, manual_positions=None)
        assert mapping[0] == ("COM3", 7, 7)
        assert mapping[1] == ("COM4", 8, 8)

    def test_manual_positions_override_round(self):
        mapping = build_port_node_map(
            ["COM3", "COM4"], round_id=1, manual_positions=[5, 6]
        )
        assert mapping[0] == ("COM3", 5, 5)
        assert mapping[1] == ("COM4", 6, 6)

    def test_auto_sequential_no_round(self):
        mapping = build_port_node_map(
            ["COM3", "COM4", "COM5"], round_id=None, manual_positions=None
        )
        assert len(mapping) == 3
        assert mapping[0] == ("COM3", 1, 1)
        assert mapping[1] == ("COM4", 2, 2)
        assert mapping[2] == ("COM5", 3, 3)

    def test_more_ports_than_positions(self):
        mapping = build_port_node_map(
            ["COM3", "COM4", "COM5"], round_id=1, manual_positions=None
        )
        assert len(mapping) == 2

    def test_fewer_ports_than_positions(self):
        mapping = build_port_node_map(
            ["COM3"], round_id=1, manual_positions=None
        )
        assert len(mapping) == 1
        assert mapping[0] == ("COM3", 1, 1)


class TestWriteManifest:
    def test_creates_valid_json(self, tmp_path):
        session_dir = tmp_path / "test_session"
        session_dir.mkdir()

        r1 = CaptureResult(1, "COM3")
        r1.frame_count = 100
        r1.success = True
        r2 = CaptureResult(2, "COM4")
        r2.frame_count = 95
        r2.success = True

        manifest = write_manifest(
            session_dir, "baseline_empty", 300.0,
            round_id=1, port_node_map=[("COM3", 1, 1), ("COM4", 2, 2)],
            results=[r1, r2], notes="test run",
            start_utc="2026-03-14T15:00:00+00:00",
            t0_host_us=1_741_000_000_000_000,
        )

        manifest_path = session_dir / "session_manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            loaded = json.load(f)

        assert loaded["scenario"] == "baseline_empty"
        assert loaded["round"] == 1
        assert loaded["duration_requested_s"] == 300.0
        assert loaded["notes"] == "test run"
        assert len(loaded["nodes"]) == 2
        assert loaded["summary"]["total_frames"] == 195
        assert loaded["summary"]["successful_nodes"] == 2
        # New sync fields
        assert loaded["start_utc"] == "2026-03-14T15:00:00+00:00"
        assert loaded["t0_host_us"] == 1_741_000_000_000_000

    def test_session_dir_structure(self, tmp_path):
        session_dir = tmp_path / "my_session"
        raw_dir = session_dir / "raw"
        raw_dir.mkdir(parents=True)

        r = CaptureResult(1, "COM3")
        write_manifest(session_dir, "custom", 60.0, None, [], [r], "")

        assert (session_dir / "session_manifest.json").exists()
        assert raw_dir.is_dir()

    def test_manifest_ai_fields(self, tmp_path):
        session_dir = tmp_path / "ai_session"
        session_dir.mkdir()
        r = CaptureResult(1, "COM3")
        write_manifest(
            session_dir, "stairs_walk", 120.0, 1, [("COM3", 1, 1)], [r], "note",
            dataset_label="stairs_walk_user01",
            capture_mode="interactive",
            data_root_resolved="D:/csi_data",
            operator="jeron",
            t0_host_us=9999,
        )

        data = json.loads((session_dir / "session_manifest.json").read_text(encoding="utf-8"))
        assert data["dataset_label"] == "stairs_walk_user01"
        assert data["capture_mode"] == "interactive"
        assert data["data_root_resolved"] == "D:/csi_data"
        assert data["operator"] == "jeron"
        assert data["t0_host_us"] == 9999

    def test_start_utc_is_set_before_capture(self, tmp_path):
        """start_utc must be a real ISO timestamp, not empty."""
        import re
        session_dir = tmp_path / "ts_session"
        session_dir.mkdir()
        r = CaptureResult(1, "COM3")
        write_manifest(
            session_dir, "baseline_empty", 30.0, 1, [], [r], "",
            start_utc="2026-03-14T16:00:00+00:00",
        )
        data = json.loads((session_dir / "session_manifest.json").read_text(encoding="utf-8"))
        iso_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        assert re.match(iso_pattern, data["start_utc"])


class TestDataRootDefault:
    def test_default_data_root_is_data(self):
        assert DEFAULT_DATA_ROOT == Path("data")


class TestRecordSessionHelpers:
    def test_sanitize_dataset_label(self):
        assert sanitize_dataset_label("BaseLine Empty #1") == "baseline_empty_1"

    def test_sanitize_dataset_label_invalid(self):
        with pytest.raises(ValueError):
            sanitize_dataset_label("###")

    def test_parse_duration_seconds_plain_number(self):
        assert parse_duration_seconds("300") == 300.0

    def test_parse_duration_seconds_seconds_suffix(self):
        assert parse_duration_seconds("120s") == 120.0

    def test_parse_duration_seconds_minutes_suffix(self):
        assert parse_duration_seconds("5m") == 300.0
        assert parse_duration_seconds("2.5min") == 150.0

    def test_parse_duration_seconds_invalid(self):
        with pytest.raises(ValueError):
            parse_duration_seconds("abc")

    def test_build_session_name(self):
        fixed_now = lambda: __import__("datetime").datetime(2026, 3, 14, 16, 20, 30)
        name = build_session_name("baseline_empty", "stairs_zone", 2, now_fn=fixed_now)
        assert name == "20260314_162030_baseline_empty_stairs_zone_round2"

    def test_ensure_unique_session_dir(self, tmp_path):
        sessions_root = tmp_path / "sessions"
        sessions_root.mkdir()
        (sessions_root / "20260314_162030_baseline_empty_stairs").mkdir()
        unique = ensure_unique_session_dir(
            sessions_root, "20260314_162030_baseline_empty_stairs"
        )
        assert unique.name.endswith("_01")

    def test_resolve_runtime_inputs_cli_flags_mode(self):
        args = type("Args", (), {
            "scenario": "baseline_empty",
            "duration": "5m",
            "data_root": "D:/csi_data",
            "dataset_label": "Escaleras Base",
        })()
        result = resolve_runtime_inputs(args)
        assert result["capture_mode"] == "cli_flags"
        assert result["duration_s"] == 300.0
        assert result["dataset_label"] == "escaleras_base"
        assert str(result["data_root"]).replace("\\", "/").endswith("D:/csi_data")

    def test_resolve_runtime_inputs_interactive_mode(self):
        args = type("Args", (), {
            "scenario": None,
            "duration": None,
            "data_root": None,
            "dataset_label": None,
        })()
        answers = iter(["baseline_empty", "5m", "D:/dataset", "stairs_walk"])
        result = resolve_runtime_inputs(args, input_fn=lambda _: next(answers))
        assert result["capture_mode"] == "interactive"
        assert result["scenario"] == "baseline_empty"
        assert result["duration_s"] == 300.0
        assert str(result["data_root"]).replace("\\", "/").endswith("D:/dataset")
        assert result["dataset_label"] == "stairs_walk"

    def test_resolve_runtime_inputs_interactive_mode_default_data_root(self):
        args = type("Args", (), {
            "scenario": None,
            "duration": None,
            "data_root": None,
            "dataset_label": None,
        })()
        # Empty answer for data root should keep the default "data".
        answers = iter(["baseline_empty", "30s", "", "test"])
        result = resolve_runtime_inputs(args, input_fn=lambda _: next(answers))
        assert result["capture_mode"] == "interactive"
        assert result["duration_s"] == 30.0
        assert result["data_root"] == DEFAULT_DATA_ROOT
        assert result["dataset_label"] == "test"

    def test_normalize_mac_valid(self):
        assert normalize_mac("D8:47:32:2E:4C:F9") == "d8:47:32:2e:4c:f9"

    def test_normalize_mac_invalid(self):
        with pytest.raises(ValueError):
            normalize_mac("d8:47:32:zz:4c:f9")

    def test_collect_mac_summary_and_warnings(self, tmp_path):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        csv_path = raw_dir / "baseline_empty_pos01_node01_20260314.csv"
        csv_path.write_text(
            "# Wi-Fi Vision 3D - CSI Capture\n"
            "timestamp_us,mac,rssi,rate,sig_mode,mcs,cwb,smoothing,not_sounding,"
            "aggregation,stbc,fec,sgi,channel,secondary_channel,rx_seq,len,"
            "first_word_invalid,csi_data\n"
            "1000,d8:47:32:2e:4c:f9,-21,11,1,7,1,1,1,0,1,0,1,6,2,1,228,0,0 0 0 0\n"
            "2000,d8:47:32:2e:4c:f9,-21,11,1,7,1,1,1,0,1,0,1,6,2,2,228,0,0 0 0 0\n"
            "3000,20:6e:f1:99:a4:98,-45,11,1,7,1,1,1,0,1,0,1,6,2,3,228,0,0 0 0 0\n",
            encoding="utf-8",
        )

        summary = collect_mac_summary(raw_dir)
        counts = summary["by_node"]["1"]
        assert counts["d8:47:32:2e:4c:f9"] == 2
        assert counts["20:6e:f1:99:a4:98"] == 1

        warns = evaluate_mac_summary(summary, expected_mac="d8:47:32:2e:4c:f9")
        assert len(warns) >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  Tests: analyze_csi.py
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadCsiFile:
    def test_load_synthetic(self, tmp_path):
        csv_path = tmp_path / "pos01_node01_test.csv"
        _write_synthetic_csv(csv_path, n_frames=50, node_id=1, position_id=1)

        data = load_csi_file(csv_path)
        assert data["n_frames"] == 50
        assert data["n_subcarriers"] == N_SUBCARRIERS
        assert data["amplitude"].shape == (50, N_SUBCARRIERS)
        assert data["phase"].shape == (50, N_SUBCARRIERS)
        assert len(data["rssi"]) == 50
        assert data["avg_hz"] >= 45.0
        assert "Node ID" in data["metadata"]

    def test_load_empty_file(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("# Empty file\ntimestamp_us,mac\n", encoding="utf-8")

        data = load_csi_file(csv_path)
        assert data["n_frames"] == 0

    def test_amplitudes_are_positive(self, tmp_path):
        csv_path = tmp_path / "pos01_node01_amp.csv"
        _write_synthetic_csv(csv_path, n_frames=20)

        data = load_csi_file(csv_path)
        assert np.all(data["amplitude"] >= 0)


class TestExportBaseline:
    def test_export_creates_json(self, tmp_path):
        data_dir = tmp_path / "round1_empty"
        data_dir.mkdir()
        _write_synthetic_csv(
            data_dir / "baseline_empty_pos01_node01_20260314.csv",
            n_frames=100, node_id=1
        )
        _write_synthetic_csv(
            data_dir / "baseline_empty_pos02_node02_20260314.csv",
            n_frames=100, node_id=2
        )

        output_path = tmp_path / "baseline.json"
        export_baseline(data_dir, output_path)

        assert output_path.exists()
        with open(output_path) as f:
            baseline = json.load(f)

        assert len(baseline) >= 1
        for node_id, node_data in baseline.items():
            assert "amplitude_mean" in node_data
            assert "amplitude_std" in node_data
            assert len(node_data["amplitude_mean"]) == N_SUBCARRIERS


# ═══════════════════════════════════════════════════════════════════════════
#  Tests: spatial_filter.py
# ═══════════════════════════════════════════════════════════════════════════

class TestLineSegmentInsideBox:
    def test_fully_inside(self):
        tx = np.array([1.0, 1.0, 1.0])
        rx = np.array([2.0, 2.0, 2.0])
        box_min = np.array([0.0, 0.0, 0.0])
        box_max = np.array([3.0, 3.0, 3.0])
        frac = _line_segment_inside_box(tx, rx, box_min, box_max)
        assert frac > 0.99

    def test_fully_outside(self):
        tx = np.array([10.0, 10.0, 10.0])
        rx = np.array([20.0, 20.0, 20.0])
        box_min = np.array([0.0, 0.0, 0.0])
        box_max = np.array([3.0, 3.0, 3.0])
        frac = _line_segment_inside_box(tx, rx, box_min, box_max)
        assert frac < 0.01

    def test_partially_inside(self):
        tx = np.array([0.0, 0.0, 0.0])
        rx = np.array([6.0, 0.0, 0.0])
        box_min = np.array([0.0, -1.0, -1.0])
        box_max = np.array([3.0, 1.0, 1.0])
        frac = _line_segment_inside_box(tx, rx, box_min, box_max)
        assert 0.4 < frac < 0.6


class TestComputeZoneWeights:
    def test_weights_in_valid_range(self):
        positions, router = _make_positions_and_router()
        weights = compute_zone_weights(positions, router)
        assert len(weights) == 8
        for nid, w in weights.items():
            assert 0.1 <= w <= 1.0, f"Node {nid} weight {w} out of range"

    def test_explicit_zone_box(self):
        positions, router = _make_positions_and_router()
        zone_box = {
            "x_min": 0.5, "x_max": 2.5,
            "y_min": 0.5, "y_max": 3.5,
            "z_min": 0.0, "z_max": 2.8,
        }
        weights = compute_zone_weights(positions, router, zone_box)
        assert len(weights) == 8
        for w in weights.values():
            assert 0.1 <= w <= 1.0


class TestSpatialZoneFilter:
    @pytest.fixture()
    def filter_setup(self):
        """Create a standard SpatialZoneFilter with synthetic baseline."""
        positions, router = _make_positions_and_router()
        zone_weights = compute_zone_weights(positions, router)
        baseline_stats = _make_baseline_stats(n_nodes=8, amplitude_base=30.0, noise_std=2.0)
        filt = SpatialZoneFilter(
            baseline_stats, zone_weights,
            activation_sigma=3.0,
            consensus_threshold=0.5,
        )
        return filt, baseline_stats

    def test_node_activity_quiet(self, filter_setup):
        filt, baseline_stats = filter_setup
        bl = baseline_stats["1"]
        quiet_amp = np.array(bl["amplitude_mean"])
        activity = filt.node_activity(1, quiet_amp)
        assert activity == 0.0

    def test_node_activity_perturbed(self, filter_setup):
        filt, baseline_stats = filter_setup
        bl = baseline_stats["1"]
        perturbed = np.array(bl["amplitude_mean"]) + 30.0
        activity = filt.node_activity(1, perturbed)
        assert activity > 0.0

    def test_node_is_active_false_when_quiet(self, filter_setup):
        filt, baseline_stats = filter_setup
        quiet = np.array(baseline_stats["1"]["amplitude_mean"])
        assert filt.node_is_active(1, quiet) is False

    def test_node_is_active_true_when_perturbed(self, filter_setup):
        filt, baseline_stats = filter_setup
        perturbed = np.array(baseline_stats["1"]["amplitude_mean"]) + 30.0
        assert filt.node_is_active(1, perturbed) is True

    def test_score_frame_none_active(self, filter_setup):
        filt, baseline_stats = filter_setup
        frame = {}
        for nid in range(1, 9):
            frame[nid] = np.array(baseline_stats[str(nid)]["amplitude_mean"])
        score = filt.score_frame(frame)
        assert score == 0.0

    def test_score_frame_all_active(self, filter_setup):
        filt, baseline_stats = filter_setup
        frame = {}
        for nid in range(1, 9):
            frame[nid] = np.array(baseline_stats[str(nid)]["amplitude_mean"]) + 50.0
        score = filt.score_frame(frame)
        assert score > 0.9

    def test_is_zone_event_false_when_quiet(self, filter_setup):
        filt, baseline_stats = filter_setup
        frame = {nid: np.array(baseline_stats[str(nid)]["amplitude_mean"])
                 for nid in range(1, 9)}
        assert filt.is_zone_event(frame) is False

    def test_is_zone_event_true_when_all_perturbed(self, filter_setup):
        filt, baseline_stats = filter_setup
        frame = {nid: np.array(baseline_stats[str(nid)]["amplitude_mean"]) + 50.0
                 for nid in range(1, 9)}
        assert filt.is_zone_event(frame) is True

    def test_unknown_node_returns_zero_activity(self, filter_setup):
        filt, _ = filter_setup
        assert filt.node_activity(99, np.zeros(N_SUBCARRIERS)) == 0.0


class TestAnalyzeSession:
    def test_analyze_session_empty_room(self):
        positions, router = _make_positions_and_router()
        zone_weights = compute_zone_weights(positions, router)
        baseline_stats = _make_baseline_stats()

        filt = SpatialZoneFilter(baseline_stats, zone_weights)

        rng = np.random.default_rng(42)
        session_data = {}
        for nid in range(1, 9):
            bl_mean = np.array(baseline_stats[str(nid)]["amplitude_mean"])
            bl_std = np.array(baseline_stats[str(nid)]["amplitude_std"])
            frames = bl_mean + rng.normal(0, bl_std * 0.5, (100, N_SUBCARRIERS))
            session_data[nid] = frames

        result = filt.analyze_session(session_data)
        assert result["n_frames"] == 100
        assert result["event_fraction"] < 0.1, (
            "Empty room should not trigger many zone events"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  SCENARIO TEST: Person inside staircase vs person outside (kitchen)
# ═══════════════════════════════════════════════════════════════════════════

class TestSpatialInsideVsOutside:
    """
    Core scenario test.  Validates that the spatial consensus filter can
    distinguish a person *inside* the staircase (affects most Tx-Rx links)
    from a person *outside* in the kitchen (affects only 2 peripheral links).

    Setup:
        - 8 nodes surrounding the staircase
        - Router on one wall (centered)
        - Baseline = quiet room (gaussian noise around a fixed amplitude)

    Scenario A -- person INSIDE:
        Perturb 7 out of 8 nodes with a large delta (+25 amplitude).
        Expected: is_zone_event() == True, high score.

    Scenario B -- person OUTSIDE (kitchen):
        Perturb only 2 peripheral nodes (e.g. nodes 4 and 8, far corner).
        Expected: is_zone_event() == False, low score.
    """

    @pytest.fixture()
    def setup(self):
        positions, router = _make_positions_and_router()
        zone_weights = compute_zone_weights(positions, router)
        baseline_stats = _make_baseline_stats(
            n_nodes=8, amplitude_base=30.0, noise_std=2.0
        )
        filt = SpatialZoneFilter(
            baseline_stats, zone_weights,
            activation_sigma=3.0,
            consensus_threshold=0.5,
        )
        return filt, baseline_stats, zone_weights

    def test_person_inside_detected(self, setup):
        """A person inside the staircase perturbs most nodes -> event detected."""
        filt, baseline_stats, _ = setup

        frame = {}
        perturbation = 25.0
        inside_nodes = [1, 2, 3, 5, 6, 7, 8]
        outside_node = 4

        for nid in range(1, 9):
            base = np.array(baseline_stats[str(nid)]["amplitude_mean"])
            if nid in inside_nodes:
                frame[nid] = base + perturbation
            else:
                frame[nid] = base.copy()

        score = filt.score_frame(frame)
        assert score >= 0.5, (
            f"Person inside should produce score >= 0.5, got {score:.3f}"
        )
        assert filt.is_zone_event(frame) is True

    def test_person_outside_not_detected(self, setup):
        """A person in the kitchen only perturbs 2 peripheral nodes -> no event."""
        filt, baseline_stats, zone_weights = setup

        frame = {}
        perturbation = 25.0
        kitchen_nodes = {4, 8}

        for nid in range(1, 9):
            base = np.array(baseline_stats[str(nid)]["amplitude_mean"])
            if nid in kitchen_nodes:
                frame[nid] = base + perturbation
            else:
                frame[nid] = base.copy()

        score = filt.score_frame(frame)
        assert score < 0.5, (
            f"Person outside should produce score < 0.5, got {score:.3f}"
        )
        assert filt.is_zone_event(frame) is False

    def test_full_session_inside_vs_outside(self, setup):
        """
        Multi-frame scenario: 100 frames of person inside vs 100 frames
        of person outside.  Verifies that event_fraction differs sharply.
        """
        filt, baseline_stats, _ = setup
        rng = np.random.default_rng(123)
        perturbation = 25.0

        # -- Person INSIDE: perturb 7/8 nodes for 100 frames --
        inside_data = {}
        for nid in range(1, 9):
            bl = np.array(baseline_stats[str(nid)]["amplitude_mean"])
            noise = rng.normal(0, 1.0, (100, N_SUBCARRIERS))
            if nid != 4:
                inside_data[nid] = bl + perturbation + noise
            else:
                inside_data[nid] = bl + noise

        inside_result = filt.analyze_session(inside_data)

        # -- Person OUTSIDE: perturb only 2/8 nodes for 100 frames --
        outside_data = {}
        for nid in range(1, 9):
            bl = np.array(baseline_stats[str(nid)]["amplitude_mean"])
            noise = rng.normal(0, 1.0, (100, N_SUBCARRIERS))
            if nid in (4, 8):
                outside_data[nid] = bl + perturbation + noise
            else:
                outside_data[nid] = bl + noise

        outside_result = filt.analyze_session(outside_data)

        assert inside_result["event_fraction"] > 0.8, (
            f"Person inside should have high event fraction, "
            f"got {inside_result['event_fraction']}"
        )
        assert outside_result["event_fraction"] < 0.2, (
            f"Person outside should have low event fraction, "
            f"got {outside_result['event_fraction']}"
        )

        assert inside_result["mean_score"] > outside_result["mean_score"], (
            "Inside scenario should have strictly higher mean score than outside"
        )

    def test_hallway_person_single_node(self, setup):
        """
        Person in the hallway affects exactly 1 node.
        Should NOT trigger zone event regardless of perturbation magnitude.
        """
        filt, baseline_stats, _ = setup

        frame = {}
        for nid in range(1, 9):
            base = np.array(baseline_stats[str(nid)]["amplitude_mean"])
            if nid == 6:
                frame[nid] = base + 50.0
            else:
                frame[nid] = base.copy()

        assert filt.is_zone_event(frame) is False, (
            "Single-node perturbation must not trigger zone event"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Tests: temporal synchronization (capture_csi.py + spatial_filter.py)
# ═══════════════════════════════════════════════════════════════════════════

T0_ANCHOR = 1_741_000_000_000_000  # arbitrary fixed epoch in µs


class TestReadT0HostUs:
    def test_reads_value_from_header(self, tmp_path):
        csv_path = tmp_path / "node01.csv"
        _write_synthetic_csv(csv_path, n_frames=10, t0_host_us=T0_ANCHOR)
        assert read_t0_host_us(csv_path) == T0_ANCHOR

    def test_returns_zero_for_legacy_file(self, tmp_path):
        """Legacy CSV without # t0_host_us line must return 0."""
        csv_path = tmp_path / "legacy.csv"
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("# Wi-Fi Vision 3D - CSI Capture\n")
            f.write("# Node ID: 1\n")
            f.write("timestamp_us,mac\n")
        assert read_t0_host_us(csv_path) == 0

    def test_returns_zero_for_missing_file(self, tmp_path):
        assert read_t0_host_us(tmp_path / "nonexistent.csv") == 0


class TestAlignNodesByTimestamp:
    """Tests for align_nodes_by_timestamp() in spatial_filter.py."""

    def _make_node_data(self, n_frames, t0, hz=50.0, n_sub=N_SUBCARRIERS,
                        amp_base=30.0, node_id=1):
        """Build a node_data entry with t_abs_us and amplitude."""
        dt_us = int(1_000_000 / hz)
        firmware_ts = np.arange(n_frames, dtype=np.int64) * dt_us
        t_abs = firmware_ts + t0
        rng = np.random.default_rng(node_id)
        amp = np.full((n_frames, n_sub), amp_base) + rng.normal(0, 1, (n_frames, n_sub))
        return {"amplitude": amp, "t_abs_us": t_abs}

    def _make_node_data_with_drift(self, n_frames, t0, hz=50.0,
                                   drift_ppm=100, n_sub=N_SUBCARRIERS,
                                   amp_base=30.0, node_id=1):
        """
        Build a node with realistic clock drift.
        drift_ppm: each frame interval grows by drift_ppm µs per second.
        E.g. drift_ppm=100 means +0.01% → at 50 Hz, each frame is 20 µs + 0.002 µs longer.
        Accumulated over 300s this gives ~600 ms total drift vs a perfect clock.
        """
        nominal_dt = 1_000_000 / hz
        # Each frame dt increases linearly: dt_i = nominal + drift_ppm/1e6 * nominal * i
        dts = nominal_dt * (1 + drift_ppm / 1e6 * np.arange(n_frames))
        firmware_ts = np.concatenate([[0], np.cumsum(dts[:-1])]).astype(np.int64)
        t_abs = firmware_ts + t0
        rng = np.random.default_rng(node_id)
        amp = np.full((n_frames, n_sub), amp_base) + rng.normal(0, 1, (n_frames, n_sub))
        return {"amplitude": amp, "t_abs_us": t_abs}

    # ── Basic shape tests ────────────────────────────────────────────────

    def test_single_node_returns_same_length(self):
        data = {1: self._make_node_data(100, T0_ANCHOR)}
        aligned = align_nodes_by_timestamp(data)
        assert aligned[1].shape[0] == 100

    def test_two_nodes_same_t0_perfectly_aligned(self):
        """Two nodes with identical t0 and same frame rate align 1:1."""
        data = {
            1: self._make_node_data(100, T0_ANCHOR, hz=50),
            2: self._make_node_data(100, T0_ANCHOR, hz=50, node_id=2),
        }
        aligned = align_nodes_by_timestamp(data)
        assert aligned[1].shape[0] == aligned[2].shape[0]
        assert aligned[1].shape[0] > 90

    # ── Multi-node (3+) alignment ────────────────────────────────────────

    def test_three_nodes_same_t0(self):
        """Three nodes with same t0 must all produce the same aligned length."""
        data = {
            1: self._make_node_data(100, T0_ANCHOR, hz=50),
            2: self._make_node_data(100, T0_ANCHOR, hz=50, node_id=2),
            3: self._make_node_data(100, T0_ANCHOR, hz=50, node_id=3),
        }
        aligned = align_nodes_by_timestamp(data)
        lengths = [aligned[nid].shape[0] for nid in (1, 2, 3)]
        assert len(set(lengths)) == 1, f"Mismatched lengths: {lengths}"
        assert lengths[0] > 90

    def test_five_nodes_small_staggered_offsets(self):
        """
        Simulate 5 nodes where each one starts 2 ms later than the previous
        (realistic barrier skew ~1-5 ms per node).  All should align well.
        """
        data = {}
        for i in range(1, 6):
            offset_us = (i - 1) * 2_000  # 0, 2, 4, 6, 8 ms stagger
            data[i] = self._make_node_data(
                100, T0_ANCHOR + offset_us, hz=50, node_id=i
            )
        aligned = align_nodes_by_timestamp(data)
        lengths = [aligned[nid].shape[0] for nid in range(1, 6)]
        # All nodes should agree on length and keep most frames
        assert len(set(lengths)) == 1, f"Lengths differ: {lengths}"
        assert lengths[0] > 85, f"Too many frames dropped: {lengths[0]}"

    def test_eight_nodes_full_deployment(self):
        """
        Simulate a complete 8-node deployment with realistic start skew
        (barrier reduces it to ~1-5 ms per node) and verify that all 8 nodes
        produce the same aligned length with minimal frame loss.
        """
        data = {}
        for i in range(1, 9):
            # Random offset 0-5 ms each (post-barrier jitter)
            rng = np.random.default_rng(i * 7)
            offset_us = int(rng.uniform(0, 5_000))
            data[i] = self._make_node_data(
                200, T0_ANCHOR + offset_us, hz=50, node_id=i
            )
        aligned = align_nodes_by_timestamp(data)
        lengths = [aligned[nid].shape[0] for nid in range(1, 9)]
        assert len(set(lengths)) == 1, f"8-node lengths differ: {lengths}"
        assert lengths[0] > 190, f"Excessive frame loss in 8-node: {lengths[0]}"

    # ── Frame rate mismatch ──────────────────────────────────────────────

    def test_two_nodes_different_hz(self):
        """
        One node captures at 47 Hz, another at 53 Hz (realistic serial jitter).
        Alignment should still produce matching lengths with few dropped frames.
        """
        data = {
            1: self._make_node_data(100, T0_ANCHOR, hz=47.0),
            2: self._make_node_data(100, T0_ANCHOR, hz=53.0, node_id=2),
        }
        aligned = align_nodes_by_timestamp(data)
        assert aligned[1].shape[0] == aligned[2].shape[0]
        # With different rates the slower node sets the pace; expect ~90% retention
        assert aligned[1].shape[0] > 80

    def test_two_nodes_different_frame_counts(self):
        """
        One node captures 480 frames, another 510 (same session, slightly different
        durations due to serial buffering).  Output must be the same length.
        """
        data = {
            1: self._make_node_data(480, T0_ANCHOR, hz=50),
            2: self._make_node_data(510, T0_ANCHOR, hz=50, node_id=2),
        }
        aligned = align_nodes_by_timestamp(data)
        assert aligned[1].shape[0] == aligned[2].shape[0]
        assert aligned[1].shape[0] >= 480  # should keep all ref frames

    # ── Clock drift ──────────────────────────────────────────────────────

    def test_two_nodes_realistic_drift_5min(self):
        """
        Simulate a 5-minute session with 100 ppm drift between two nodes.
        At 50 Hz × 300s = 15,000 frames.  After 5 min the drift is ~30 ms,
        which is just outside the ±20 ms window for the last few frames.
        Verify that > 95% of frames are retained (drift only matters near the end).
        """
        n_frames = 15_000  # 5 min at 50 Hz
        data = {
            1: self._make_node_data(n_frames, T0_ANCHOR, hz=50),
            2: self._make_node_data_with_drift(
                n_frames, T0_ANCHOR, hz=50, drift_ppm=100, node_id=2
            ),
        }
        aligned = align_nodes_by_timestamp(data)
        assert aligned[1].shape[0] == aligned[2].shape[0]
        retention = aligned[1].shape[0] / n_frames
        assert retention > 0.95, (
            f"Drift caused excessive frame loss: {retention:.1%} retention "
            f"({aligned[1].shape[0]}/{n_frames} frames kept)"
        )

    # ── Temporal correctness (not just shape) ───────────────────────────

    def test_aligned_frames_are_temporally_closest(self):
        """
        Verify that the alignment pairs each reference frame with the
        temporally nearest frame from the other node, not an arbitrary one.
        Node 2 starts offset_us later than node 1 but at the same frame rate.
        Since offset (3 ms) < half-period (10 ms), frame i of node 1 should
        pair with frame i of node 2.  We encode the frame index in the
        amplitude so we can check: aligned_node2[i] ≈ 100 + i (not 100 + i±1).
        """
        n = 20
        dt_us = 20_000  # 50 Hz → period = 20 ms
        offset_us = 3_000  # 3 ms — well inside ±10 ms half-period

        # Node 1: amplitude[i] = i (index-labelled)
        t_abs_1 = T0_ANCHOR + np.arange(n, dtype=np.int64) * dt_us
        amp_1 = np.arange(n, dtype=float).reshape(n, 1) * np.ones((1, N_SUBCARRIERS))

        # Node 2 starts 3 ms later; amplitude[i] = 100 + i
        t_abs_2 = T0_ANCHOR + offset_us + np.arange(n, dtype=np.int64) * dt_us
        amp_2 = (100 + np.arange(n, dtype=float)).reshape(n, 1) * np.ones((1, N_SUBCARRIERS))

        data = {
            1: {"amplitude": amp_1, "t_abs_us": t_abs_1},
            2: {"amplitude": amp_2, "t_abs_us": t_abs_2},
        }
        aligned = align_nodes_by_timestamp(data)

        assert aligned[1].shape[0] > 0, "No frames survived alignment"
        for i in range(aligned[1].shape[0]):
            # amp_1[i] = i;  amp_2[j] = 100+j  →  if paired correctly: amp_2 - amp_1 = 100
            ref_val = aligned[1][i, 0]    # should be i
            other_val = aligned[2][i, 0]  # should be 100 + i
            pairing_delta = other_val - ref_val
            assert abs(pairing_delta - 100) < 2, (
                f"Frame {i}: node1={ref_val:.0f}, node2={other_val:.0f} — "
                f"expected offset=100 (same-index pairing), got {pairing_delta:.0f}"
            )

    # ── Edge cases and fallbacks ──────────────────────────────────────────

    def test_two_nodes_small_offset_within_window(self):
        """5 ms offset between nodes — within ±20 ms window, should align fully."""
        offset_us = 5_000  # 5 ms
        data = {
            1: self._make_node_data(100, T0_ANCHOR, hz=50),
            2: self._make_node_data(100, T0_ANCHOR + offset_us, hz=50, node_id=2),
        }
        aligned = align_nodes_by_timestamp(data)
        assert aligned[1].shape[0] == aligned[2].shape[0]
        assert aligned[1].shape[0] > 90

    def test_two_nodes_large_offset_outside_window(self):
        """2 second offset, much larger than the 1s total duration — zero overlap."""
        offset_us = 2_000_000  # 2000 ms, >> 1s total session length
        data = {
            1: self._make_node_data(50, T0_ANCHOR, hz=50),
            2: self._make_node_data(50, T0_ANCHOR + offset_us, hz=50, node_id=2),
        }
        aligned = align_nodes_by_timestamp(data)
        for nid, arr in aligned.items():
            assert arr.shape[0] == 0, (
                f"Node {nid}: expected 0 frames after large offset, got {arr.shape[0]}"
            )

    def test_legacy_fallback_when_no_timestamps(self):
        """If t_abs_us is all zeros (legacy), falls back to index pairing."""
        data = {
            1: {"amplitude": np.ones((80, N_SUBCARRIERS)),
                "t_abs_us": np.zeros(80, dtype=np.int64)},
            2: {"amplitude": np.ones((100, N_SUBCARRIERS)),
                "t_abs_us": np.zeros(100, dtype=np.int64)},
        }
        aligned = align_nodes_by_timestamp(data)
        # Legacy path: truncated to min length
        assert aligned[1].shape[0] == 80
        assert aligned[2].shape[0] == 80

    def test_csv_t0_roundtrip(self, tmp_path):
        """Write a synthetic CSV with t0, load it, compute t_abs_us, verify alignment."""
        t0 = T0_ANCHOR
        path1 = tmp_path / "node01.csv"
        path2 = tmp_path / "node02.csv"
        _write_synthetic_csv(path1, n_frames=50, node_id=1, t0_host_us=t0)
        _write_synthetic_csv(path2, n_frames=50, node_id=2, t0_host_us=t0)

        from analyze_csi import load_csi_file
        for path in (path1, path2):
            d = load_csi_file(path)
            t0_read = int(d["metadata"].get("t0_host_us", 0))
            assert t0_read == t0, f"t0_host_us mismatch in {path.name}"
            t_abs = d["timestamps"] + t0_read
            assert t_abs[0] >= t0  # first frame must be at or after anchor
