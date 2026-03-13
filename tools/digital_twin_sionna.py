#!/usr/bin/env python3
"""
Wi-Fi Vision 3D - Digital Twin Configuration for NVIDIA Sionna

Sets up the electromagnetic simulation environment:
- Staircase geometry as a 3D scene
- 1 Transmitter (router) + 8 Receivers (ESP32-S3 positions)
- Dielectric material properties for walls, floor, ceiling, and human body
- CSI tensor generation matching ESP32-S3 HT40 format (114 subcarriers)

Prerequisites:
    pip install sionna tensorflow

Usage:
    python digital_twin_sionna.py --generate-scene    # Create scene geometry
    python digital_twin_sionna.py --simulate           # Run ray tracing simulation
    python digital_twin_sionna.py --calibrate baseline.json  # Calibrate vs real data
"""

import argparse
import json
import sys
import numpy as np
from pathlib import Path

SCENE_CONFIG = {
    "name": "Wi-Fi Vision 3D - Staircase",
    "room": {
        "width": 3.0,    # X (meters) - adjust to your staircase
        "depth": 4.0,    # Y (meters)
        "height": 2.8,   # Z (meters)
    },
    "carrier_frequency": 2.437e9,  # Channel 6 center frequency (Hz)
    "bandwidth": 40e6,             # HT40 bandwidth
    "n_subcarriers": 114,          # ESP32-S3 HT40 subcarrier count
    "subcarrier_spacing": 312.5e3, # OFDM subcarrier spacing for HT40
}

ROUTER_TX = {
    "id": "tx_router",
    "label": "Dedicated 2.4GHz Router (Tx)",
    "position": [1.5, 2.0, 1.5],
    "antenna": "omnidirectional",
    "power_dbm": 20,
}

RECEIVERS = {
    1: {"pos": [0.0, 0.0, 2.5], "label": "Techo Frontal Izquierda"},
    2: {"pos": [3.0, 0.0, 2.5], "label": "Techo Frontal Derecha"},
    3: {"pos": [0.0, 4.0, 2.5], "label": "Techo Trasera Izquierda"},
    4: {"pos": [3.0, 4.0, 2.5], "label": "Techo Trasera Derecha"},
    5: {"pos": [0.0, 0.0, 0.15], "label": "Suelo Frontal Izquierda"},
    6: {"pos": [3.0, 0.0, 0.15], "label": "Suelo Frontal Derecha"},
    7: {"pos": [0.0, 4.0, 0.15], "label": "Suelo Trasera Izquierda"},
    8: {"pos": [3.0, 4.0, 0.15], "label": "Suelo Trasera Derecha"},
}

MATERIALS = {
    "drywall": {
        "description": "Tabique de yeso (paredes interiores)",
        "relative_permittivity": 2.94,
        "conductivity": 0.0386,
        "scattering_coefficient": 0.3,
        "scattering_pattern": "lambertian",
    },
    "concrete": {
        "description": "Hormigon (suelo, techo estructural)",
        "relative_permittivity": 5.31,
        "conductivity": 0.0707,
        "scattering_coefficient": 0.5,
        "scattering_pattern": "lambertian",
    },
    "wood": {
        "description": "Madera (escalones, barandilla)",
        "relative_permittivity": 1.99,
        "conductivity": 0.0047,
        "scattering_coefficient": 0.2,
        "scattering_pattern": "lambertian",
    },
    "metal_railing": {
        "description": "Metal (barandilla metalica)",
        "relative_permittivity": 1.0,
        "conductivity": 1e7,
        "scattering_coefficient": 0.1,
        "scattering_pattern": "directive",
    },
    "human_tissue": {
        "description": "Tejido humano equivalente (phantom liquido a 2.45 GHz)",
        "relative_permittivity": 39.2,  # Weighted avg: skin+muscle+fat
        "conductivity": 1.8,
        "scattering_coefficient": 0.6,
        "scattering_pattern": "lambertian",
        "notes": "Based on Cole-Cole model at 2.45 GHz. "
                 "Skin: er=38.1, s=1.46. Muscle: er=52.7, s=1.74. "
                 "Fat: er=5.28, s=0.10. Bone: er=11.4, s=0.39"
    },
}


def generate_scene_config(output_path: Path):
    """Generate the scene configuration JSON for the digital twin."""
    config = {
        "scene": SCENE_CONFIG,
        "transmitter": ROUTER_TX,
        "receivers": {str(k): v for k, v in RECEIVERS.items()},
        "materials": MATERIALS,
        "esp32s3_emulation": {
            "quantization": "int8",
            "first_word_invalid": True,
            "noise_floor_dbm": -90,
            "antenna_gain_dbi": 2.0,
            "clock_drift_ppm": 20,
            "phase_noise_std_rad": 0.15,
        },
        "simulation_params": {
            "max_reflections": 6,
            "max_diffractions": 2,
            "max_scattering": 2,
            "num_samples": 1e6,
            "los_enabled": True,
        }
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"Scene configuration saved to: {output_path}")
    return config


def simulate_csi(config: dict):
    """
    Run electromagnetic ray tracing simulation using Sionna.
    Generates synthetic CSI tensors for all 8 receivers.
    """
    try:
        import tensorflow as tf
        import sionna
        from sionna.rt import Scene, Transmitter, Receiver, PlanarArray
        from sionna.rt import RadioMaterial, LambertianPattern
    except ImportError:
        print("ERROR: Sionna and TensorFlow are required for simulation.")
        print("Install with: pip install sionna tensorflow")
        print()
        print("For now, generating mock CSI data for pipeline testing...")
        return generate_mock_csi(config)

    print("Initializing Sionna ray tracing engine...")

    scene = Scene()
    fc = config["scene"]["carrier_frequency"]
    bw = config["scene"]["bandwidth"]
    n_sc = config["scene"]["n_subcarriers"]

    for mat_name, mat_props in config["materials"].items():
        radio_mat = RadioMaterial(
            mat_name,
            relative_permittivity=mat_props["relative_permittivity"],
            conductivity=mat_props["conductivity"],
        )
        scene.add(radio_mat)

    tx_cfg = config["transmitter"]
    tx = Transmitter(
        name=tx_cfg["id"],
        position=tx_cfg["position"],
    )
    scene.add(tx)

    for rx_id, rx_cfg in config["receivers"].items():
        rx = Receiver(
            name=f"rx_{rx_id}",
            position=rx_cfg["pos"],
        )
        scene.add(rx)

    print(f"Scene configured: 1 Tx + {len(config['receivers'])} Rx")
    print(f"Frequency: {fc/1e9:.3f} GHz, Bandwidth: {bw/1e6:.0f} MHz")
    print(f"Subcarriers: {n_sc}")

    paths = scene.compute_paths(max_depth=6)

    frequencies = tf.cast(
        tf.linspace(fc - bw/2, fc + bw/2, n_sc),
        dtype=tf.float64
    )
    cfr = paths.cfr(frequencies=frequencies)

    print(f"CFR tensor shape: {cfr.shape}")
    return cfr


def generate_mock_csi(config: dict) -> np.ndarray:
    """
    Generate physically-plausible mock CSI data for pipeline testing
    when Sionna is not available.
    """
    n_sc = config["scene"]["n_subcarriers"]
    n_rx = len(config["receivers"])
    n_frames = 1000

    print(f"Generating mock CSI: {n_frames} frames x {n_rx} receivers x {n_sc} subcarriers")

    np.random.seed(42)
    csi_data = np.zeros((n_frames, n_rx, n_sc, 2))  # [frames, rx, sc, (real, imag)]

    for rx_idx in range(n_rx):
        rx_pos = np.array(list(config["receivers"].values())[rx_idx]["pos"])
        tx_pos = np.array(config["transmitter"]["position"])
        distance = np.linalg.norm(rx_pos - tx_pos)

        path_loss_db = 20 * np.log10(distance) + 40
        base_amplitude = 10 ** (-path_loss_db / 20) * 100

        sc_indices = np.arange(n_sc)
        phase_slope = 2 * np.pi * distance / 0.122  # wavelength at 2.4 GHz

        for frame in range(n_frames):
            amp = base_amplitude * (1 + 0.3 * np.sin(2 * np.pi * sc_indices / n_sc * 3))
            amp += np.random.normal(0, base_amplitude * 0.1, n_sc)
            amp = np.clip(amp, 0, 127)

            phase = phase_slope * sc_indices / n_sc + np.random.normal(0, 0.15, n_sc)

            csi_data[frame, rx_idx, :, 0] = amp * np.cos(phase)  # real
            csi_data[frame, rx_idx, :, 1] = amp * np.sin(phase)  # imag

    csi_int8 = np.clip(np.round(csi_data), -128, 127).astype(np.int8)

    output_path = Path("data/mock_csi_synthetic.npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, csi=csi_int8, config=json.dumps(config))
    print(f"Mock CSI data saved to: {output_path}")
    print(f"Shape: {csi_int8.shape} (frames, receivers, subcarriers, real/imag)")

    return csi_int8


def calibrate_simulation(config: dict, baseline_path: Path):
    """
    Compare simulated CSI against real captured baseline to tune
    material properties and antenna characteristics.
    """
    with open(baseline_path, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    print(f"\nCalibrating against baseline: {baseline_path}")
    print(f"Nodes in baseline: {len(baseline)}")

    for node_id, node_data in baseline.items():
        real_amp_mean = np.array(node_data["amplitude_mean"])
        real_amp_std = np.array(node_data["amplitude_std"])
        print(f"\n  Node {node_id}:")
        print(f"    Real RSSI: {node_data['rssi_mean']:.1f} +/- {node_data['rssi_std']:.1f} dBm")
        print(f"    Real amplitude mean: {np.mean(real_amp_mean):.2f}")
        print(f"    Real amplitude std:  {np.mean(real_amp_std):.2f}")
        print(f"    Subcarriers: {len(real_amp_mean)}")

    print("\n  [TODO] When Sionna is installed, this function will:")
    print("  1. Run simulation with current material parameters")
    print("  2. Compare simulated CSI vs real baseline per receiver")
    print("  3. Optimize material permittivity/conductivity via gradient descent")
    print("  4. Report calibration error (RMSE) per subcarrier")


def main():
    parser = argparse.ArgumentParser(
        description="Wi-Fi Vision 3D - Digital Twin (Sionna)"
    )
    parser.add_argument("--generate-scene", action="store_true",
                        help="Generate scene configuration JSON")
    parser.add_argument("--simulate", action="store_true",
                        help="Run ray tracing simulation")
    parser.add_argument("--calibrate", type=str, default=None,
                        help="Calibrate against a real baseline JSON file")
    parser.add_argument("--output", default="data/scene_config.json",
                        help="Output path for scene config")
    args = parser.parse_args()

    if args.generate_scene:
        config = generate_scene_config(Path(args.output))
        print("\nNext step: Install Sionna and run --simulate")

    elif args.simulate:
        config_path = Path(args.output)
        if config_path.exists():
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            print("Scene config not found, generating...")
            config = generate_scene_config(config_path)

        simulate_csi(config)

    elif args.calibrate:
        config_path = Path(args.output)
        if not config_path.exists():
            print("Scene config not found, generating...")
            generate_scene_config(config_path)
        with open(config_path, "r") as f:
            config = json.load(f)
        calibrate_simulation(config, Path(args.calibrate))

    else:
        config = generate_scene_config(Path(args.output))
        print("\nUse --simulate to run ray tracing or --calibrate to tune parameters")


if __name__ == "__main__":
    main()
