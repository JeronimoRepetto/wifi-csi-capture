#!/usr/bin/env python3
"""
Wi-Fi Vision 3D - Measurement Protocol Manager

Defines the 8 node positions and guides the sequential capture process
using 2 ESP32-S3 across 4 rounds.

Usage:
    python measurement_protocol.py                     # Show protocol
    python measurement_protocol.py --round 1           # Execute round 1
    python measurement_protocol.py --round 1 --empty   # Capture empty room
    python measurement_protocol.py --round 1 --person  # Capture with person
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

POSITIONS = {
    1: {
        "label": "Techo - Esquina Frontal Izquierda",
        "zone": "ceiling",
        "x": 0.0, "y": 0.0, "z": 2.5,
        "orientation": "pointing_down",
        "notes": "Esquina superior, mirando hacia el centro de la escalera"
    },
    2: {
        "label": "Techo - Esquina Frontal Derecha",
        "zone": "ceiling",
        "x": 3.0, "y": 0.0, "z": 2.5,
        "orientation": "pointing_down",
        "notes": "Esquina superior opuesta"
    },
    3: {
        "label": "Techo - Esquina Trasera Izquierda",
        "zone": "ceiling",
        "x": 0.0, "y": 4.0, "z": 2.5,
        "orientation": "pointing_down",
        "notes": "Esquina superior posterior izquierda"
    },
    4: {
        "label": "Techo - Esquina Trasera Derecha",
        "zone": "ceiling",
        "x": 3.0, "y": 4.0, "z": 2.5,
        "orientation": "pointing_down",
        "notes": "Esquina superior posterior derecha"
    },
    5: {
        "label": "Suelo - Esquina Frontal Izquierda",
        "zone": "floor",
        "x": 0.0, "y": 0.0, "z": 0.15,
        "orientation": "pointing_up",
        "notes": "Zocalo inferior, debajo de posicion 1"
    },
    6: {
        "label": "Suelo - Esquina Frontal Derecha",
        "zone": "floor",
        "x": 3.0, "y": 0.0, "z": 0.15,
        "orientation": "pointing_up",
        "notes": "Zocalo inferior, debajo de posicion 2"
    },
    7: {
        "label": "Suelo - Esquina Trasera Izquierda",
        "zone": "floor",
        "x": 0.0, "y": 4.0, "z": 0.15,
        "orientation": "pointing_up",
        "notes": "Zocalo inferior, debajo de posicion 3"
    },
    8: {
        "label": "Suelo - Esquina Trasera Derecha",
        "zone": "floor",
        "x": 3.0, "y": 4.0, "z": 0.15,
        "orientation": "pointing_up",
        "notes": "Zocalo inferior, debajo de posicion 4"
    },
}

ROUNDS = {
    1: {"node_a": 1, "node_b": 2, "desc": "Techo frontal (izquierda + derecha)"},
    2: {"node_a": 3, "node_b": 4, "desc": "Techo trasero (izquierda + derecha)"},
    3: {"node_a": 5, "node_b": 6, "desc": "Suelo frontal (izquierda + derecha)"},
    4: {"node_a": 7, "node_b": 8, "desc": "Suelo trasero (izquierda + derecha)"},
}

# Router (Tx) position
ROUTER_POSITION = {
    "label": "Dedicated 2.4GHz Router (Tx)",
    "x": 1.5, "y": 2.0, "z": 1.5,
    "notes": "Pared central de la escalera, a media altura"
}

EMPTY_DURATION = 300   # 5 minutes
PERSON_DURATION = 120  # 2 minutes


def print_protocol():
    """Display the full measurement protocol."""
    print("=" * 70)
    print("  Wi-Fi Vision 3D - PROTOCOLO DE MEDICION")
    print("=" * 70)
    print()
    print("  ROUTER EMISOR (Tx):")
    r = ROUTER_POSITION
    print(f"    {r['label']}")
    print(f"    Posicion: ({r['x']}, {r['y']}, {r['z']}) metros")
    print(f"    Nota: {r['notes']}")
    print()
    print("-" * 70)
    print("  POSICIONES DE NODOS RECEPTORES (Rx):")
    print("-" * 70)
    print()

    for pos_id, pos in POSITIONS.items():
        marker = "^" if pos["zone"] == "ceiling" else "v"
        print(f"  [{marker}] Posicion {pos_id}: {pos['label']}")
        print(f"      Coordenadas: ({pos['x']}, {pos['y']}, {pos['z']}) m")
        print(f"      Orientacion: {pos['orientation']}")
        print(f"      Notas: {pos['notes']}")
        print()

    print("-" * 70)
    print("  RONDAS DE CAPTURA (2 ESP32-S3 por ronda):")
    print("-" * 70)
    print()

    for round_id, rd in ROUNDS.items():
        pa = POSITIONS[rd["node_a"]]
        pb = POSITIONS[rd["node_b"]]
        print(f"  RONDA {round_id}: {rd['desc']}")
        print(f"    ESP32 #A -> Posicion {rd['node_a']}: ({pa['x']},{pa['y']},{pa['z']})")
        print(f"    ESP32 #B -> Posicion {rd['node_b']}: ({pb['x']},{pb['y']},{pb['z']})")
        print(f"    Pasos:")
        print(f"      1. Colocar ESP32-S3 en las posiciones indicadas")
        print(f"      2. Fotografiar la ubicacion exacta de cada nodo")
        print(f"      3. Captura VACIA: {EMPTY_DURATION}s con habitacion despejada")
        print(f"      4. Captura PERSONA: {PERSON_DURATION}s con persona caminando")
        print(f"      5. Retirar ESP32-S3 y mover a siguiente ronda")
        print()

    print("-" * 70)
    print("  SISTEMA DE COORDENADAS:")
    print("-" * 70)
    print()
    print("  Origen (0,0,0): Esquina frontal izquierda del suelo")
    print("  X: Ancho de la escalera (izquierda -> derecha)")
    print("  Y: Profundidad de la escalera (frente -> fondo)")
    print("  Z: Altura (suelo -> techo)")
    print()
    print("  IMPORTANTE: Medir las dimensiones reales de tu escalera y")
    print("  actualizar las coordenadas en este archivo antes de capturar.")
    print()


def export_positions(filepath: Path):
    """Export positions to JSON for use by analysis scripts."""
    data = {
        "router": ROUTER_POSITION,
        "positions": POSITIONS,
        "rounds": ROUNDS,
        "coordinate_system": {
            "origin": "Esquina frontal izquierda del suelo",
            "x": "Ancho (izquierda -> derecha)",
            "y": "Profundidad (frente -> fondo)",
            "z": "Altura (suelo -> techo)",
            "units": "meters"
        }
    }
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Posiciones exportadas a: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Wi-Fi Vision 3D - Measurement Protocol Manager"
    )
    parser.add_argument("--round", type=int, choices=[1, 2, 3, 4],
                        help="Execute a specific capture round")
    parser.add_argument("--empty", action="store_true",
                        help="Capture empty room (5 min)")
    parser.add_argument("--person", action="store_true",
                        help="Capture with person walking (2 min)")
    parser.add_argument("--port1", help="Serial port for node A")
    parser.add_argument("--port2", help="Serial port for node B")
    parser.add_argument("--export", default="data/positions.json",
                        help="Export positions JSON file")
    args = parser.parse_args()

    if args.round is None:
        print_protocol()
        export_positions(Path(args.export))
        return

    rd = ROUNDS[args.round]
    print(f"\n=== RONDA {args.round}: {rd['desc']} ===\n")

    if not args.port1 or not args.port2:
        print("ERROR: Debes especificar --port1 y --port2 para ejecutar una ronda.")
        print(f"  Ejemplo: python measurement_protocol.py --round {args.round} "
              f"--empty --port1 COM3 --port2 COM4")
        sys.exit(1)

    if not args.empty and not args.person:
        print("ERROR: Especifica --empty (habitacion vacia) o --person (con persona).")
        sys.exit(1)

    duration = EMPTY_DURATION if args.empty else PERSON_DURATION
    mode = "empty" if args.empty else "person"

    print(f"Modo: {mode}")
    print(f"Duracion: {duration}s")
    print(f"Nodo A (pos {rd['node_a']}): {args.port1}")
    print(f"Nodo B (pos {rd['node_b']}): {args.port2}")
    print()

    output_dir = f"data/round{args.round}_{mode}"

    cmd = [
        sys.executable, "tools/capture_csi.py",
        "--port1", args.port1,
        "--port2", args.port2,
        "--position", str(args.round),
        "--duration", str(duration),
        "--output", output_dir,
    ]

    print(f"Ejecutando: {' '.join(cmd)}\n")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
