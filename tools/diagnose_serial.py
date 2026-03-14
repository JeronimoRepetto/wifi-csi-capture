#!/usr/bin/env python3
"""
Wi-Fi Vision 3D - Serial Diagnostic Tool

Connects to an ESP32-S3 and prints ALL serial output to help diagnose
connection, baud rate, WiFi, and CSI capture issues.

Can auto-detect the correct baud rate by trying 921600 and 115200.

Usage:
    python diagnose_serial.py --port COM3
    python diagnose_serial.py --port COM3 --baud 115200
    python diagnose_serial.py --port COM3 --duration 20
"""

import argparse
import sys
import time

try:
    import serial
except ImportError:
    print("ERROR: pyserial is required. Install with: pip install pyserial")
    sys.exit(1)

BAUD_RATES_TO_TRY = [921600, 115200]
CSI_PREFIX = "CSI_DATA,"


def is_readable_text(data: bytes) -> bool:
    """Check if bytes look like readable ASCII/UTF-8 text."""
    try:
        text = data.decode("utf-8", errors="strict")
        printable = sum(1 for c in text if c.isprintable() or c in "\r\n\t")
        return len(text) > 0 and printable / len(text) > 0.7
    except (UnicodeDecodeError, ZeroDivisionError):
        return False


def auto_detect_baud(port: str) -> int | None:
    """Try each baud rate and return the one that produces readable output."""
    for baud in BAUD_RATES_TO_TRY:
        print(f"  Probando {baud} baud... ", end="", flush=True)
        try:
            ser = serial.Serial(port, baud, timeout=1.5)
            readable_lines = 0
            for _ in range(10):
                raw = ser.readline()
                if raw and is_readable_text(raw):
                    readable_lines += 1
            ser.close()

            if readable_lines >= 2:
                print(f"OK ({readable_lines} lineas legibles)")
                return baud
            else:
                print(f"sin datos legibles ({readable_lines}/10)")
        except serial.SerialException as e:
            print(f"error: {e}")
    return None


def run_diagnostic(port: str, baud: int, duration: float):
    """Read serial output and collect diagnostic statistics."""
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTICO SERIAL - {port} @ {baud} baud")
    print(f"  Duracion: {duration}s")
    print(f"{'='*60}\n")

    try:
        ser = serial.Serial(port, baud, timeout=1)
    except serial.SerialException as e:
        print(f"ERROR: No se puede abrir {port}: {e}")
        return

    stats = {
        "total_lines": 0,
        "csi_lines": 0,
        "boot_lines": 0,
        "wifi_connected": False,
        "wifi_ip": None,
        "wifi_gateway": None,
        "sendto_errors": 0,
        "csi_enabled": False,
        "ping_started": False,
        "first_csi_time": None,
        "last_csi_time": None,
        "error_lines": [],
        "subcarrier_counts": [],
        "sig_mode_legacy": 0,
        "sig_mode_ht": 0,
        "sig_mode_vht": 0,
    }

    start = time.time()
    print("--- Salida serial en vivo ---\n")

    try:
        while time.time() - start < duration:
            raw = ser.readline()
            if not raw:
                continue

            try:
                line = raw.decode("utf-8", errors="replace").strip()
            except Exception:
                continue

            if not line:
                continue

            stats["total_lines"] += 1
            now = time.time() - start

            if line.startswith(CSI_PREFIX):
                stats["csi_lines"] += 1
                if stats["first_csi_time"] is None:
                    stats["first_csi_time"] = now
                stats["last_csi_time"] = now

                parts = line.split(",")
                n_data = len(parts) - 19
                if n_data > 0:
                    stats["subcarrier_counts"].append(n_data // 2)

                if len(parts) > 5:
                    try:
                        sig = int(parts[5])
                        if sig == 0:
                            stats["sig_mode_legacy"] += 1
                        elif sig == 1:
                            stats["sig_mode_ht"] += 1
                        else:
                            stats["sig_mode_vht"] += 1
                    except ValueError:
                        pass

                if stats["csi_lines"] <= 3 or stats["csi_lines"] % 100 == 0:
                    sc = n_data // 2 if n_data > 0 else 0
                    sig_str = parts[5] if len(parts) > 5 else "?"
                    print(f"  [{now:6.1f}s] CSI_DATA #{stats['csi_lines']} "
                          f"({sc} sc, rssi={parts[3] if len(parts)>3 else '?'}, sig_mode={sig_str})")
            else:
                tag = "LOG"
                if "Connected" in line and "Gateway" in line:
                    stats["wifi_connected"] = True
                    for part in line.split():
                        if part.count(".") == 3 and part[0].isdigit():
                            if stats["wifi_ip"] is None:
                                stats["wifi_ip"] = part.rstrip(",")
                            else:
                                stats["wifi_gateway"] = part.rstrip(",")
                    tag = "WIFI"
                elif "CSI capture enabled" in line:
                    stats["csi_enabled"] = True
                    tag = "CSI"
                elif "Starting traffic generator" in line:
                    stats["ping_started"] = True
                    tag = "PING"
                elif "sendto failed" in line:
                    stats["sendto_errors"] += 1
                    tag = "ERR"
                elif "Reconnecting" in line or "WIFI_EVENT" in line:
                    tag = "WIFI"
                elif any(kw in line for kw in ["Error", "error", "ERROR", "fail", "FAIL"]):
                    stats["error_lines"].append(line[:120])
                    tag = "ERR"

                if "rst:" in line.lower() or "boot:" in line.lower():
                    stats["boot_lines"] += 1

                print(f"  [{now:6.1f}s] [{tag:4s}] {line[:150]}")

    except KeyboardInterrupt:
        print("\n\n  (Interrumpido por el usuario)")
    finally:
        ser.close()

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  RESUMEN DIAGNOSTICO")
    print(f"{'='*60}")
    print(f"  Puerto:            {port}")
    print(f"  Baud rate:         {baud}")
    print(f"  Duracion:          {elapsed:.1f}s")
    print(f"  Lineas totales:    {stats['total_lines']}")
    print(f"  Lineas CSI_DATA:   {stats['csi_lines']}")
    print(f"  Boot detectado:    {'Si' if stats['boot_lines'] > 0 else 'No'}")
    print(f"  WiFi conectado:    {'Si' if stats['wifi_connected'] else 'No'}")
    if stats["wifi_ip"]:
        print(f"  IP asignada:       {stats['wifi_ip']}")
    if stats["wifi_gateway"]:
        print(f"  Gateway:           {stats['wifi_gateway']}")
    print(f"  CSI habilitado:    {'Si' if stats['csi_enabled'] else 'No detectado'}")
    print(f"  Ping iniciado:     {'Si' if stats['ping_started'] else 'No detectado'}")
    print(f"  Errores sendto:    {stats['sendto_errors']}")

    total_sig = stats["sig_mode_legacy"] + stats["sig_mode_ht"] + stats["sig_mode_vht"]
    if total_sig > 0:
        print(f"  Frames Legacy:     {stats['sig_mode_legacy']} ({100*stats['sig_mode_legacy']/total_sig:.0f}%)")
        print(f"  Frames HT:         {stats['sig_mode_ht']} ({100*stats['sig_mode_ht']/total_sig:.0f}%)")
        if stats["sig_mode_vht"] > 0:
            print(f"  Frames VHT:        {stats['sig_mode_vht']} ({100*stats['sig_mode_vht']/total_sig:.0f}%)")

    if stats["csi_lines"] > 0:
        csi_duration = stats["last_csi_time"] - stats["first_csi_time"]
        hz = stats["csi_lines"] / csi_duration if csi_duration > 0 else 0
        avg_sc = sum(stats["subcarrier_counts"]) / len(stats["subcarrier_counts"])
        print(f"  Tasa CSI:          {hz:.1f} Hz")
        print(f"  Subportadoras:     {avg_sc:.0f} (promedio)")
    else:
        print(f"  Tasa CSI:          0 Hz (sin datos CSI)")

    if stats["error_lines"]:
        print(f"\n  Errores encontrados ({len(stats['error_lines'])}):")
        for err in stats["error_lines"][:10]:
            print(f"    - {err}")

    print(f"{'='*60}")

    if not stats["wifi_connected"] and stats["csi_lines"] == 0:
        print("\n  DIAGNOSTICO: El ESP32 no se conecto al WiFi.")
        print("  -> Verifica SSID y password con: idf.py menuconfig")
        print("  -> Verifica que el router esta encendido y en el canal correcto")
    elif stats["wifi_connected"] and stats["csi_lines"] == 0:
        print("\n  DIAGNOSTICO: WiFi conectado pero sin datos CSI.")
        print("  -> Posible filtro demasiado estricto (sig_mode >= 1)")
        print("  -> Intenta reducir PING_INTERVAL_MS en el firmware")
    elif stats["sendto_errors"] > stats["csi_lines"]:
        print("\n  DIAGNOSTICO: Muchos errores sendto (falta de memoria).")
        print("  -> Reduce buffers WiFi en sdkconfig.defaults")
    elif stats["csi_lines"] > 0:
        print("\n  DIAGNOSTICO: Todo parece funcionar correctamente.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Wi-Fi Vision 3D - Serial Diagnostic Tool"
    )
    parser.add_argument("--port", required=True, help="Serial port (e.g., COM3)")
    parser.add_argument("--baud", type=int, default=0,
                        help="Baud rate (default: auto-detect)")
    parser.add_argument("--duration", type=float, default=15,
                        help="Diagnostic duration in seconds (default: 15)")
    args = parser.parse_args()

    if args.baud == 0:
        print(f"Auto-detectando baud rate en {args.port}...")
        detected = auto_detect_baud(args.port)
        if detected is None:
            print(f"\nERROR: No se detecto baud rate valido en {args.port}.")
            print("Verifica que el ESP32-S3 esta conectado y flasheado.")
            sys.exit(1)
        baud = detected
    else:
        baud = args.baud

    run_diagnostic(args.port, baud, args.duration)


if __name__ == "__main__":
    main()
