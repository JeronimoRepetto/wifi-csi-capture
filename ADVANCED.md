# Advanced Configuration & Performance Guide

This document covers firmware tuning, multi-node deployment, data management, and troubleshooting for the Wi-Fi Vision 3D CSI capture system.

## Firmware Configuration (menuconfig)

All firmware settings are managed via `idf.py menuconfig`. They are stored in `sdkconfig` (gitignored). The file `sdkconfig.defaults` provides safe baseline values that are applied on first build.

### CSI Capture Configuration menu

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `CSI_WIFI_SSID` | string | `YOUR_SSID` | SSID of the dedicated 2.4 GHz router. Must be a router you control, configured for 802.11n, 40 MHz bandwidth, fixed channel. |
| `CSI_WIFI_PASSWORD` | string | `YOUR_PASSWORD` | Password for the router's Wi-Fi network. |
| `CSI_WIFI_CHANNEL` | int (1-13) | `6` | Fixed channel matching the router. Must be identical on both sides. |
| `CSI_UART_BAUD` | int | `921600` | Serial baud rate. 921600 is the minimum for streaming 114-subcarrier CSI at 100 Hz without queue drops. |
| `CSI_FILTER_TARGET_MAC` | bool | `n` | Enable source-MAC filtering in the CSI callback. Strongly recommended for training data. |
| `CSI_TARGET_MAC` | string | (empty) | MAC address to accept (e.g., `d8:47:32:2e:4c:f9`). Only visible when MAC filtering is enabled. Use the router's BSSID. |

### sdkconfig.defaults (system-level)

These are applied automatically on the first `idf.py build`:

| Setting | Value | Purpose |
|---------|-------|---------|
| `CONFIG_ESP_WIFI_CSI_ENABLED` | `y` | Enable CSI extraction in the Wi-Fi driver |
| `CONFIG_ESP_WIFI_STATIC_RX_BUFFER_NUM` | `10` | Static RX buffers for Wi-Fi driver |
| `CONFIG_ESP_WIFI_DYNAMIC_RX_BUFFER_NUM` | `32` | Dynamic RX buffers — more buffers reduce drops under load |
| `CONFIG_ESP_WIFI_RX_BA_WIN` | `16` | Block Ack window size for 802.11n |
| `CONFIG_ESP_CONSOLE_UART_BAUDRATE` | `921600` | Console baud rate matching `CSI_UART_BAUD` |
| `CONFIG_ESP_MAIN_TASK_STACK_SIZE` | `4096` | Stack for `app_main()` initialization |
| `CONFIG_LOG_DEFAULT_LEVEL` | `3` (INFO) | Show status reports without flooding with DEBUG output |
| `CONFIG_ESP_WIFI_STA_DISCONNECTED_PM_ENABLE` | `n` | Disable Wi-Fi power save — radio stays on 100% of the time |
| `CONFIG_FREERTOS_HZ` | `1000` | 1 ms tick resolution for precise ICMP ping scheduling |

## Performance Characteristics

| Metric | Typical Value | Notes |
|--------|---------------|-------|
| Internal CSI rate | 100 Hz | Driven by ICMP ping interval (10 ms) |
| Serial throughput | 50-93 Hz | Bottleneck is UART at 921600 baud, depends on line length |
| Subcarriers per frame | 114 | HT40 mode (40 MHz bandwidth) |
| Queue capacity | 32 slots | FreeRTOS queue, zero drops under normal load |
| Free heap at runtime | ~212 KB | Leaves room for future features |
| CSV file size per minute | ~5-8 MB per node | 114 subcarriers x 2 values x ~50 Hz |

### Tuning serial throughput

The UART output at 921600 baud can carry approximately 90,000 characters/second. Each CSI line is ~800-1000 characters (18 header fields + 228 int8 values). At 100 Hz internal rate, the UART becomes the bottleneck around 90-100 lines/second.

To maximize throughput:
- Keep `CSI_UART_BAUD` at 921600 (maximum supported by ESP32-S3 USB-UART bridge)
- Use short USB cables (< 1 m) to minimize electrical noise
- Avoid running `idf.py monitor` simultaneously — it adds overhead

### ICMP ping interval

The firmware uses ICMP Echo Requests to trigger CSI extraction from the router's Echo Replies. The ping interval is set to 10 ms (100 Hz) in the firmware source. Modifying this requires changing `PING_INTERVAL_MS` in `main/csi_capture_main.c` and recompiling.

Lower intervals (e.g., 5 ms = 200 Hz) are theoretically possible but:
- Exceed UART throughput capacity at 921600 baud
- May cause queue drops
- Produce diminishing returns for motion detection

### AGC and gain stability

The ESP32-S3's Wi-Fi radio uses Automatic Gain Control (AGC) that can cause amplitude fluctuations unrelated to the environment. These show up as sudden baseline shifts in the amplitude plot.

Mitigation strategies:
- **Firmware-level gain lock**: Lock the AGC and FFT scaling registers after initial connection. This stabilizes amplitude measurements significantly. (Planned feature based on ESPectre research.)
- **CV normalization**: The activity indicator panel uses `CV = std(amplitudes) / mean(amplitudes)`, which is invariant to linear gain scaling. This already partially compensates for AGC fluctuations.
- **Baseline subtraction**: When analyzing session data, subtract the mean baseline to remove static offset.

## MAC Filtering

### Why it matters

Without MAC filtering, the CSI callback captures frames from **any** Wi-Fi transmitter in range (nearby routers, phones, IoT devices). Mixed-MAC data contaminates training datasets because different transmitters produce different amplitude/phase profiles that the AI would learn as environment features.

### Setup

1. Find the router's BSSID (MAC address):
   ```bash
   # From a device connected to the router's network:
   # Windows
   netsh wlan show interfaces | findstr BSSID
   # Linux
   iw dev wlan0 link | grep Connected
   # macOS
   /System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport -I | grep BSSID
   ```

2. Configure in firmware:
   ```bash
   idf.py menuconfig
   # CSI Capture Configuration → Filter CSI by target source MAC: [*]
   # CSI Capture Configuration → Target source MAC: d8:47:32:2e:4c:f9
   ```

3. Rebuild and flash:
   ```bash
   idf.py build && idf.py -p COM3 flash
   ```

4. Validate in Python (optional):
   ```bash
   python tools/record_session.py --expected-mac d8:47:32:2e:4c:f9 --duration 30s
   ```
   The recorder warns if captured CSVs contain unexpected MAC addresses.

## Multi-Node Deployment

### Recommended topology

Place 8 ESP32-S3 nodes around the target zone:
- 4 at ceiling height (~2.5 m), one per corner
- 4 at floor level (~0.3 m), one per corner
- Router centered on one wall, outside the zone
- All nodes outside the zone perimeter

This creates 8 independent Tx-Rx links that cross the zone from different angles. The spatial filter uses weighted consensus across all links to distinguish internal activity from external movement.

### Position assignment

Positions are defined in `tools/measurement_protocol.py`:

| Position | Location | Zone | Height |
|----------|----------|------|--------|
| pos01-pos04 | Ceiling corners | techo | ~2.5 m |
| pos05-pos08 | Floor corners | suelo | ~0.3 m |

During sequential capture (2 nodes at a time), use `--round` to assign position pairs:
```bash
# Round 1: positions 1,2
python tools/record_session.py --round 1 --duration 300 --scenario baseline_empty
# Round 2: positions 3,4
python tools/record_session.py --round 2 --duration 300 --scenario baseline_empty
# ... rounds 3 and 4
```

### Scaling to 8 parallel nodes

When all 8 ESP32-S3 are connected simultaneously:
```bash
# Auto-detects all 8 ports, captures in parallel
python tools/record_session.py --scenario baseline_empty --duration 300
```

USB hub requirements:
- Use a **powered** USB hub (8 nodes draw ~2A total)
- Ensure each port can sustain 921600 baud without buffering issues
- Test with `diagnose_serial.py` on each port first

## Data Storage

### Directory structure

```
<data_root>/
  sessions/
    20260314_162219_baseline_empty_smoke30_round1/
      raw/
        baseline_empty_pos01_node01_20260314_162224.csv
        baseline_empty_pos02_node02_20260314_162224.csv
      session_manifest.json
```

### Custom data root

By default, data is saved to `data/` (gitignored). To use a different drive:

```bash
# CLI argument
python tools/record_session.py --data-root "G:\dataset" --duration 300

# Interactive mode (just answer the prompt)
python tools/record_session.py --round 1
# Data root folder [data]: G:\dataset
```

### Session manifest

Each session produces a `session_manifest.json` with:
- Timestamp, scenario, duration, round number
- List of nodes with port, position, and output file path
- Dataset label and operator name
- MAC filtering results (expected vs. observed MACs)
- Capture mode (sequential 2-node or parallel N-node)

## Troubleshooting

### `internal compiler error: Segmentation fault` during build

The ESP-IDF toolchain (GCC for Xtensa) can crash with certain optimization levels.

Fix:
```bash
idf.py fullclean
# Disable ccache temporarily:
set IDF_CCACHE_ENABLE=0   # Windows
# Or change optimization:
idf.py menuconfig
# Compiler options → Optimization Level → -O2 or -Og
idf.py build
```

### No CSI data (0 frames in visualizer)

Check sequentially:
1. **Baud rate mismatch**: Ensure `CSI_UART_BAUD` in menuconfig matches `--baud` in the script (default: 921600)
2. **Wi-Fi not connected**: Check serial monitor for `WIFI_EVENT_STA_GOT_IP`. If missing, verify SSID/password/channel
3. **Wrong channel**: The firmware and router must use the exact same channel
4. **MAC filter too strict**: If `CSI_FILTER_TARGET_MAC` is enabled but the MAC is wrong, all frames are silently discarded. Check the status line in serial output: `accepted: 0, rejected: N`

### Ports not detected by record_session.py

The auto-detection scans for USB-serial devices with ESP32-S3 descriptors. If it fails:
```bash
# List all serial ports manually
python -m serial.tools.list_ports -v

# Specify ports explicitly
python tools/record_session.py --ports COM3,COM5 --positions 1,2
```

### CSV file appears empty or has very few lines

- Check that the ESP32-S3 was connected and streaming during the capture window
- Verify Wi-Fi connection by running `diagnose_serial.py --port COMx --duration 15` first
- Check free disk space on the target drive

### Visualizer shows blank plot for CSV files

Ensure the CSV was generated by `capture_csi.py` or `record_session.py`. The visualizer auto-detects the format by looking for the `timestamp_us` header. If the file has a non-standard format, it falls back to raw serial parsing, which may produce no matches.

## Visualizer Headless Mode

Generate screenshots without opening a GUI window:

```bash
python tools/visualize_csi.py --file "path/to/capture.csv" --save-fig output.png
python tools/visualize_csi.py --file "path/to/capture.csv" --save-fig output.png --save-after 150
```

The `--save-after` flag controls how many frames to render before saving (default: 80). More frames produce a richer spectrogram and smoother activity/RSSI trends.
