# wifi-csi-capture

Firmware and tools for extracting Wi-Fi Channel State Information (CSI) from ESP32-S3 microcontrollers. Captures amplitude and phase data across 114 OFDM subcarriers (HT40, 40MHz) and exports it as structured CSV for downstream analysis, digital twin calibration, or AI training pipelines.

## What is CSI?

Modern Wi-Fi (802.11n/ac/ax) uses Orthogonal Frequency-Division Multiplexing (OFDM), splitting the channel into many narrow subcarriers. CSI exposes the **amplitude and phase** of each subcarrier at the physical layer. When a person moves through the RF field, their body (mostly water) reflects and absorbs microwaves, altering the multipath propagation pattern. These perturbations are encoded in the CSI matrix and can be used to detect presence, movement, and even body pose -- all without cameras.

## What this project does

```
  Dedicated Router          ESP32-S3 nodes             PC
  ┌──────────────┐       ┌────────────────┐       ┌──────────────┐
  │  2.4GHz Tx    │~WiFi~>│  CSI Extraction │~USB~>│  Capture &    │
  │  802.11n HT40 │       │  114 subcarriers│       │  Validation   │
  │  Fixed channel│       │  CSV over serial│       │  CSV export   │
  └──────────────┘       └────────────────┘       └──────────────┘
```

1. **Firmware** flashes onto ESP32-S3 devices, connects to a dedicated router, and streams raw CSI frames over USB serial.
2. **Python tools** capture, visualize, analyze, and export the CSI data from one or more nodes simultaneously.
3. **Sequential protocol** allows mapping an environment from 8 receiver positions using only 2 physical devices (4 rounds of capture).

The exported data (CSV + JSON baseline) is designed to feed into a separate digital twin / AI training project.

## Hardware requirements

- **ESP32-S3 dev board** (x2 minimum, up to 8) -- any board with USB serial
- **Dedicated Wi-Fi router** -- configured for 2.4GHz only, 802.11n, 40MHz bandwidth, fixed channel. Must be isolated (no internet needed).
- **USB cables** -- one per ESP32-S3, for power and data

## Software requirements

- [ESP-IDF v5.5.3](https://docs.espressif.com/projects/esp-idf/en/v5.5.3/esp32s3/get-started/index.html)
- Python 3.11+
- Dependencies: `pip install -r tools/requirements.txt`

## Quick start

```bash
# 1. Set your router credentials
idf.py menuconfig
# -> CSI Capture Configuration -> WiFi SSID, Password, Channel

# 2. Build
idf.py build

# 3. Flash an ESP32-S3 and open serial monitor
idf.py -p COM3 flash monitor
# You should see CSI_DATA lines streaming

# 4. Install Python tools
pip install -r tools/requirements.txt

# 5. Visualize CSI in real time
python tools/visualize_csi.py --port COM3

# 6. Capture data from 2 nodes for 5 minutes
python tools/capture_csi.py --port1 COM3 --port2 COM4 --position 1 --duration 300
```

## Project structure

```
├── CMakeLists.txt                  ESP-IDF project root
├── sdkconfig.defaults              Default config (CSI enabled, 921600 baud)
├── main/
│   ├── CMakeLists.txt              Component dependencies
│   ├── csi_capture_main.c          Firmware: WiFi STA + CSI extraction + UDP traffic gen
│   └── Kconfig.projbuild           Configurable SSID, password, channel
└── tools/
    ├── requirements.txt            Python dependencies (pyserial, matplotlib, numpy)
    ├── capture_csi.py              Serial capture to CSV (1-2 nodes, threaded)
    ├── visualize_csi.py            Real-time amplitude, phase, spectrogram plots
    ├── measurement_protocol.py     4-round sequential capture across 8 positions
    ├── analyze_csi.py              Statistics, empty-vs-person comparison, baseline export
    └── digital_twin_sionna.py      Scene config for NVIDIA Sionna ray tracing
```

## CSI output format

Each line from the ESP32-S3 serial port:

```
CSI_DATA,<timestamp_us>,<mac>,<rssi>,<rate>,<sig_mode>,<mcs>,<cwb>,<smoothing>,
  <not_sounding>,<aggregation>,<stbc>,<fec>,<sgi>,<channel>,<secondary_ch>,
  <rx_seq>,<len>,<first_word_invalid>,<imag0>,<real0>,<imag1>,<real1>,...
```

In HT40 mode: 114 subcarriers = 228 int8 values (imaginary, real pairs).

## Configuration

All Wi-Fi settings are configurable via `idf.py menuconfig` (stored in `sdkconfig`, which is gitignored). The file `sdkconfig.defaults` contains safe defaults:

| Setting | Default | Purpose |
|---------|---------|---------|
| `CONFIG_ESP_WIFI_CSI_ENABLED` | `y` | Enable CSI subsystem |
| `CONFIG_ESP_CONSOLE_UART_BAUDRATE` | `921600` | High baud rate for CSI throughput |
| `CONFIG_FREERTOS_HZ` | `1000` | 1ms tick resolution for precise ping timing |

## License

MIT
