/*
 * Wi-Fi Vision 3D - CSI Capture Firmware for ESP32-S3
 *
 * Extracts Channel State Information (CSI) from Wi-Fi packets received
 * from a dedicated 2.4GHz router (802.11n, 40MHz HT40, fixed channel).
 * Outputs amplitude+phase data for all subcarriers via USB serial in CSV format.
 *
 * Architecture: CSI callback -> FreeRTOS queue -> serial output task
 * This decoupling prevents the serial I/O from blocking the WiFi stack.
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/queue.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "driver/uart.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"
#include "ping/ping_sock.h"

/* ── Configuration ─────────────────────────────────────────────────────────── */

#define WIFI_SSID         CONFIG_CSI_WIFI_SSID
#define WIFI_PASSWORD     CONFIG_CSI_WIFI_PASSWORD
#define WIFI_CHANNEL      CONFIG_CSI_WIFI_CHANNEL

#define PING_INTERVAL_MS  10
#define PING_TARGET_IP    "0.0.0.0"

#define MAX_CSI_BYTES     228   /* 114 subcarriers x 2 (imag+real) */
#define CSI_LINE_SIZE     1400  /* max formatted CSV line length */
#define CSI_QUEUE_DEPTH   32    /* frames buffered between callback and output */

static const char *TAG = "CSI_CAPTURE";

/* ── Event group for Wi-Fi connection synchronization ──────────────────────── */

static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1

static int s_retry_num = 0;
#define MAX_RETRY 10

static char s_gateway_ip[16] = {0};
static volatile uint32_t s_csi_frame_count = 0;
static volatile uint32_t s_csi_drop_count = 0;

/* ── CSI Queue ─────────────────────────────────────────────────────────────── */

typedef struct {
    int len;
    char data[CSI_LINE_SIZE];
} csi_line_t;

static QueueHandle_t s_csi_queue = NULL;

/* ── Wi-Fi Event Handler ───────────────────────────────────────────────────── */

static void event_handler(void *arg, esp_event_base_t event_base,
                          int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_num < MAX_RETRY) {
            esp_wifi_connect();
            s_retry_num++;
            ESP_LOGW(TAG, "Reconnecting... attempt %d/%d", s_retry_num, MAX_RETRY);
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        snprintf(s_gateway_ip, sizeof(s_gateway_ip), IPSTR, IP2STR(&event->ip_info.gw));
        ESP_LOGI(TAG, "Connected. IP: " IPSTR ", Gateway: %s",
                 IP2STR(&event->ip_info.ip), s_gateway_ip);
        s_retry_num = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

/* ── CSI Callback (non-blocking: formats and enqueues) ─────────────────────── */

static void IRAM_ATTR wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf || !s_csi_queue) {
        return;
    }

    wifi_pkt_rx_ctrl_t *rx = &info->rx_ctrl;
    int64_t timestamp = esp_timer_get_time();

    int start_idx = info->first_word_invalid ? 4 : 0;
    int end_idx = info->len;
    if (end_idx > MAX_CSI_BYTES) {
        end_idx = MAX_CSI_BYTES;
    }
    int output_len = end_idx - start_idx;

    csi_line_t line;
    int pos = snprintf(line.data, CSI_LINE_SIZE,
        "CSI_DATA,%" PRId64 ",%02x:%02x:%02x:%02x:%02x:%02x,%d,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
        timestamp,
        info->mac[0], info->mac[1], info->mac[2],
        info->mac[3], info->mac[4], info->mac[5],
        rx->rssi, rx->rate, rx->sig_mode, rx->mcs, rx->cwb,
        rx->smoothing, rx->not_sounding, rx->aggregation,
        rx->stbc, rx->fec_coding, rx->sgi,
        rx->channel, rx->secondary_channel,
        info->rx_seq, output_len, info->first_word_invalid);

    for (int i = start_idx; i < end_idx && pos < CSI_LINE_SIZE - 6; i++) {
        pos += snprintf(line.data + pos, CSI_LINE_SIZE - pos, ",%d", info->buf[i]);
    }

    line.data[pos++] = '\n';
    line.data[pos] = '\0';
    line.len = pos;

    s_csi_frame_count++;

    if (xQueueSendFromISR(s_csi_queue, &line, NULL) != pdTRUE) {
        s_csi_drop_count++;
    }
}

/* ── Serial Output Task (dedicated, handles blocking I/O) ──────────────────── */

static void csi_output_task(void *pvParameters)
{
    csi_line_t line;
    ESP_LOGI(TAG, "CSI output task started (queue depth=%d)", CSI_QUEUE_DEPTH);

    while (1) {
        if (xQueueReceive(s_csi_queue, &line, portMAX_DELAY) == pdTRUE) {
            fwrite(line.data, 1, line.len, stdout);
        }
    }
}

/* ── Wi-Fi Initialization ──────────────────────────────────────────────────── */

static void wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                    ESP_EVENT_ANY_ID, &event_handler, NULL, &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                    IP_EVENT_STA_GOT_IP, &event_handler, NULL, &instance_got_ip));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = WIFI_SSID,
            .password = WIFI_PASSWORD,
            .channel = WIFI_CHANNEL,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT40));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    ESP_LOGI(TAG, "Waiting for Wi-Fi connection to '%s' on channel %d...", WIFI_SSID, WIFI_CHANNEL);

    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
                                           WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
                                           pdFALSE, pdFALSE, portMAX_DELAY);

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Wi-Fi connected successfully");
    } else if (bits & WIFI_FAIL_BIT) {
        ESP_LOGE(TAG, "Failed to connect to Wi-Fi after %d attempts", MAX_RETRY);
    }
}

/* ── CSI Configuration ─────────────────────────────────────────────────────── */

static void promiscuous_rx_cb(void *buf, wifi_promiscuous_pkt_type_t type)
{
    /* Required for promiscuous mode; CSI extraction happens separately */
    (void)buf;
    (void)type;
}

static void csi_init(void)
{
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous_rx_cb(promiscuous_rx_cb));
    ESP_LOGI(TAG, "Promiscuous mode enabled for full CSI capture");

    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = true,
        .stbc_htltf2_en    = true,
        .ltf_merge_en      = true,
        .channel_filter_en = false,
        .manu_scale        = false,
        .shift             = 0,
        .dump_ack_en       = true,
    };

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(TAG, "CSI capture enabled: lltf=%d htltf=%d stbc=%d filter=%d dump_ack=%d",
             csi_config.lltf_en, csi_config.htltf_en, csi_config.stbc_htltf2_en,
             csi_config.channel_filter_en, csi_config.dump_ack_en);
}

/* ── Traffic Generator (ICMP ping to gateway for real responses) ────────────── */

static void start_icmp_ping(void)
{
    ip_addr_t target;
    ipaddr_aton(s_gateway_ip, &target);

    esp_ping_config_t ping_config = ESP_PING_DEFAULT_CONFIG();
    ping_config.target_addr = target;
    ping_config.count = ESP_PING_COUNT_INFINITE;
    ping_config.interval_ms = PING_INTERVAL_MS;
    ping_config.timeout_ms = 100;
    ping_config.data_size = 1;
    ping_config.task_stack_size = 3072;
    ping_config.task_prio = 5;

    esp_ping_callbacks_t cbs = { 0 };
    esp_ping_handle_t ping_handle = NULL;

    esp_err_t err = esp_ping_new_session(&ping_config, &cbs, &ping_handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create ICMP ping session: %s", esp_err_to_name(err));
        return;
    }

    esp_ping_start(ping_handle);
    ESP_LOGI(TAG, "ICMP ping started -> %s every %d ms (1-byte payload)", s_gateway_ip, PING_INTERVAL_MS);
}

/* ── Status Monitor Task ───────────────────────────────────────────────────── */

static void status_task(void *pvParameters)
{
    uint32_t last_count = 0;
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
        uint32_t current = s_csi_frame_count;
        uint32_t hz_10s = (current - last_count);
        last_count = current;
        ESP_LOGI(TAG, "[STATUS] CSI: %lu total (%lu/10s = %.1f Hz) | Drops: %lu | Heap: %lu",
                 (unsigned long)current,
                 (unsigned long)hz_10s,
                 hz_10s / 10.0f,
                 (unsigned long)s_csi_drop_count,
                 (unsigned long)esp_get_minimum_free_heap_size());
    }
}

/* ── Main Entry Point ──────────────────────────────────────────────────────── */

#define CSI_BAUD_RATE 921600

void app_main(void)
{
    uart_set_baudrate(CONFIG_ESP_CONSOLE_UART_NUM, CSI_BAUD_RATE);

    ESP_LOGI(TAG, "==============================================");
    ESP_LOGI(TAG, "  Wi-Fi Vision 3D - CSI Capture Node");
    ESP_LOGI(TAG, "  UART forced to %d baud", CSI_BAUD_RATE);
    ESP_LOGI(TAG, "  Target: ESP32-S3 | Band: 2.4GHz HT40");
    ESP_LOGI(TAG, "  Queue: %d slots x %d bytes", CSI_QUEUE_DEPTH, CSI_LINE_SIZE);
    ESP_LOGI(TAG, "==============================================");

    s_csi_queue = xQueueCreate(CSI_QUEUE_DEPTH, sizeof(csi_line_t));
    if (!s_csi_queue) {
        ESP_LOGE(TAG, "Failed to create CSI queue!");
        return;
    }

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    wifi_init_sta();
    csi_init();

    xTaskCreate(&csi_output_task, "csi_out", 4096, NULL, 6, NULL);
    start_icmp_ping();
    xTaskCreate(&status_task, "status_task", 2048, NULL, 3, NULL);
}
