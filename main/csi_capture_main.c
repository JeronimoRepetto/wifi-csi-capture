/*
 * Wi-Fi Vision 3D - CSI Capture Firmware for ESP32-S3
 *
 * Extracts Channel State Information (CSI) from Wi-Fi packets received
 * from a dedicated 2.4GHz router (802.11n, 40MHz HT40, fixed channel).
 * Outputs amplitude+phase data for all subcarriers via USB serial in CSV format.
 *
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"

/* ── Configuration ─────────────────────────────────────────────────────────── */

#define WIFI_SSID         CONFIG_CSI_WIFI_SSID
#define WIFI_PASSWORD     CONFIG_CSI_WIFI_PASSWORD
#define WIFI_CHANNEL      CONFIG_CSI_WIFI_CHANNEL

#define PING_INTERVAL_MS  20
#define PING_TARGET_IP    "0.0.0.0"   /* resolved to gateway at runtime */

#define CSI_OUTPUT_TYPE_CSV  0
#define CSI_OUTPUT_TYPE_RAW  1
#define CSI_OUTPUT_TYPE      CSI_OUTPUT_TYPE_CSV

static const char *TAG = "CSI_CAPTURE";

/* ── Event group for Wi-Fi connection synchronization ──────────────────────── */

static EventGroupHandle_t s_wifi_event_group;
#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1

static int s_retry_num = 0;
#define MAX_RETRY 10

static char s_gateway_ip[16] = {0};
static uint32_t s_csi_frame_count = 0;

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

/* ── CSI Callback ──────────────────────────────────────────────────────────── */

/*
 * CSV format per line:
 * CSI_DATA,<timestamp_us>,<mac>,<rssi>,<rate>,<sig_mode>,<mcs>,<cwb>,<smoothing>,
 *   <not_sounding>,<aggregation>,<stbc>,<fec>,<sgi>,<channel>,<secondary_ch>,
 *   <rx_seq>,<len>,<first_word_invalid>,<data[0]>,<data[1]>,...<data[len-1]>
 *
 * Data bytes are int8 pairs: [imag0, real0, imag1, real1, ...] for each subcarrier.
 * For HT40 (40MHz): up to 114 subcarriers = 228 int8 values.
 */
static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) {
        return;
    }

    int64_t timestamp = esp_timer_get_time();
    wifi_pkt_rx_ctrl_t *rx = &info->rx_ctrl;

    s_csi_frame_count++;

    char mac_str[18];
    snprintf(mac_str, sizeof(mac_str), "%02x:%02x:%02x:%02x:%02x:%02x",
             info->mac[0], info->mac[1], info->mac[2],
             info->mac[3], info->mac[4], info->mac[5]);

    printf("CSI_DATA,%" PRId64 ",%s,%d,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u,%u",
           timestamp,
           mac_str,
           rx->rssi,
           rx->rate,
           rx->sig_mode,
           rx->mcs,
           rx->cwb,
           rx->smoothing,
           rx->not_sounding,
           rx->aggregation,
           rx->stbc,
           rx->fec_coding,
           rx->sgi,
           rx->channel,
           rx->secondary_channel,
           info->rx_seq,
           info->len,
           info->first_word_invalid);

    int start_idx = info->first_word_invalid ? 4 : 0;

    for (int i = start_idx; i < info->len; i++) {
        printf(",%d", info->buf[i]);
    }
    printf("\n");
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

    /* Force 40MHz bandwidth for maximum subcarrier resolution */
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT40));

    ESP_ERROR_CHECK(esp_wifi_start());

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

static void csi_init(void)
{
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = true,
        .stbc_htltf2_en    = true,
        .ltf_merge_en      = true,
        .channel_filter_en = false, /* raw subcarrier independence */
        .manu_scale        = false,
        .shift             = 0,
        .dump_ack_en       = true,  /* capture ACK frames for bidirectional CSI */
    };

    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(TAG, "CSI capture enabled: lltf=%d htltf=%d stbc=%d filter=%d dump_ack=%d",
             csi_config.lltf_en, csi_config.htltf_en, csi_config.stbc_htltf2_en,
             csi_config.channel_filter_en, csi_config.dump_ack_en);
}

/* ── Traffic Generator (UDP ping to gateway) ───────────────────────────────── */

static void ping_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Starting traffic generator -> %s every %d ms", s_gateway_ip, PING_INTERVAL_MS);

    struct sockaddr_in dest_addr;
    dest_addr.sin_addr.s_addr = inet_addr(s_gateway_ip);
    dest_addr.sin_family = AF_INET;
    dest_addr.sin_port = htons(5555);

    int sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (sock < 0) {
        ESP_LOGE(TAG, "Failed to create socket: errno %d", errno);
        vTaskDelete(NULL);
        return;
    }

    /* Minimal UDP payload to stimulate CSI extraction */
    const char payload[] = "CSI_PING";
    uint32_t ping_count = 0;

    while (1) {
        int err = sendto(sock, payload, sizeof(payload), 0,
                         (struct sockaddr *)&dest_addr, sizeof(dest_addr));
        if (err < 0) {
            ESP_LOGW(TAG, "sendto failed: errno %d", errno);
        }

        ping_count++;
        if (ping_count % 500 == 0) {
            ESP_LOGI(TAG, "Ping count: %lu, CSI frames: %lu",
                     (unsigned long)ping_count, (unsigned long)s_csi_frame_count);
        }

        vTaskDelay(pdMS_TO_TICKS(PING_INTERVAL_MS));
    }

    close(sock);
    vTaskDelete(NULL);
}

/* ── Status Monitor Task ───────────────────────────────────────────────────── */

static void status_task(void *pvParameters)
{
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
        ESP_LOGI(TAG, "[STATUS] CSI frames captured: %lu | Free heap: %lu bytes",
                 (unsigned long)s_csi_frame_count,
                 (unsigned long)esp_get_minimum_free_heap_size());
    }
}

/* ── Main Entry Point ──────────────────────────────────────────────────────── */

void app_main(void)
{
    ESP_LOGI(TAG, "==============================================");
    ESP_LOGI(TAG, "  Wi-Fi Vision 3D - CSI Capture Node");
    ESP_LOGI(TAG, "  Target: ESP32-S3 | Band: 2.4GHz HT40");
    ESP_LOGI(TAG, "==============================================");

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    wifi_init_sta();
    csi_init();

    xTaskCreate(&ping_task, "ping_task", 4096, NULL, 5, NULL);
    xTaskCreate(&status_task, "status_task", 2048, NULL, 3, NULL);
}
