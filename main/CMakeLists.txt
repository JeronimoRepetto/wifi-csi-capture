idf_component_register(SRCS "csi_capture_main.c"
                       PRIV_REQUIRES esp_wifi nvs_flash esp_netif esp_timer
                       INCLUDE_DIRS "")
