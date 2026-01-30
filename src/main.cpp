#include "mbed.h"
#include "arm_math.h"
#include <math.h>
#include <string.h>

#include "ble/BLE.h"
#include "ble/gatt/GattService.h"
#include "ble/gatt/GattCharacteristic.h"
#include "ble/Gap.h"
#include "ble/gap/AdvertisingDataBuilder.h"
#include "events/EventQueue.h"

using namespace ble;
using namespace events;

/* Serial for printf (PlatformIO monitor / Teleplot) */
BufferedSerial serial_port(USBTX, USBRX, 115200);
FileHandle *mbed::mbed_override_console(int) { return &serial_port; }

/* BLE globals */
BLE &ble_interface = BLE::Instance();
EventQueue event_queue;
/* BLE callbacks sometimes allocate on stack; giving this thread more stack avoids HardFault */
Thread ble_thread(osPriorityNormal, 4096);

/* Custom BLE UUIDs for state service + characteristic */
const UUID STATE_SERVICE_UUID("A0E1B2C3-D4E5-F6A7-B8C9-D0E1F2A3B4C5");
const UUID STATE_CHAR_UUID   ("A1E2B3C4-D5E6-F7A8-B9C0-D1E2F3A4B5C6");

/* BLE string payload buffer (must fit the longest state string + '\0') */
#define MAX_STATE_LEN 16
uint8_t stateValue[MAX_STATE_LEN] = {0};

/* Expose state via READ + NOTIFY */
ReadOnlyArrayGattCharacteristic<uint8_t, MAX_STATE_LEN> stateCharacteristic(
    STATE_CHAR_UUID,
    stateValue,
    GattCharacteristic::BLE_GATT_CHAR_PROPERTIES_NOTIFY |
    GattCharacteristic::BLE_GATT_CHAR_PROPERTIES_READ
);

GattCharacteristic *charTable[] = { &stateCharacteristic };
GattService stateService(STATE_SERVICE_UUID, charTable, 1);

/* IMU I2C */
I2C i2c(PB_11, PB_10);

/* Data-ready interrupt from LSM6DSL INT1 */
InterruptIn int1(PD_11, PullDown);
volatile bool sample_ready = false;
void imu_drdy_isr() { sample_ready = true; }

/* LEDs: use them as quick on-device debug indicators */
DigitalOut led1(LED1);
DigitalOut led2(LED2);
DigitalOut led3(LED3);

/* LSM6DSL registers */
#define LSM6DSL_ADDR   (0x6A << 1)
#define WHO_AM_I       0x0F
#define CTRL1_XL       0x10
#define CTRL2_G        0x11
#define CTRL3_C        0x12
#define DRDY_PULSE_CFG 0x0B
#define INT1_CTRL      0x0D
#define STATUS_REG     0x1E
#define OUTX_L_G       0x22
#define OUTX_L_XL      0x28

/* 3s window at ~52Hz */
#define TARGET_FS_HZ    52.0f
#define WINDOW_SAMPLES  156
#define FFT_SIZE        512   /* power-of-2, with zero padding */

/* Window buffers store magnitudes (|a| and |g|) */
static float win_a_mag[WINDOW_SAMPLES];
static float win_g_mag[WINDOW_SAMPLES];
static int   win_idx = 0;

/* FFT buffers */
static float fft_input[FFT_SIZE];
static float fft_output[FFT_SIZE];
static float fft_mag[FFT_SIZE / 2];

arm_rfft_fast_instance_f32 S;

/* States: tremor and dyskinesia are both split into mild/severe */
enum State {
    NORMAL = 0,
    TREMOR_MINOR,
    TREMOR_SEVERE,
    DYSK_MILD,
    DYSK_SEVERE,
    WALK,
    FOG
};

volatile State current_state = NORMAL;
static State prev_state = NORMAL;          /* previous 3s decision, used for FOG rule */
volatile float fs_est = TARGET_FS_HZ;      /* measured sampling rate */

/* Tremor detection: require energy in 3–7Hz, clear peak, and not too large amplitude */
const float PEAK_MIN_MAG        = 1.5f;
const float P_TREMOR_MIN        = 150.0f;
const float PEAK_RATIO_MIN      = 2.5f;
const float ACC_RMS_TREMOR_MAX  = 0.25f;

/* Tremor severity: if any of these is large, treat as severe */
const float P_TREMOR_SEVERE     = 5000.0f;
const float ACC_RMS_TREMOR_SEV  = 0.25f;
const float PEAK_MAG_SEVERE     = 15.0f;

/* Walk: low-frequency (0.5–3Hz) dominates and is strong enough */
const float P_WALK_MIN          = 400.0f;
const float WALK_DOMINANCE      = 1.8f;

/* Dyskinesia: high amplitude + high jerk, or high rotation */
const float ACC_RMS_DYSK        = 0.4f;
const float JERK_RMS_DYSK       = 5.0f;
const float GYRO_RMS_DYSK       = 120.0f;

/* Dyskinesia severity: higher thresholds than the dysk detection */
const float ACC_RMS_DYSK_SEV    = 0.85f;
const float JERK_RMS_DYSK_SEV   = 20.0f;
const float GYRO_RMS_DYSK_SEV   = 400.0f;

/* RMS helper: used for acc_rms, jerk_rms, gyro_rms */
static inline float rms_of(const float *x, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (double)x[i] * (double)x[i];
    return (n > 0) ? sqrtf((float)(s / (double)n)) : 0.0f;
}

/* LED mapping: NORMAL uses LED1; mild uses LED2; severe/walk/fog uses LED3 */
static inline void set_leds(State s) {
    led1 = (s == NORMAL);
    led2 = (s == TREMOR_MINOR || s == DYSK_MILD);
    led3 = (s == TREMOR_SEVERE || s == DYSK_SEVERE || s == WALK || s == FOG);
}

/* I2C register helpers */
static inline void write_reg(uint8_t reg, uint8_t val) {
    char data[2] = {(char)reg, (char)val};
    i2c.write(LSM6DSL_ADDR, data, 2);
}

static inline bool read_reg(uint8_t reg, uint8_t &val) {
    char r = (char)reg;
    if (i2c.write(LSM6DSL_ADDR, &r, 1, true) != 0) return false;
    if (i2c.read(LSM6DSL_ADDR, &r, 1) != 0) return false;
    val = (uint8_t)r;
    return true;
}

static inline int16_t read_int16(uint8_t reg_low) {
    uint8_t lo, hi;
    read_reg(reg_low, lo);
    read_reg(reg_low + 1, hi);
    return (int16_t)((hi << 8) | lo);
}

/* Configure LSM6DSL: accel 52Hz ±2g + gyro 52Hz ±250dps, data-ready on INT1 */
bool init_sensor() {
    uint8_t who = 0;
    if (!read_reg(WHO_AM_I, who) || who != 0x6A) return false;

    write_reg(CTRL3_C, 0x44);        /* BDU + auto-increment */

    /* CTRL1_XL: ODR_XL=52Hz (0b0011), FS_XL=±2g (0b00) => 0x30 */
    write_reg(CTRL1_XL, 0x30);

    /* CTRL2_G: ODR_G=52Hz (0b0011), FS_G=±250dps (0b00) => 0x30 */
    write_reg(CTRL2_G,  0x30);

    write_reg(INT1_CTRL, 0x03);      /* XLDA + GDA -> INT1 */
    write_reg(DRDY_PULSE_CFG, 0x80); /* pulsed DRDY */

    uint8_t dummy;
    read_reg(STATUS_REG, dummy);
    ThisThread::sleep_for(100ms);
    return true;
}

/* FFT bin helpers use measured fs_est */
static inline float bin_size_hz() { return fs_est / (float)FFT_SIZE; }

static inline int hz_to_bin_floor(float hz) {
    float bs = bin_size_hz();
    int b = (int)floorf(hz / bs);
    if (b < 1) b = 1;
    if (b >= (FFT_SIZE / 2)) b = (FFT_SIZE / 2) - 1;
    return b;
}

static inline int hz_to_bin_ceil(float hz) {
    float bs = bin_size_hz();
    int b = (int)ceilf(hz / bs);
    if (b < 1) b = 1;
    if (b >= (FFT_SIZE / 2)) b = (FFT_SIZE / 2) - 1;
    return b;
}

/* Convert state enum to BLE string */
static inline const char* state_to_string(State s) {
    switch (s) {
        case NORMAL:        return "NORMAL";
        case TREMOR_MINOR:  return "TREMOR_MINOR";
        case TREMOR_SEVERE: return "TREMOR_SEVERE";
        case DYSK_MILD:     return "DYSK_MILD";
        case DYSK_SEVERE:   return "DYSK_SEVERE";
        case WALK:          return "WALK";
        case FOG:           return "FOG";
        default:            return "UNKNOWN";
    }
}

/* Send current state over BLE as a null-terminated string */
void ble_send_state(State s) {
    const char *msg = state_to_string(s);
    memset(stateValue, 0, MAX_STATE_LEN);
    strncpy((char*)stateValue, msg, MAX_STATE_LEN - 1);

    ble_interface.gattServer().write(
        stateCharacteristic.getValueHandle(),
        stateValue,
        strlen((char*)stateValue) + 1
    );

    printf("[BLE] notify: %s\n", (char*)stateValue);
}

/* BLE event dispatch hook */
void schedule_ble_events(BLE::OnEventsToProcessCallbackContext*) {
    event_queue.call(callback(&ble_interface, &BLE::processEvents));
}

/* BLE init callback: add service and start advertising */
void on_ble_init_complete(BLE::InitializationCompleteCallbackContext *params) {
    if (params->error != BLE_ERROR_NONE) {
        printf("BLE initialization failed.\n");
        return;
    }

    memset(stateValue, 0, MAX_STATE_LEN);
    strncpy((char*)stateValue, "NORMAL", MAX_STATE_LEN - 1);

    ble_interface.gattServer().addService(stateService);

    uint8_t adv_buffer[LEGACY_ADVERTISING_MAX_SIZE];
    AdvertisingDataBuilder adv_data(adv_buffer);

    adv_data.setFlags();
    adv_data.setName("RTES-Monitor");

    ble_interface.gap().setAdvertisingParameters(
        LEGACY_ADVERTISING_HANDLE,
        AdvertisingParameters(
            advertising_type_t::CONNECTABLE_UNDIRECTED,
            adv_interval_t(160) /* 100ms */
        )
    );

    ble_interface.gap().setAdvertisingPayload(
        LEGACY_ADVERTISING_HANDLE,
        adv_data.getAdvertisingData()
    );

    ble_interface.gap().startAdvertising(LEGACY_ADVERTISING_HANDLE);

    printf("BLE advertising started as RTES-Monitor\n");

    ble_send_state(NORMAL);
}

int main() {
    led1 = 0; led2 = 0; led3 = 0;
    i2c.frequency(400000);

    printf("Start: 3s FFT @52Hz + WALK/DYSK + Tremor/Dysk severity + FOG + BLE notify\n");

    ble_interface.onEventsToProcess(schedule_ble_events);
    ble_interface.init(on_ble_init_complete);
    ble_thread.start(callback(&event_queue, &EventQueue::dispatch_forever));

    arm_rfft_fast_init_f32(&S, FFT_SIZE);

    if (!init_sensor()) {
        printf("Sensor init failed.\n");
        while (1) ThisThread::sleep_for(1s);
    }

    int1.rise(&imu_drdy_isr);

    Timer ts;
    ts.start();
    int samples_per_sec = 0;

    State last_sent_state = NORMAL;

    while (true) {
        if (!sample_ready) {
            ThisThread::sleep_for(1ms);
            continue;
        }
        sample_ready = false;

        /* Read accel in g (±2g => 0.061 mg/LSB = 0.000061 g/LSB) */
        int16_t ax_raw = read_int16(OUTX_L_XL);
        int16_t ay_raw = read_int16(OUTX_L_XL + 2);
        int16_t az_raw = read_int16(OUTX_L_XL + 4);

        float ax = ax_raw * 0.000061f;
        float ay = ay_raw * 0.000061f;
        float az = az_raw * 0.000061f;
        float a_mag = sqrtf(ax*ax + ay*ay + az*az);

        /* Read gyro in dps (±250 dps => 8.75 mdps/LSB = 0.00875 dps/LSB) */
        int16_t gx_raw = read_int16(OUTX_L_G);
        int16_t gy_raw = read_int16(OUTX_L_G + 2);
        int16_t gz_raw = read_int16(OUTX_L_G + 4);

        float gx = gx_raw * 0.00875f;
        float gy = gy_raw * 0.00875f;
        float gz = gz_raw * 0.00875f;
        float g_mag = sqrtf(gx*gx + gy*gy + gz*gz);

        /* Fill the 3-second window */
        if (win_idx < WINDOW_SAMPLES) {
            win_a_mag[win_idx] = a_mag;
            win_g_mag[win_idx] = g_mag;
            win_idx++;
        }

        /* Estimate Fs once per second */
        samples_per_sec++;
        if (ts.elapsed_time() >= 1s) {
            fs_est = (float)samples_per_sec;
            printf("samples_per_sec=%d (Fs_est=%.1f)\n", samples_per_sec, (double)fs_est);
            samples_per_sec = 0;
            ts.reset();
        }

        if (win_idx < WINDOW_SAMPLES) continue;

        /* Remove DC by subtracting window mean */
        double mean_a = 0.0, mean_g = 0.0;
        for (int i = 0; i < WINDOW_SAMPLES; i++) {
            mean_a += win_a_mag[i];
            mean_g += win_g_mag[i];
        }
        mean_a /= (double)WINDOW_SAMPLES;
        mean_g /= (double)WINDOW_SAMPLES;

        static float a_det[WINDOW_SAMPLES];
        static float g_det[WINDOW_SAMPLES];
        for (int i = 0; i < WINDOW_SAMPLES; i++) {
            a_det[i] = (float)(win_a_mag[i] - mean_a);
            g_det[i] = (float)(win_g_mag[i] - mean_g);
        }

        /* Time-domain features */
        float acc_rms = rms_of(a_det, WINDOW_SAMPLES);

        static float jerk[WINDOW_SAMPLES];
        float dt = 1.0f / fmaxf(fs_est, 1.0f);
        jerk[0] = 0.0f;
        for (int i = 1; i < WINDOW_SAMPLES; i++) {
            jerk[i] = (a_det[i] - a_det[i - 1]) / dt;
        }
        float jerk_rms = rms_of(jerk, WINDOW_SAMPLES);
        float gyro_rms = rms_of(g_det, WINDOW_SAMPLES);

        /* FFT on detrended |a|, zero-pad to FFT_SIZE */
        for (int i = 0; i < WINDOW_SAMPLES; i++) fft_input[i] = a_det[i];
        for (int i = WINDOW_SAMPLES; i < FFT_SIZE; i++) fft_input[i] = 0.0f;

        arm_rfft_fast_f32(&S, fft_input, fft_output, 0);
        arm_cmplx_mag_f32(fft_output, fft_mag, FFT_SIZE / 2);

        /* Band energy: walk (0.5–3Hz) vs tremor (3–7Hz) */
        int b05 = hz_to_bin_floor(0.5f);
        int b3  = hz_to_bin_floor(3.0f);
        int b7  = hz_to_bin_ceil(7.0f);

        float P_walk = 0.0f;
        float P_trem = 0.0f;
        for (int i = b05; i < b3; i++)  { float m = fft_mag[i]; P_walk += m * m; }
        for (int i = b3;  i <= b7; i++) { float m = fft_mag[i]; P_trem += m * m; }

        /* Find peak inside 3–7Hz, also compute peak-to-mean sharpness */
        float peak_val = 0.0f;
        int   peak_bin = b3;
        double band_mean = 0.0;
        int count = 0;

        for (int i = b3; i <= b7; i++) {
            float m = fft_mag[i];
            band_mean += m;
            count++;
            if (m > peak_val) {
                peak_val = m;
                peak_bin = i;
            }
        }
        band_mean = (count > 0) ? (band_mean / (double)count) : 0.0;
        float peak_ratio = peak_val / (float)(band_mean + 1e-6f);
        float freq_hz = peak_bin * bin_size_hz();

        /* Main decision: Walk -> Dysk -> Tremor -> Normal */
        State s = NORMAL;

        bool is_walk = (P_walk > P_WALK_MIN) && (P_walk > WALK_DOMINANCE * P_trem);
        if (is_walk) {
            s = WALK;
        } else {
            bool dysk_by_acc  = (acc_rms > ACC_RMS_DYSK) && (jerk_rms > JERK_RMS_DYSK);
            bool dysk_by_gyro = (gyro_rms > GYRO_RMS_DYSK);

            if (dysk_by_acc || dysk_by_gyro) {
                bool dysk_severe =
                    (acc_rms  > ACC_RMS_DYSK_SEV) ||
                    (jerk_rms > JERK_RMS_DYSK_SEV) ||
                    (gyro_rms > GYRO_RMS_DYSK_SEV);

                s = dysk_severe ? DYSK_SEVERE : DYSK_MILD;
            } else {
                bool trem_ok =
                    (P_trem > P_TREMOR_MIN) &&
                    (peak_val > PEAK_MIN_MAG) &&
                    (peak_ratio > PEAK_RATIO_MIN) &&
                    (acc_rms < ACC_RMS_TREMOR_MAX) &&
                    (freq_hz >= 3.0f && freq_hz <= 7.0f);

                if (trem_ok) {
                    bool trem_severe =
                        (P_trem  > P_TREMOR_SEVERE) ||
                        (acc_rms > ACC_RMS_TREMOR_SEV) ||
                        (peak_val > PEAK_MAG_SEVERE);

                    s = trem_severe ? TREMOR_SEVERE : TREMOR_MINOR;
                } else {
                    s = NORMAL;
                }
            }
        }

        /* FOG rule: if the previous 3s window was WALK and now becomes NORMAL, label as FOG */
        if (prev_state == WALK && s == NORMAL) {
            s = FOG;
        }

        current_state = s;
        prev_state = s;
        set_leds(current_state);

        /* Frequency output is meaningful only when tremor energy exists */
        float out_freq = 0.0f;
        if (P_trem > P_TREMOR_MIN && peak_val > PEAK_MIN_MAG) out_freq = freq_hz;

        /* Teleplot / debug prints (once per window) */
        printf(">Freq_Hz:%f\n", out_freq);
        printf(">P_walk_0_5_3:%f\n", P_walk);
        printf(">P_tremor_3_7:%f\n", P_trem);
        printf(">AccRMS:%f\n", acc_rms);
        printf(">JerkRMS:%f\n", jerk_rms);
        printf(">GyroRMS:%f\n", gyro_rms);
        printf(">PeakMag:%f\n", peak_val);
        printf(">State:%d\n", (int)current_state);
        printf("STATE: %s\n", state_to_string(current_state));

        /* BLE notify only when state actually changes */
        if (current_state != last_sent_state) {
            ble_send_state(current_state);
            last_sent_state = current_state;
        }

        win_idx = 0;
    }
}
