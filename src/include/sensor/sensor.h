#pragma once

#include <stdint.h>
#include <i2c/smbus.h>
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include "../nn_math.h"


#define DEV_ID 0x76
#define DEV_PATH "/dev/i2c-1"
#define LOCAL_HASL 79.0

#define IDENT 0xD0
#define SOFT_RESET 0xE0
#define CTRL_HUM 0xF2
#define STATUS 0xF3
#define CTRL_MEAS 0xF4
#define CONFIG 0xF5

#define DATA_START_ADDR 0xF7
#define DATA_LENGTH 8

#define CAL_DATA0_START_ADDR 0x88
#define CAL_DATA0_LENGTH 25
#define CAL_DATA1_START_ADDR 0xE1
#define CAL_DATA1_LENGTH 7

#define G 9.80665
#define M 0.0289644
#define T 288.15
#define R 8.3144598

int32_t t_fine = 0;

uint16_t dig_T1 = 0;
int16_t dig_T1 = 0;
int16_t dig_T2 = 0;

uint16_t dig_P1 = 0;
int16_t dig_P2 = 0;
int16_t dig_P3 = 0;
int16_t dig_P4 = 0;
int16_t dig_P5 = 0;
int16_t dig_P6 = 0;
int16_t dig_P7 = 0;
int16_t dig_P8 = 0;
int16_t dig_P9 = 0;

uint8_t dig_H1 = 0;
int16_t dig_H2 = 0;
uint8_t dig_H3 = 0;
int16_t dig_H4 = 0;
int16_t dig_H5 = 0;
int8_t dig_H6 = 0;

int fd = 0;

double BME280_compensate_T_double(int32_t adc_T) {
    double var1, var2, T;
    var1 = (((double)adc_T) / 16384.0 - ((double)dig_T1) / 1024.0) * ((double)dig_T2);
    var2 = ((((double)adc_T) / 131072.0 - ((double)dig_T1) / 8192.0) *
            (((double)adc_T) / 131072.0 - ((double)dig_T1) / 8192.0)) *
           ((double)dig_T3);
    t_fine = (int32_t)(var1 + var2);
    T = (var1 + var2) / 5120.0;
    return T;
}

/* Returns pressure in Pa as double. Output value of “96386.2” equals 96386.2 Pa
 * = 963.862 hPa
 */
double BME280_compensate_P_double(int32_t adc_P) {
    double var1, var2, p;
    var1 = ((double)t_fine / 2.0) - 64000.0;
    var2 = var1 * var1 * ((double)dig_P6) / 32768.0;
    var2 = var2 + var1 * ((double)dig_P5) * 2.0;
    var2 = (var2 / 4.0) + (((double)dig_P4) * 65536.0);
    var1 = (((double)dig_P3) * var1 * var1 / 524288.0 + ((double)dig_P2) * var1) / 524288.0;
    var1 = (1.0 + var1 / 32768.0) * ((double)dig_P1);
    /* avoid exception caused by division by zero */
    if (var1 == 0.0) {
        return 0;
    }
    p = 1048576.0 - (double)adc_P;
    p = (p - (var2 / 4096.0)) * 6250.0 / var1;
    var1 = ((double)dig_P9) * p * p / 2147483648.0;
    var2 = p * ((double)dig_P8) / 32768.0;
    p = p + (var1 + var2 + ((double)dig_P7)) / 16.0;
    return p;
}

double BME280_compensate_H_double(int32_t adc_H) {
    double var_H;
    var_H = (((double)t_fine) - 76800.0);
    var_H = (adc_H - (((double)dig_H4) * 64.0 + ((double)dig_H5) / 16384.0 * var_H)) *
            (((double)dig_H2) / 65536.0 *
             (1.0 + ((double)dig_H6) / 67108864.0 * var_H *
                        (1.0 + ((double)dig_H3) / 67108864.0 * var_H)));
    var_H = var_H * (1.0 - ((double)dig_H1) * var_H / 524288.0);
    if (var_H > 100.0)
        var_H = 100.0;
    else if (var_H < 0.0)
        var_H = 0.0;
    return var_H;
}

/* Read calibration data and determine trimming parameters */
void setCompensationParams(int fd) {
    uint8_t calData0[25];
    uint8_t calData1[7];

    /* read calibration data */
    i2c_smbus_read_i2c_block_data(fd, CAL_DATA0_START_ADDR, CAL_DATA0_LENGTH, calData0);
    i2c_smbus_read_i2c_block_data(fd, CAL_DATA1_START_ADDR, CAL_DATA1_LENGTH, calData1);

    /* trimming parameters */
    dig_T1 = calData0[1] << 8 | calData0[0];
    dig_T2 = calData0[3] << 8 | calData0[2];
    dig_T3 = calData0[5] << 8 | calData0[4];

    dig_P1 = calData0[7] << 8 | calData0[6];
    dig_P2 = calData0[9] << 8 | calData0[8];
    dig_P3 = calData0[11] << 8 | calData0[10];
    dig_P4 = calData0[13] << 8 | calData0[12];
    dig_P5 = calData0[15] << 8 | calData0[14];
    dig_P6 = calData0[17] << 8 | calData0[16];
    dig_P7 = calData0[19] << 8 | calData0[18];
    dig_P8 = calData0[21] << 8 | calData0[20];
    dig_P9 = calData0[23] << 8 | calData0[22];

    dig_H1 = calData0[24];
    dig_H2 = calData1[1] << 8 | calData1[0];
    dig_H3 = calData1[2];
    dig_H4 = calData1[3] << 4 | (calData1[4] & 0xF);
    dig_H5 = calData1[5] << 4 | (calData1[4] >> 4);
    dig_H6 = calData1[6];
}

float sta2sea(float station_press)
{
    return station_press * fast_exp((-M * G * -LOCAL_HASL) / (R * T));
}

int sensor_setup()
{
    /* open i2c comms */
    if ((fd = open(DEV_PATH, O_RDWR)) < 0) {
        perror("Unable to open i2c device");
        return -1;
    }

    /* configure i2c slave */
    if (ioctl(fd, I2C_SLAVE, DEV_ID) < 0) {
        perror("Unable to configure i2c slave device");
        close(fd);
        return -2;
    }

    /* check our identification */
    if (i2c_smbus_read_byte_data(fd, IDENT) != 0x60) {
        perror("device ident error");
        close(fd);
        return -3;
    }

    /* device soft reset */
    i2c_smbus_write_byte_data(fd, SOFT_RESET, 0xB6);
    usleep(50000);

    /* read and set compensation parameters */
    setCompensationParams(fd);

    return 0;
}

int getWeatherInfo(float* vals) // float[3] array variable passed into argument
{
    uint8_t dataBlock[8];
    int32_t temp_int = 0;
    int32_t press_int = 0;
    int32_t hum_int = 0;
    double station_press = 0.0;

    /* humidity o/s x 1 */
    i2c_smbus_write_byte_data(fd, CTRL_HUM, 0x1);

    /* filter off */
    i2c_smbus_write_byte_data(fd, CONFIG, 0);

    /* set forced mode, pres o/s x 1, temp o/s x 1 and take 1st reading */
    i2c_smbus_write_byte_data(fd, CTRL_MEAS, 0x25);

    /* check data is ready to read */
    while ((i2c_smbus_read_byte_data(fd, STATUS) & 0x9) != 0) {
        printf("%s\n", "Error, data not ready");
        sleep(1);
        continue;
    }

    /* read data registers */
    i2c_smbus_read_i2c_block_data(fd, DATA_START_ADDR, DATA_LENGTH, dataBlock);

    /* awake and take next reading */
    i2c_smbus_write_byte_data(fd, CTRL_MEAS, 0x25);

    /* get raw temp */
    temp_int = (dataBlock[3] << 16 | dataBlock[4] << 8 | dataBlock[5]) >> 4;

    /* get raw pressure */
    press_int = (dataBlock[0] << 16 | dataBlock[1] << 8 | dataBlock[2]) >> 4;

    /* get raw humidity */
    hum_int = dataBlock[6] << 8 | dataBlock[7];

    vals[0] = (1.8 * BME280_compensate_T_double(temp_int)) + 32;

    vals[1] = 0.02953 * sta2sea(BME280_compensate_P_double(press_int) / 100.0);

    vals[2] = BME280_compensate_H_double(hum_int);

    i2c_smbus_write_byte_data(fd, CTRL_HUM, 0x0);

    i2c_smbus_write_byte_data(fd, CTRL_MEAS, 0x0);

    return 0;
}



