'''
D A T A    A C Q U I S I T I O N    M O D U L E

Created on 31/03/2021
Author: Alberto Quesada (albertoquesadaleon@gmail.com)
'''

### IMPORTS ###
import bme280
import smbus2
from time import sleep, time
from datetime import datetime
from sds011 import *
import csv

### INITIALIZE ###
# BME config
port = 1
address = 0x76
bus = smbus2.SMBus(port)
bme280.load_calibration_params(bus,address)
# SDS011 config
sds011 = SDS011("/dev/ttyUSB0")
sds011.sleep(sleep=True)
### MAIN ###
while True:
    init_time = time()
    sds011.sleep(sleep=False) # Turn ON sds011 sensor
    sleep(15)
    pm_2_5,pm_10 = sds011.query() # Request measure to the SDS011 sensor
    sds011.sleep(sleep=True) # Turn OFF sds011 sensor
    bme280_data = bme280.sample(bus,address) # Request measure to the BME280 sensor
    humidity  = bme280_data.humidity
    pressure  = bme280_data.pressure
    temperature = bme280_data.temperature
    current_row = [datetime.now(), temperature, pressure, humidity, pm_2_5, pm_10]
    with open('/home/pi/Projects/data_acquisition/results.csv', 'a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(current_row)
        csvfile.flush()
    time_spent = time()-init_time
    sleep(60-time_spent)