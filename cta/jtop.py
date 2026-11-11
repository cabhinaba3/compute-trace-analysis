import os
import json
from .util import *
from typing import List, Dict, Any
# ############
# GLOBAL VARIABLES
# ############
CPU_UTILIZATION = "cpu_utilization"
GPU_UTILIZATION = "gpu_utilization"
GPU_MEMORY_USAGE = "gpu_memory_usage"
RAM_USAGE = "ram_usage"
SWAP_MEMORY_USAGE = "swap_memory_usage"
TEMPERATURE = "temperature"
ENERGY_CONSUMPTION = "energy_consumption"
GPU_TEMPERATURE = "gpu_temperature"
class Jtop:
    def __init__(self, trace_path,
                 cpu_core=6,
                 jtop_interval = 1.0 # always give in seconds
                 ):
        # check if the path exits of not
        self.trace_path = trace_path
        self.obj = load_json(trace_path)

        self.obj_len = len(self.obj)

        self.num_cpu_core = cpu_core
        self.jtop_interval = jtop_interval

        self.result = {
            CPU_UTILIZATION: [],
            GPU_UTILIZATION: [],
            GPU_MEMORY_USAGE: [],
            RAM_USAGE: [],
            SWAP_MEMORY_USAGE: [],
            TEMPERATURE: [],
            ENERGY_CONSUMPTION: [],
            GPU_TEMPERATURE: []
        }
        # declaire inference start and end time
        self.inference_start_time = None
        self.inference_end_time = None
        
        self.calculate_parameters()
    def update_result_key(self, key_name, value):
        self.result[key_name].append(value) 
    def calculate_cpu_usage(self,):
        for item in self.obj:
            cpu_stats = 0.0
            for i in range(self.num_cpu_core):
                cpu_stats += item[f"CPU{i+1}"]
            cpu_stats /= self.num_cpu_core
            self.update_result_key(CPU_UTILIZATION, cpu_stats)
    def calculate_energy_consumption(self,):
        # self.update_result_key(ENERGY_CONSUMPTION, self.obj[0]['Power TOT'] * 10**(-3))
        power_rail = "Power TOT"
        for i in range(1, len(self.obj)):
            # self.obj[i][power_rail]-=self.obj[i-1][power_rail]
            power_consumed_ = self.obj[i][power_rail]
            time_interval = self.obj[i]["time"]-self.obj[i-1]["time"]
            power_consumed = power_consumed_ * time_interval * 0.5 * 10**(-12)
            if len(self.result[ENERGY_CONSUMPTION])>0:
                power_consumed+= self.result[ENERGY_CONSUMPTION][-1]
            self.update_result_key(ENERGY_CONSUMPTION,
                                   power_consumed
                                #    self.obj[i]['Power TOT'] * time_interval * 10**(-12) 
                                   )
    
    def calculate_parameters(self,):
        self.calculate_cpu_usage()
        self.calculate_energy_consumption()
        for item in self.obj:
            # update inference time
            if self.inference_start_time is None:
                self.inference_start_time = item["time"]
            # update inference end time
            self.inference_end_time = item["time"]
            self.update_result_key(GPU_UTILIZATION,item["GPU"])
            self.update_result_key(GPU_MEMORY_USAGE, item["gpu_memory_usage"])
            self.update_result_key(SWAP_MEMORY_USAGE, item["SWAP"])
            self.update_result_key(RAM_USAGE, item["RAM"])
            try:
                self.update_result_key(GPU_TEMPERATURE,item["Temp gpu"])
            except Exception as err:
                print(f"Error for Jetson Nano, may change later:{err}")
    @property
    def get_jtop_result(self,):
        return self.result
    
    @property
    def get_cpu_utilization(self,):
        return self.result[CPU_UTILIZATION]
    
    @property
    def get_gpu_utilization(self,):
        return self.result[GPU_UTILIZATION]
    @property
    def get_swap_memory_usage(self,):
        return self.result[SWAP_MEMORY_USAGE]
    @property
    def get_ram_usage(self,):
        return self.result[RAM_USAGE]
    @property
    def get_energy_consumption(self,):
        """
        Get total energy consumption in J
        """
        assert type(self.result[ENERGY_CONSUMPTION]) == list
        # return the last element
        return self.result[ENERGY_CONSUMPTION][-1]
    @property
    def get_gpu_memory_usage(self,):
        return self.result[GPU_MEMORY_USAGE]
    @property
    def get_gpu_temperature(self,):
        return self.result[GPU_TEMPERATURE]
    @property
    def get_power_cosnumption(self,):
        """
        Get power consumption in J/second (W)
        """

        return self.get_energy_consumption * 1e9 / (self.inference_end_time - self.inference_start_time)
        # import statistics
        # return statistics.median(list(set(self.result[ENERGY_CONSUMPTION])))

    @property
    def get_peak_gpu_memory_usage(self):
        return max(self.result[GPU_MEMORY_USAGE])