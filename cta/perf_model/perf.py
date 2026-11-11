import os
import numpy as np
from .math_model import *
from ..nsys import multi_report as rpt

class GatherData:
    """This class gathers data from each and every nsys-rep file of each experiment and returns the GPU sync event,
    Execution Context event.
    """
    def __init__(self,nsys_paths:list):
        """Initialization
        """
        self.nsys_paths = nsys_paths
        assert type(self.nsys_paths) == list and len(self.nsys_paths) > 0
        print("Assert check done..")
        try:
            self.sync_data = rpt.kernel_stats(nsys_paths)
        except RuntimeError as err:
            raise RuntimeError(f"Runtime Error Occured: {err}") 
        try:
            # this data should come from the trace not the nsys own mean or median
            # do the nsys-trace file and do the calculations [TODO]
            self.inference_execution_data = rpt.nvtx_stats(nsys_paths)
        except RuntimeError as err:
            raise RuntimeError(f"Runtime Error Occured:{err}")
        
        self.values_map = {
            "sync_data": self.sync_data,
            "infer_data": self.inference_execution_data
        }
    
    def sync_api_data(self,**kwargs):
        """Gather GPU API data based on field names
        For ex: Gather cuda api for nvidia gpu
        """
        # is_mean = kwargs.get("mean", True)
        # if is_mean: return np.mean(self.values_map["sync_data"])
        return self.values_map["sync_data"]
    
    def infer_api_data(self, **kwargs):
        """Gather inference api data based on field names
        For ex: Gather TensorRT data for NVIDIA Gpus
        """
        # is_mean = kwargs.get("mean", True)
        return self.values_map["infer_data"]
        # return self.values_map["infer_data"]
    

    def communication_api_data(self,):
        pass


    def energy_api_data(self,):
        pass

    def get_values_map(self,):
        return self.values_map