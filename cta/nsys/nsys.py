from .db import *
import ast
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from abc import ABC, abstractmethod

# handles individual nsys reports and sends analysis files
class generic_event:
    def __init__(self, file_path, *args, **kwargs):
        """
        Class to handle individual nsys reports and reports back some analysis
        The GENERIC_EVENTS table contains the following columns
            - GPU Clock Frequency
                The average GPC clock frequncy in Hz
            - SYS Clock Frequency
            - Unallocated Warps in Active SM
            - Compute Warps In Flight
                - The ratio of active compute shader warps resident on the SMs to the maximum number of warps per SM as a percentage
            - Pixel Warps In Flight
            - Vertex\/Tess\/Geometry Warps In Flight
            - Total SM Occupancy
            - GR Active
            - Sync Compute In Flight
            - Async Compute In Flight
                Total percentage of cycles with asynchrnous compute in flight
            - NVLink bytes recieved
            - NVLink bytes transmitted
            - PCIe Rx Throughput
            - PCIe Tx Throughput
            - DRAM Read Throughput
            - DRAM Write Throughput
            - Tensor Active \/ FP16 Active
                The ratio of cycles that SM tensor pipes were active issuing tensor instructions to the number of cycles in the sample period as a percentage.
            - SM Issue
                The ratio of cycles that SM subpartitions(warp scheduler) issued an instruction to the number of cycles in the sample period as a percentage
            - SM Active
                The ratio of cycles SMs had at least 1 warp in flight (allocated on SM) to the number of cycles as a percentage. A value of 0 indicates all SMs were idle (no warps in flight). A value of 50% can indicate some gradient between all SMs active 50% of the sample period or 50% of SMs active 100% of the sample period
                
        """
        self.trace = file_path
        self.db_object = db(nsys_report_path=self.trace, 
                       output_path="inherit")
        # for newer nsight version the events changed to GPU_METRICS
        # table_name = "GENERIC_EVENT"
        table_name = "GPU_METRICS"
        
        # the table name is GENERIC_EVENTS
        try:
            column_names = ["value"]
            # column_names = ["data"]
            self.db_object.query(column_names=column_names,
                                table_name=table_name)
            # print(self.db_object_query)
            self.events = self.db_object.fetch_query
        except Exception as err:
            raise RuntimeError(f"Problem in fetching the database: {err}")
        
    def util_values(self,column_name):
        values = []
        # print(self.events)
        for event in self.events:
            print(event,"something is happening here")
            event = ast.literal_eval(event[0])
            try:
                if event[column_name] != '0':
                    values.append(float(event[column_name]))
            except Exception as err:
                continue
        try:
            assert len(values) > 0
        except Exception as err:
            raise RuntimeError(f"Problem in trace path: {self.trace}:{err}")
        return values
    
    @property 
    def tensor_active_values(self,):
        return self.util_values(column_name="Tensor Active [Throughput %]")
    
    @property
    def sm_issue_values(self,):
        return self.util_values(column_name="SM Issue [Throughput %]")
    
    @property
    def compute_warps_values(self,):
        return self.util_values(column_name="Compute Warps in Flight [Avg Warps per Cycle]")
    @property
    def sm_active_values(self,):
        return self.util_values(column_name="SMs Active [Throughput %]")
    
    # @staticmethod
    # def line_plot(values,
    #               label,
    #               xlabel,
    #               ylabel,
    #               title,
    #               full=False):
    #     if full == False:
    #         max_len = min(500,len(values)/10)
    #     max_len = len(values)
    #     x = np.linspace(0, max_len, max_len)
    #     plt.figure(figsize=(25,3))
    #     plt.plot(x, values[:max_len], label=label)
    #     plt.xlabel(xlabel)
    #     plt.ylabel(ylabel)
    #     plt.title(title)

    #     plt.grid(True)
    #     plt.show()

    @staticmethod
    def moving_average(series, options):
        window_size = options.get("window_size",None)
        assert window_size is not None
        return np.convolve(series, np.ones(window_size) / window_size, mode='valid')
    
    @staticmethod
    def exponential_moving_average(series, options,):
        smoothing_factor = options.get("smoothing_factor", None)
        assert smoothing_factor is not None
        ema = np.zeros_like(series)
        ema[0] = series[0]
        for t in range(1, len(series)):
            ema[t] = smoothing_factor * series[t] + (1 - smoothing_factor) * ema[t-1]
        return ema
    
    @staticmethod
    def savgol_smooth(series,options,):
        window_size = options.get("window_size", None)
        polyorder = options.get("polyorder", None)
        assert None not in [window_size, polyorder]
        return savgol_filter(series, window_length = window_size, polyorder = polyorder)
    
    # @staticmethod
    # def plot(values,
    #          plot_label,
    #          ):
    #     window_size = 501
    #     smoothing_factor = 0.1
    #     polyorder = 3
        
    #     sma = generic_event.moving_average(values,window_size)
    #     ema = generic_event.exponential_moving_average(values, smoothing_factor)
    #     savgol = generic_event.savgol_smooth(values, window_size, polyorder)

    #     plt.figure(figsize=(14,8))

    #     plt.plot(values, label="Original Time Series", alpha=0.5)
    #     plt.plot(sma, label="Moving Average", color="pink")
    #     plt.plot(ema, label="Exponential Moving Average", color='green')
    #     plt.plot(savgol, label="Savitzksy-Golay Filter", color='red')

    #     plt.title("Smoothing methods")
    #     plt.xlabel("Sampling instances")
    #     plt.yabel("Amplitude")

    #     plt.legend()

    #     plt.tight_layout()
    #     plt.show()

    @staticmethod
    def get_median(value):
        import statistics
        return statistics.median(value)
    @staticmethod
    def get_max(value):
        return max(value)
    @staticmethod
    def get_mean(value):
        return np.mean(value)


# ACTIVITY EVENTS
# These events returns the avg value
# Will have same structure
class activity_event(ABC):
    def __init__(self, trace_path, 
                 table_name, 
                 column_names,
                 *args, **kwargs):
        """
        This class is the parent class of all the activity events the sqlite format
        """
        if type(column_names) == str and column_names.lower() == "all":
            column_names = ["*"]
        assert type(column_names) == list
        self.db_object = db(nsys_report_path=trace_path,
                            output_path="inherit"
                            )
        self.table_name = table_name
        try:
            self.db_object.query(column_names = column_names,
                                 table_name = table_name
                                 )
            self.events = self.db_object.fetch_query
        except Exception as err:
            raise RuntimeError(f"Problem in fetching the database: {err}")
        try:
            self.events = self.update_function_name()
        except Exception as err:
            pass
            # raise RuntimeWarning("Problem occured: {err}")
        return self.events
    
    def update_function_name(self):
        """
        From the nameId we need to get the name of the function.
        Calls the stringId table to get the name.
        The structure of the table stringIds is: id and value. We want the value.
        """
        result = []
        for event in self.events:
            try:
                if event[0] is None:
                    continue
                value = self.db_object.get_stringIds_value(_id=int(event[0]))
                event = list(event)
                event.append(value[0][0])
                result.append(tuple(event))
                # print(event)
                # event = tuple(list(event).extend(value[0][0]))
            except Exception as err:
                raise RuntimeError(f"Problem in fetching the stringIds database for {event}: {err}")
        return result
    @property
    def get_events(self,):
        return self.events
    
    # @abstractmethod
    # def process_events(self,):
    #     pass
class cupti_activity_event(activity_event):
    def __init__(self,trace_path, *args, **kwargs):
        """
        This class handles summary of CUDA API function and their executions times.

        It looks at the table CUPTI_ACTIVITY_KIND_RUNTIME
        It has the following columns:
        - start: event start time in ns
        - end: event end time in ns
        - eventClass: references ENUM_NSYS_EVENT_CLASS (id)
        - globalTid: Serialized global id
        - correlationId: ID used to identify events that this function calls has triggered
        - nameId: References stringIDs(id) --  function name
        - returnValue: return value of the function call
        - callchainId: References CUDA_CALLCHAINS
        """
        super().__init__(trace_path = trace_path, table_name="CUPTI_ACTIVITY_KIND_RUNTIME",
                         column_names = ["nameId","start","end"],
                         *args, **kwargs
                         )
    def process_event(self,):
        pass

class os_event(activity_event):
    def __init__(self, trace_path, *args, **kwargs):
        """
        Class to handle individual nsys os events and reports back the analysis.
        Reported the source code for nsys recipe for os events.

        Note: Turn on osrt tracking in nsys to be able to use it
        This class sends you the data for different os classes.
        Users can specify a class name to class description of that os event.
        We are mainly interested in threads switch and locks.

        Users can implement their own methods here
        """
        super().__init__(trace_path=trace_path,
                         table_name="OSRT_API",
                         column_names = ["nameId","start","end"],
                         *args,
                         *kwargs
                         )
    def process_event(self,):
        pass

class nvtx_pushpop(activity_event):
    def __init__(self, trace_path, *args, **kwargs):
        """
        Class to handle individual nsys nvtx range events and reports back the analysis.
        Note: This analysis is very useful to get the stats for TensorRT executionContext
        """
        super().__init__(trace_path=trace_path,
                         table_name="NVTX_EVENTS",
                         column_names=["textId","start", "end"],
                         *args,
                         **kwargs
                         )
    def process_event(self,):
        pass

class osrt_sum(activity_event):
    def __init__(self, trace_path, *args, **kwargs):
        """
        Class to handle osrt_sum for jetson_nano
        """
        super().__init__(trace_path=trace_path,
                         table_name="OSRT_SUM",
                         column_names=["textId","start","end"],
                         *args,
                         **kwargs
                         )
    def process_event(self):
        pass