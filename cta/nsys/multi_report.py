from .nsys import generic_event
from .nsys import os_event
from .nsys import cupti_activity_event
from .nsys import nvtx_pushpop
from .nsys import osrt_sum

import matplotlib.pyplot as plt

def mapping_decorator(mapping, error_message):
    def decorator(func):
        def wrapper(key):
            try:
                return mapping[key]
            except KeyError as err:
                raise RuntimeError(f"{error_message}: {err}")
        return wrapper
    return decorator
function_mappings = {
    "smoothing": {
        "savitzky_golay_filter": generic_event.savgol_smooth,
        "exponential_moving_average": generic_event.exponential_moving_average,
        "moving_average": generic_event.moving_average,
    },
    "criteria": {
        "median": generic_event.get_median,
        "max": generic_event.get_max,
        "mean": generic_event.get_mean,
    },
}

@mapping_decorator(function_mappings["smoothing"],"Provide a valid name for smoothing")
def get_smoothing_function_name(smoothing):
    pass

@mapping_decorator(function_mappings["criteria"],"provide a valid plotting criteria")
def get_criteria_function(criteria):
    pass

def util_function(trace_paths,
                  _type,
                #   plot_label,
                  smoothing,
                #   raw_data,
                #   title,
                  **kwargs
                  ):
    assert type(trace_paths) == list

    if smoothing is not None:
        if type(smoothing)==str:
            smoothing = [smoothing]
    # plt.figure(figsize=(5,3))

    stats = []
    smoothing_stats = {}
    plotting_criteria = kwargs.get("plotting_criteria","median")

    plotting_func = get_criteria_function(plotting_criteria)
    # if all_data is mentioned as True, plotting_criterial is ignored and returns all the non zero values 
    all_data = kwargs.get("all_data", False) 
    for item in trace_paths:
        obj = generic_event(item)
        f_names = {
            "tensor_active": obj.tensor_active_values,
            "sm_issue": obj.sm_issue_values,
            "sm_active": obj.sm_active_values,
            "compute_warps": obj.compute_warps_values
        }
        try:
            values = f_names[_type]
        except KeyError as err:
            raise RuntimeError(f"Invalid function type requested: {err}")
        if all_data is False:
            stats.append(plotting_func(values))
        else:
            stats.append(values)
        if smoothing:
            for key in smoothing:
                func = get_smoothing_function_name(key)
                _t = func(values, kwargs)
                if all_data is False: _t = plotting_criteria(_t)
                if smoothing_stats.get(key, None) is None:
                    smoothing_stats.update({
                        key:[_t]
                    })
                else:
                    smoothing_stats[key].append(_t)
    return {
        "raw_stats": stats,
        "smoothing_stats": smoothing_stats
    }

def event_handler(event_type):
    """
    Decorator to handle the events with a specific type

    Parameters;
        -event_type
    Returns:
        - A decorator function that wraps the original functions
    """
    def decorator(func):
        def wrapper(trace_paths, 
                    # plot_label, 
                    smoothing, 
                    # raw_data, 
                    # title, 
                    **kwargs):
            # call the utility function
            return util_function(trace_paths=trace_paths,
                  _type = event_type,
                #   plot_label=plot_label,
                  smoothing=smoothing,
                #   raw_data=raw_data,
                #   title = title,
                  **kwargs
                  )
        return wrapper
    return decorator

# use the decorator to define specific event function
@event_handler(event_type="tensor_active")
def tensor_active(trace_paths,
                #   plot_label,
                  smoothing,
                #   raw_data,
                #   title,
                  **kwargs
                  ):
    """
    handle tensor active event
    """
    pass

@event_handler(event_type="sm_active")
def sm_active(trace_paths,
            # plot_label,
            smoothing,
            # raw_data,
            # title,
            **kwargs):
     pass

@event_handler(event_type="sm_issue")
def sm_issue(trace_paths,
            # plot_label,
            smoothing,
            # raw_data,
            # title,
            **kwargs):
    pass

@event_handler(event_type="compute_warps")
def compute_warps(trace_paths,
            # plot_label,
            smoothing,
            # raw_data,
            # title,
            **kwargs):
    pass


# ############
# OS & CUDA REPORTS
# ############

import pandas as pd
os_cuda_function_mappings = {
    "os_stats": os_event,
    "kernel_stats": cupti_activity_event,
    "nvtx_stats": nvtx_pushpop,
    "osrt_sum": osrt_sum
}
def activity_handler(function_type,error_message):
    """
    Decorator to handle the event activity with a specific type
    Parameters:
    - activity_type
    Returns:
    - A decorator function that warps the original functions
    """
    def decorator(func):
        def wrapper(trace_paths, *args, **kwargs):
            # get the variable
            # return the count as well
            try:
                assert type(trace_paths) == list
                stats = []
                for trace in trace_paths:
                    try:
                        obj = os_cuda_function_mappings[function_type](trace_path = trace)
                        events = pd.DataFrame(obj.get_events)
                        # print(events)
                        try:
                            events[4] = events[2] - events[1]
                            events = events.groupby(3)[4].agg(['mean','median', 'count', 'sum'])
                            # print(events.to_dict())
                        except Exception as err:
                            print(f"Problem occured: {err}")
                        # assert type(events) == pd.core.series.series
                        stats.append(events.to_dict())
                    except Exception as err:
                        raise RuntimeError(f"Problem in some trace file: {err}")
                # returns a list of dict
                # [{
                #    mean:{}, median:{}, count:{}
                #  },
                #   {},
                #   {}...
                # ]
                return stats
            except Exception as err:
                raise RuntimeError(f"{error_message}: {err}")
        return wrapper
    return decorator

@activity_handler(function_type="os_stats", error_message="Problem in pthread analysis")
def os_stats(trace_paths, *args, **kwargs):
    pass

@activity_handler(function_type="kernel_stats", error_message="Problem in cuda kernel analysis")
def kernel_stats(trace_paths, *args, **kwargs):
    pass

@activity_handler(function_type="nvtx_stats", error_message="Problem in cuda nvtx analysis")
def nvtx_stats(trace_paths, *args, **kwargs):
    pass

@activity_handler(function_type="osrt_sum", error_message="Problem in osrt_sum")
def osrt_sum(trace_paths, *args, **kwargs):
    pass