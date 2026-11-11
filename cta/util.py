import os
import json
# ###########
# LOAD JSON FUCNTION
# ###########
def requires_existing_path(func):
    def wrapper(trace_path, *args, **kwargs):
        assert os.path.exists(trace_path), f"File {trace_path} does not exist."
        return func(trace_path, *args, **kwargs)
    return wrapper

@requires_existing_path
def load_json(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
        return data

@requires_existing_path
def check_path(file_path):
    try:
        assert file_path.endswith(".nsys-rep") or file_path.endswith(".sqlite")
    except Exception as err:
        raise RuntimeError(f"File Path: {file_path}, Error: {err}")
    return True

@requires_existing_path
def get_directory_name(file_path):
    try:
        return os.path.dirname(file_path)
    except Exception as err:
        raise RuntimeError(f"Error occured while extractoing path: {err}")
    
@requires_existing_path
def get_file_name(file_path):
    try:
        _name = os.path.basename(file_path)
        assert _name.endswith("nsys-rep") or _name.endswith("sqlite")
        return _name.split(".")[0]
    except Exception as err:
        raise RuntimeError(f"Error occured while extracting file name: {err}")
    