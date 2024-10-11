import csv, yaml, time, json, random
from datetime import datetime
from types import SimpleNamespace

########################################################################################################
########################################################################################################

# LOAD CSV INTO DICT ARRAY
def load_csv(file_path: str):
    container = []

    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.DictReader(file)
        
        for row in csv_reader:
            container.append(row)

    return container

########################################################################################################
########################################################################################################

# TIMESTAMPED PRINT STATEMENT
def log(msg, with_break=False):
    now = datetime.now()
    timestamp = now.strftime("%H:%M:%S.%f")[:-3]

    if with_break:
        print(f'\n[{timestamp}]\t {msg}', flush=True)
    else:
        print(f'[{timestamp}]\t {msg}', flush=True)
    
########################################################################################################
########################################################################################################

# VALIDATE AN INPUT DICT BASED ON A REFERENCE DICT
def validate_dict(input_data: dict, reference: dict, ns=False):
    container = {}

    # REFERENCE DICT: KEY_NAME => TYPE_FUNC
    for prop_name, type_cast_func in reference.items():

        # MAKE SURE DICT KEY EXIST
        if prop_name not in input_data:
            raise Exception(f'KEY ERROR: MISSING PROPERTY ({prop_name})')
        
        # MAKE SURE EVERY VALUE CAN BE CAST TO ITS EXPECTED TYPE
        try:
            input_value = input_data[prop_name]
            container[prop_name] = type_cast_func(input_value)

        except Exception as error:
            raise Exception(f'CASTING ERROR (prop: {prop_name}): {error}')
    
    # CONVERT TO NAMESPACE ON-REQUEST
    if ns: return TO_NAMESPACE(container)

    return container

########################################################################################################
########################################################################################################

def load_yaml(file_path, ns=False):
    with open(file_path, 'r') as file:
        data_dict: dict = yaml.safe_load(file)
        if ns: return TO_NAMESPACE(data_dict)
        return data_dict
    
def save_yaml(file_path, data_dict):
    with open(f'{file_path}', 'w') as file:
        yaml.dump(data_dict, file, default_flow_style=False)

########################################################################################################
########################################################################################################

# CONVERT DICT TO CLASS NAMESPACE
def TO_NAMESPACE(d):
    if isinstance(d, dict):
        # Recursively convert the dictionary to a namespace
        for key, value in d.items():
            if isinstance(value, dict):
                d[key] = TO_NAMESPACE(value)
            elif isinstance(value, list):
                # Process each item in the list
                d[key] = [TO_NAMESPACE(item) if isinstance(item, dict) else item for item in value]
        return SimpleNamespace(**d)
    elif isinstance(d, list):
        # Process a list if the outermost structure is a list
        return [TO_NAMESPACE(item) if isinstance(item, dict) else item for item in d]
    else:
        # If d is neither a dict nor a list, return it as is
        return d
    
########################################################################################################
########################################################################################################

class create_timer:
    def __init__(self):
        self.start_time: float = time.time()

    def stop(self):
        self.end_time: float = time.time()
        return round(self.end_time - self.start_time, 3)
    
########################################################################################################
########################################################################################################

def pprint(data: dict):
    assert type(data) == dict, 'YOU CAN ONLY PRETTY PRINT DICTS'
    print(json.dumps(data, indent=4))

########################################################################################################
########################################################################################################

def timeout_range(min, max):
    duration = random.uniform(min, max)
    time.sleep(duration)

########################################################################################################
########################################################################################################