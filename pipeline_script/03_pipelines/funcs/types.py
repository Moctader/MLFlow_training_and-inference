import time
from datetime import datetime

# EXPECTED INPUT/OUTPUT FOR KAFKA/CASSANDRA
# USED BY misc.validate_dict

###################################################################################################
###################################################################################################

# CONVERT STRING TEXT TO UNIX TIMESTAMP
def string_to_unix(date_text: str|int) -> int:

    if type(date_text) == int:
        return date_text

    date_tuple = datetime.strptime(date_text, '%Y-%m-%d')
    return int(time.mktime(date_tuple.timetuple()))

# WHAT SHOULD CLEAN STOCK DATA LOOK LIKE?
REFINED_STOCK_DATA: dict = {
    'symbol': str,
    'timestamp': string_to_unix,
    'high': float,
    'low': float,
    'open': float,
    'close': float,
    'adjusted_close': float,
    'volume': lambda x: int(float(x)),
}

# WERE GOING TO MAKE A FINAL DECISION BASED ON MULTIPLE MODELS' PREDICTION OUTPUT.
# WHAT SHOULD THE BATCH DATA LOOK LIKE?
PREDICTION_BATCH: dict = {
    'input_row': dict,
    'predictions': dict
}

# {
#     'input_row': {
#         'symbol': str,
#         'timestamp': int,
#         'high': float,
#         'low': float,
#         'open': float,
#         'close': float,
#         'adjusted_close': float,
#         'volume': int,
#     },
#     'predictions': {
#         'model_1_name': 'prediction_1',
#         'model_2_name': 'prediction_3',
#         'model_n_name': 'prediction_n',
#     }
# }