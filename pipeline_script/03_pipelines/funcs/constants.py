from dataclasses import dataclass
import funcs.misc as misc

###################################################################################################
###################################################################################################

# KAFKA MQ TOPICS
@dataclass(frozen=True)
class kafka:
    DATA_REFINERY: str = 'data_refinery'
    MODEL_DISPATCH: str = 'model_dispatch'
    DECISION_SYNTHESIS: str = 'decision_synthesis'
    DRIFT_ANALYSIS: str = 'drift_analysis'

###################################################################################################
###################################################################################################

# CASSANDRA DB TABLES
@dataclass(frozen=True)
class cassandra:
    STOCKS_TABLE: str = 'john.refined_stock_data'

###################################################################################################
###################################################################################################

# REDIS CACHE KEYSPACES
@dataclass(frozen=True)
class redis:
    MODEL_PIPELINES: str = 'model_pipelines'

###################################################################################################
###################################################################################################

def global_config():
    data_dict: dict = misc.load_yaml('../00_configs/global_config.yaml')
    return misc.TO_NAMESPACE(data_dict)

###################################################################################################
###################################################################################################