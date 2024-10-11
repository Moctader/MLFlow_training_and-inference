# !pip install mlflow
import mlflow, time
from mlflow.tracking import MlflowClient
from funcs import constants, misc, thread_utils

global_config = constants.global_config()
MLFLOW_BROKER: str = f"http://{global_config.endpoints.host}:{global_config.endpoints.ports.mlflow}"

########################################################################################################
########################################################################################################

class create_instance:
    def __init__(self):
        
        # POINT REGISTRY & MODEL TRACKING TOWARDS THE CLUSTER BROKER
        mlflow.set_tracking_uri(MLFLOW_BROKER)
        mlflow.set_registry_uri(MLFLOW_BROKER)

        # CREATE A REUSABLE CLIENT
        self.instance = MlflowClient()

    ########################################################################################################
    ########################################################################################################

    def list_all_models(self):
        container = {}
        
        for item in self.instance.search_registered_models():
            version_data = item.__dict__['_latest_version']
            assert len(version_data) == 1, '[MLFLOW] THERE IS AN ABNORMAL NUMBER OF MODEL VERSIONS'

            container[item._name] = item.aliases

            # container[item._name] = {
            #     'latest_version': version_data[0].version,
            #     'version_aliases': item.aliases
            # }
    
        return container

    ########################################################################################################
    ########################################################################################################

    def model_exists(self, model_name: str, model_version: int) -> bool:
        assert isinstance(model_name, str), 'MODEL NAME MUST BE OF TYPE STR'
        assert isinstance(model_version, int), 'MODEL VERSION MUST BE OF TYPE INT'

        try:
            result = self.instance.get_model_version(name=model_name, version=model_version)
            return True
        
        except:
            return False

    ########################################################################################################
    ########################################################################################################

    def load_model(self, model_name: str, model_version: int):
        assert isinstance(model_name, str), 'MODEL NAME MUST BE OF TYPE STR'
        assert isinstance(model_version, int), 'MODEL VERSION MUST BE OF TYPE INT'
        
        model_uri = f'models:/{model_name}/{model_version}' 
        model = mlflow.pyfunc.load_model(model_uri)

        return model

    ########################################################################################################
    ########################################################################################################

    def load_fake_model(self, model_name: str, version_alias: str, version_number: int|str):
        return mock_ml_model(model_name, version_alias, version_number)
    
    ########################################################################################################
    ########################################################################################################

    def subscribe(self, callback_func, process_beacon):
        def consume_events():
            previous_value = None
            misc.log(f'[MLFLOW] STARTED POLLING')

            while process_beacon.is_active():
                try:
                    current_value = self.list_all_models()
                    
                    # IF THE VALUE HAS CHANGED -- RUN CALLBACK FUNC
                    if current_value != previous_value:
                        callback_func(current_value)
                        previous_value = current_value
                    
                    time.sleep(global_config.pipeline.polling_cooldown)

                except Exception as error:
                    misc.log(f'[MLFLOW] CONSUME ERROR: {error}')
                    
        # START CONSUMING EVENTS IN BACKGROUND THREAD
        thread_utils.start_thread(consume_events)
    
########################################################################################################
########################################################################################################

class mock_ml_model:
    def __init__(self, model_name, version_alias, version_number):
        self.model_name = model_name
        self.version_alias = version_alias
        self.version_number = version_number

    def predict(self, input):
        misc.timeout_range(0.01, 0.02)
        return f'{input} -> [{self.model_name}, v_{self.version_number}]'