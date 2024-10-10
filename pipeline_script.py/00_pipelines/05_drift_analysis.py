from funcs import kafka_utils, cassandra_utils, mlflow_utils, jaeger_utils
from funcs import thread_utils, misc, constants

########################################################################################
########################################################################################

class pipeline_component:
    def __init__(self, structs):

        # CREATE INSTANCED CLIENTS
        self.cassandra = cassandra_utils.create_instance()
        self.kafka = kafka_utils.create_instance()
        self.jaeger = jaeger_utils.create_instance('DRIFT_ANALYSIS')

        # IN A BACKGROUND THREAD, DO...
        self.kafka.subscribe(constants.kafka.DRIFT_ANALYSIS, self.on_kafka_event, structs.thread_beacon)

    ########################################################################################
    ########################################################################################

    def on_kafka_event(self, kafka_input: dict):
        assert 'trace' in kafka_input, "[DRIFT] EVENT MISSING 'trace' PROPERTY"
        trace_predecessor = kafka_input['trace']

        with self.jaeger.create_span('MEASURING DRIFT', trace_predecessor) as span:
            misc.timeout_range(0.02, 0.04)
            misc.pprint(kafka_input)

########################################################################################
########################################################################################

thread_utils.start_coordinator(pipeline_component)