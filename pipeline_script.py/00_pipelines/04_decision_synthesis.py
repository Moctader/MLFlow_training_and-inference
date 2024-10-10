from funcs import kafka_utils, redis_utils, cassandra_utils, mlflow_utils, jaeger_utils
from funcs import thread_utils, misc, constants

########################################################################################
########################################################################################

class pipeline_component:
    def __init__(self, structs):

        # CREATE INSTANCED CLIENTS
        self.cassandra = cassandra_utils.create_instance()
        self.kafka = kafka_utils.create_instance()
        self.jaeger = jaeger_utils.create_instance('DECISION_SYNTHESIS')

        # IN A BACKGROUND THREAD, DO...
        self.kafka.subscribe(constants.kafka.DECISION_SYNTHESIS, self.on_kafka_event, structs.thread_beacon)
        self.redis.subscribe(constants.redis.MODEL_PIPELINES, self.on_redis_change, structs.thread_beacon)

    ########################################################################################
    ########################################################################################

    def on_kafka_event(self, kafka_input: dict):
        try:
            assert 'trace' in kafka_input, "[DECISION] EVENT MISSING 'trace' PROPERTY"
            assert 'prediction_batch' in kafka_input, "[DECISION] EVENT MISSING 'prediction_batch' PROPERTY"

            trace_predecessor = kafka_input['trace']
            prediction_batch = kafka_input['prediction_batch']

            # FORCE STOCK SYMBOL TO LOWERCASE TO AVOID CAPITALIZATION ACCIDENTS
            stock_symbol = prediction_batch['input_row']['symbol'].lower()
            # assert stock_symbol in self.deployed_model_pipes, f"[INFERENCE] NO PIPES EXIST FOR STOCK SYMBOL '{stock_symbol}'"

            # MAKE MONETARY DECISION BASED ON PIPE PREDICTIONS
            def decision():
                with self.jaeger.create_span(f"MAKING '{stock_symbol}' BUY/SELL/HOLD DECISION", trace_predecessor) as span:
                    misc.timeout_range(0.01, 0.02)
                    misc.pprint(kafka_input)

            # SAVE MODEL PREDICTIONS IN DB
            def save_data():
                with self.jaeger.create_span("DB: SAVING MODEL PREDICTIONS", trace_predecessor) as span:
                    misc.timeout_range(0.01, 0.02)

            thread_utils.start_thread(decision)
            thread_utils.start_thread(save_data)

        except AssertionError as error:
            misc.log(error)
    
    ########################################################################################
    ########################################################################################

    def on_redis_change(self, latest_value: dict):
        pass

########################################################################################
########################################################################################

thread_utils.start_coordinator(pipeline_component)