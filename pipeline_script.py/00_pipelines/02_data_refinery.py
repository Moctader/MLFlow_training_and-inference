from funcs import kafka_utils, cassandra_utils, jaeger_utils
from funcs import thread_utils, misc, constants, types

########################################################################################
########################################################################################

class pipeline_component:
    def __init__(self, structs):

        # CREATE INSTANCED CLIENTS
        self.cassandra = cassandra_utils.create_instance()
        self.kafka = kafka_utils.create_instance()
        self.jaeger = jaeger_utils.create_instance('DATA_REFINERY')

        # IN A BACKGROUND THREAD, DO...
        self.kafka.subscribe(constants.kafka.DATA_REFINERY, self.on_kafka_event, structs.thread_beacon)

    ########################################################################################
    ########################################################################################

    def on_kafka_event(self, kafka_input: dict):

        # FORMAT & VALIDATE UNPROCESSED STOCK DATA 
        with self.jaeger.create_span('REFINING RAW STOCK DATA') as parent:
            refined_stock_data: dict = misc.validate_dict(kafka_input, types.REFINED_STOCK_DATA)
            stock_symbol: str = refined_stock_data['symbol'].lower()

        # PUSH VALID STOCK DATA TO THE DISPATCHER
        def kafka_inject():
            with self.jaeger.create_span(f"KAFKA: FORWARDING '{stock_symbol}' STOCK DATA", parent) as span:
                self.kafka.push(constants.kafka.MODEL_DISPATCH, {
                    'trace': self.jaeger.create_context(span),
                    'refined_stock_data': refined_stock_data
                })

        # WRITE THE ROW TO THE DB
        def cassandra_inject():
            with self.jaeger.create_span("DB: SAVING VALID STOCK DATA", parent) as span:
                self.cassandra.write(constants.cassandra.STOCKS_TABLE, refined_stock_data)

        thread_utils.start_thread(kafka_inject)
        thread_utils.start_thread(cassandra_inject)

########################################################################################
########################################################################################

thread_utils.start_coordinator(pipeline_component)