from confluent_kafka import Consumer, Producer
from funcs import misc, constants, thread_utils
import json, re

global_config = constants.global_config()
KAFKA_BROKERS = f'{global_config.endpoints.host}:{global_config.endpoints.ports.kafka}'

########################################################################################################
########################################################################################################

class create_instance:
    def __init__(self):
        
        # CREATE PRODUCER & CHECK CONNECTION
        self.producer = Producer({ 'bootstrap.servers': KAFKA_BROKERS })
        assert self.check_connection(), f'COULD NOT CONNECT TO KAFKA SERVERS'

    def __del__(self):
        try:
            # KILL ANY ACTIVE CONSUMER
            if hasattr(self, 'consumer'):
                self.consumer.close()

            misc.log('[KAFKA] INSTANCE TERMINATED')

        # PREVENTS THROWN ERRORS FOR INGESTION SCRIPTS
        except ImportError:
            pass

    ########################################################################################################
    ########################################################################################################

    def check_connection(self):
        try:
            metadata = self.producer.list_topics(timeout=0.5)
            return True
        except:
            return False

    ########################################################################################################
    ########################################################################################################

    # JSON/DICT => BYTES
    def json_serializer(self, json_data: dict) -> list[bool, bytes|str]:
        try:
            return json.dumps(json_data).encode('UTF-8')
        
        except Exception as error:
            misc.log(f'[KAFKA] SERIALIZATION ERROR: {error}')

    # BYTES => JSON DICT
    def json_deserializer(self, raw_bytes: bytes) -> list[bool, dict|str]:
        try:
            return json.loads(raw_bytes.decode('UTF-8'))
        
        except Exception as error:
            misc.log(f'[KAFKA] DESERIALIZATION ERROR: {error}')

    ########################################################################################################
    ########################################################################################################

    def push(self, topic_name: str, data_dict: dict):
        assert isinstance(topic_name, str), '[KAFKA] TOPIC NAME MUST BE A STRING'
        assert isinstance(data_dict, dict), '[KAFKA] VALUE MUST BE A DICT'

        # TRY TO CONVERT THE DICT TO BYTES
        bytes_data = self.json_serializer(data_dict)

        # THEN PUSH THE MESSAGE TO A KAFKA TOPIC
        self.producer.produce(
            topic_name, 
            value=bytes_data,
            on_delivery=self._push_callback,
        )

        # ASYNC ACKNOWLEDGE
        if global_config.pipeline.kafka.async_producer_ack:
            self.producer.poll(1)
            return
        
        # OTHERWISE, ACKNOWLEDGE SYNCHRONOUSLY
        self.producer.flush()

    def _push_callback(self, error, message):
        if error:
            misc.log(f'[KAFKA ERROR] {error}')
            return
    
        misc.log(f'[KAFKA] PUSHED EVENT ({message.topic()})')

    ########################################################################################################
    ########################################################################################################

    def subscribe(self, kafka_topics, callback_func, process_beacon):
        assert isinstance(kafka_topics, (str, list)), '[KAFKA] TOPICS MUST BE OF TYPE STR OR LIST[STR]'

        # BLOCK MULTI-SUBSCRIPTION ATTEMPTS
        # IF YOU NEED THIS, PROVIDE MULTIPLE TOPICS IN THE 'kafka_topics' ARG
        assert not hasattr(self, 'consumer'), '[KAFKA] YOU ARE LIMITED TO ONE CONSUMER'

        # ARE WE SUBSCRIBING TO ONE TOPIC OR MORE? FORMAT ACCORDINGLY
        kafka_topics = [kafka_topics] if type(kafka_topics) == str else kafka_topics

        # CREATE THE CONSUMER CLIENT
        self.consumer = Consumer({
            'bootstrap.servers': KAFKA_BROKERS,
            'group.id': ','.join(kafka_topics) + '.consumers',
            'enable.auto.commit': global_config.pipeline.kafka.consumer_auto_commit,
            'on_commit': self._consume_callback,
            'auto.offset.reset': global_config.pipeline.kafka.consumer_stategy,
        })

        # FINALLY, SUBSCRIBE TO THE PROVIDED KAFKA TOPIC
        self.consumer.subscribe(kafka_topics, self._assigned, self._revoked, self._lost)

        def consume_events():
            misc.log(f'[KAFKA] STARTED POLLING ({kafka_topics})')

            # KEEP POLLING WHILE THE MAIN THREAD LIVES
            while process_beacon.is_active():
                try:
                    # POLL NEXT MESSAGE
                    event = self.consumer.poll(global_config.pipeline.polling_cooldown)

                    # SKIP EMPTY MESSAGES
                    if event is None:
                        continue

                    # CATCH & PARSE KAFKA ERRORS
                    if event.error():
                        match = re.search(r'str="([^"]+)"', str(event.error()))
                        if match:
                            error_message = match.group(1)
                            misc.log(f'[KAFKA] EVENT ERROR: {error_message}')
                            continue

                    # IF AUTO-COMMITTING IS DISABLED
                    # COMMIT THE EVENT TO PREVENT OTHERS FROM TAKING IT
                    if not global_config.pipeline.kafka.consumer_auto_commit:
                        self.consumer.commit(event, asynchronous=global_config.pipeline.kafka.async_consumer_commit)

                    # DESERIALIZE THE MESSAGE, AND CHECK WHICH TOPIC IT CAME FROM
                    deserialized_dict: dict = self.json_deserializer(event.value())
                    topic_name: str = event.topic()
                    misc.log(f'[KAFKA] EVENT RECEIVED ({topic_name})')

                    # ATTEMPT TO RUN THE CALLBACK FUNC
                    try:
                        callback_func(deserialized_dict)
                    except Exception as error:
                        misc.log(f'[KAFKA] CALLBACK ERROR: {error}')

                except Exception as error:
                    misc.log(f'[KAFKA] CONSUMER ERROR: {error}')

        # START CONSUMING EVENTS IN A BACKGROUND THREAD
        thread_utils.start_thread(consume_events)

    def _consume_callback(self, error, partitions):
        if error:
            misc.log(f'[KAFKA] CONSUMER ACK ERROR: {error}')

    def _assigned(self, consumer, partition_data):
        partitions = [p.partition for p in partition_data]
        misc.log(f'[KAFKA] CONSUMER ASSIGNED PARTITIONS: {partitions}')

    def _revoked(self, consumer, partition_data):
        partitions = [p.partition for p in partition_data]
        misc.log(f'[KAFKA] CONSUMER PARTITIONS REVOKED: {partitions}')

    def _lost(self, consumer, partition_data):
        misc.log(f'[KAFKA] CONSUMER PARTITIONS LOST: {consumer} {partition_data}')

    ########################################################################################################
    ########################################################################################################