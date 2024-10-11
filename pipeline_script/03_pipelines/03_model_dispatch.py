from funcs import kafka_utils, redis_utils, mlflow_utils, jaeger_utils
from funcs import thread_utils, misc, constants, types

########################################################################################
########################################################################################

class pipeline_component:
    def __init__(self, structs):

        # CREATE INSTANCED CLIENTS
        self.kafka = kafka_utils.create_instance()
        self.redis = redis_utils.create_instance()
        self.mlflow = mlflow_utils.create_instance()
        self.jaeger = jaeger_utils.create_instance('MODEL_DISPATCH')

        # IN A BACKGROUND THREAD, DO...
        self.kafka.subscribe(constants.kafka.MODEL_DISPATCH, self.on_kafka_event, structs.thread_beacon)
        self.redis.subscribe(constants.redis.MODEL_PIPELINES, self.on_redis_change, structs.thread_beacon)
        # self.mlflow.subscribe(self.on_mlflow_change, structs.thread_beacon)

        # MOCK MLFLOW VERSION API
        self.mock_version_repo: str = 'model_versions'
        self.redis.subscribe(self.mock_version_repo, self.on_mlflow_change, structs.thread_beacon)

        # STATE FOR MOST UP-TO-DATE LOADED MODELS
        # NOTE THAT THREE PARALLEL THREADS WILL BE USING/MODIFYING THIS
        self.state = {}
        self.state_mutex = thread_utils.create_mutex()

    ########################################################################################
    ########################################################################################

    def on_kafka_event(self, kafka_input: dict):
        try:
            with self.state_mutex:
                assert 'trace' in kafka_input, "[INFERENCE] EVENT MISSING 'trace' PROPERTY"
                assert 'refined_stock_data' in kafka_input, "[INFERENCE] EVENT MISSING 'refined_stock_data' PROPERTY"

                trace_predecessor = kafka_input['trace']
                refined_stock_data = kafka_input['refined_stock_data']

                # FORCE STOCK SYMBOL TO LOWERCASE TO AVOID CAPITALIZATION ACCIDENTS
                # MAKE SURE AT LEAST ONE PIPE EXISTS
                stock_symbol = refined_stock_data['symbol'].lower()
                assert stock_symbol in self.state, f"[INFERENCE] NO PIPES EXIST FOR STOCK SYMBOL '{stock_symbol}'"

                # SHARD CONTAINER FOR PIPE PREDICTIONS
                prediction_batch = thread_utils.thread_safe_dict({
                    'predictions': {},
                    'models_used': [],
                    'last_span': None
                })

                threads = []

                # RUN EVERY UNIQUE PIPE IN A SEPARATE THREAD
                for pipe_name, model_sequence in self.state[stock_symbol].items():
                    threads.append(
                        thread_utils.start_thread(self.run_pipe, (
                            pipe_name,
                            model_sequence,
                            stock_symbol,
                            refined_stock_data,
                            prediction_batch,
                            trace_predecessor
                        ))
                    )
                
                # WAIT FOR ALL THREADS TO FINISH
                [thread.join() for thread in threads]

                # EVERY PIPE IS DONE -- PUSH PREDICTION BATCH TO DECISION SYNTHESIS
                def trigger_decision():
                    with self.jaeger.create_span("KAFKA: FORWARDING PREDICTION BATCH", prediction_batch['last_span']) as decision_span:
                        self.kafka.push(constants.kafka.DECISION_SYNTHESIS, {
                            'trace': self.jaeger.create_context(decision_span),
                            'prediction_batch': {
                                'input_row': refined_stock_data,
                                'pipe_predictions': prediction_batch['predictions']
                            }
                        })

                # PUSH MODEL METADATA TO DRIFT ANALYSIS
                def trigger_analysis():
                    with self.jaeger.create_span("KAFKA: FORWARDING MODEL METADATA", prediction_batch['last_span']) as drift_span:
                        self.kafka.push(constants.kafka.DRIFT_ANALYSIS, {
                            'trace': self.jaeger.create_context(drift_span),
                            'models_used': prediction_batch['models_used']
                        })

                # RUN BOTH PROCESSES IN PARALLEL
                thread_utils.start_thread(trigger_decision)
                thread_utils.start_thread(trigger_analysis)

        except AssertionError as error:
            with self.jaeger.create_span(f"PIPE USAGE FAILED: {error}", trace_predecessor) as span:
                pass

    # IN A SEPARATE THREAD -- INFER WITH MODEL PIPE
    def run_pipe(self, pipe_name, model_sequence, stock_symbol, init_value, prediction_batch, parent_span):
        local_predecessor = parent_span

        # MOCK INIT VALUE DURING DEVELOPMENT
        # latest_value = init_value
        latest_value = 'init_value'
        local_models = []

        # RECURSIVELY PREDICT WITH EACH MODEL
        # PREDECESSOR MODELS OUTPUT BECOMES SUCCESSOR MODELS INPUT
        for model in model_sequence:
            with self.jaeger.create_span(f"PREDICTING WITH '{stock_symbol}.{pipe_name}.{model.model_name}'", local_predecessor) as span:
                latest_value = model.predict(latest_value)
                local_predecessor = span
                local_models.append({
                    'model_name': model.model_name,
                    'model_version': model.version_number
                })

        # ADD THE PIPES CONTRIBUTION TO THE PREDICTION BATCH
        prediction_batch['predictions'][pipe_name] = latest_value
        prediction_batch['models_used'] += local_models
        prediction_batch['last_span'] = local_predecessor

    ########################################################################################
    ########################################################################################

    def on_mlflow_change(self, latest_model_versions: dict):
        with self.jaeger.create_span('MLFLOW MODEL VERSIONS HAVE CHANGED') as parent_span:
            with self.state_mutex:
                try:
                    assert isinstance(latest_model_versions, dict), '[MODEL STATE] REDIS VALUE WAS NOT A DICT'

                    threads = []

                    # COPY THE CURRENT STATE INTO A THREAD-SAFE CONTAINER
                    new_state = thread_utils.thread_safe_dict({
                        key: value for key, value in self.state.items()
                    })

                    # LOOP THROUGH EACH MODEL PIPE
                    for stock_symbol, stock_pipes in self.state.items():
                        for pipe_name, model_sequence in stock_pipes.items():

                            # AUDIT PIPE IN SEPARATE THREAD
                            threads.append(
                                thread_utils.start_thread(self.audit_pipe, (
                                    pipe_name,
                                    model_sequence, 
                                    latest_model_versions,
                                    stock_symbol,
                                    new_state,
                                    parent_span,
                                ))
                            )

                    # WAIT FOR ALL THREADS TO FINISH
                    [thread.join() for thread in threads]

                    # UPDATE STATE
                    self.state = new_state.dict
                    misc.log('[MODEL STATE] AUDIT FINISHED')

                except AssertionError as error:
                    with self.jaeger.create_span(f'AUDIT FAILED: {error}', parent_span) as span:
                        pass

    # IN A SEPARATE THREAD -- AUDIT MODEL PIPE
    def audit_pipe(self, pipe_name, model_sequence, latest_model_versions, stock_symbol, new_state, parent_span):
        with self.jaeger.create_span(f"AUDITING PIPE '{pipe_name}'", parent_span) as span:
            for nth_model, model in enumerate(model_sequence):

                # MAKE SURE THE MODEL STILL EXISTS
                assert model.model_name in latest_model_versions, f"[MODEL STATE] MODEL '{model.model_name}' NO LONGER EXISTS"

                ### TODO: ON ASSERT FAIL, THE PIPE IS BROKEN
                ### TODO: ON ASSERT FAIL, THE PIPE IS BROKEN

                # IF THE DEPLOYED MODEL IS BOUND TO A VERSION ALIAS
                # MAKE SURE THAT THE VERSION STILL MATCHES
                if model.version_alias:
                    
                    # MAKE SURE THE ALIAS STILL EXISTS
                    assert model.version_alias in latest_model_versions[model.model_name], f"[MODEL STATE] VERSION ALIAS '{model.version_alias}' NO LONGER EXISTS FOR MODEL '{model.model_name}'"
                    
                    ### TODO: ON ASSERT FAIL, THE PIPE IS BROKEN
                    ### TODO: ON ASSERT FAIL, THE PIPE IS BROKEN

                    old_version = model.version_number
                    new_version = latest_model_versions[model.model_name][model.version_alias]

                    # VERSION HAS CHANGED, SO UPDATE MODEL IN PIPELINE STATE
                    if old_version != new_version:
                        with self.jaeger.create_span(f"UPDATING MODEL '{stock_symbol}.{pipe_name}.{model.model_name}' VERSION ({old_version} -> {new_version})", span) as span:

                            new_model = self.mlflow.load_fake_model(model.model_name, model.version_alias, new_version)
                            new_state[stock_symbol][pipe_name][nth_model] = new_model

                # OTHERWISE, VERIFY THAT THE NUMERIC VERSION STILL EXISTS
                # else:
                #     assert self.mlflow.model_exists(model.model_name, model.version_number), f"[MODEL STATE] MODEL '{model.model_name}' VERSION {model.version_number} NO LONGER EXISTS"

    ########################################################################################
    ########################################################################################

    def on_redis_change(self, latest_model_pipelines: dict):
        with self.jaeger.create_span('MODEL PIPELINE STRUCTURES HAVE CHANGED') as parent_span:
            with self.state_mutex:
                try:
                    assert isinstance(latest_model_pipelines, dict), '[PIPELINE STATE] REDIS VALUE WAS NOT A DICT'

                    # LATER: FETCH UPDATED LIST OF MLFLOW MODELS
                    # latest_model_versions = self.mlflow.list_all_models()
                    latest_model_versions = self.redis.get(self.mock_version_repo)

                    # CONSTRUCT MODEL PIPELINES FROM JSON DATA
                    for stock_symbol, stock_pipes in latest_model_pipelines.items():
                        pipes = thread_utils.thread_safe_dict()
                        threads = []

                        for pipe_name, model_sequence in stock_pipes.items():
                            threads.append(
                                thread_utils.start_thread(self.deploy_pipe, (
                                    pipe_name, 
                                    model_sequence,
                                    stock_symbol, 
                                    latest_model_versions, 
                                    pipes, 
                                    parent_span,
                                ))
                            )

                        # WAIT FOR EVERY THREAD TO FINISH
                        [thread.join() for thread in threads]

                        # UPDATE THE STOCK SYMBOLS MODEL STATE
                        self.state[stock_symbol] = pipes.dict

                    misc.log('[PIPELINE STATE] AUDIT FINISHED')

                except AssertionError as error:
                    with self.jaeger.create_span(f'AUDIT FAILED: {error}', parent_span) as span:
                        pass
    
    # IN A SEPARATE THREAD -- DEPLOY MODEL PIPES FOR STOCK
    def deploy_pipe(self, pipe_name, model_sequence, stock_symbol, latest_model_versions, pipes, parent_span):
        local_predecessor = parent_span
        models = []

        # CONSTRUCT REQUESTED MODEL
        for block in model_sequence:
            assert 'model_name' in block, f"[PIPELINE STATE] MODEL PROPERTY 'model_name' MISSING"
            assert 'model_version' in block, f"[PIPELINE STATE] MODEL PROPERTY 'model_version' MISSING"

            model_name = block['model_name']
            version_number = block['model_version']
            version_alias = None

            # MAKE SURE MODEL EXISTS IN MLFLOW
            assert model_name in latest_model_versions, f"[PIPELINE STATE] MODEL '{model_name}' DOES NOT EXIST"

            # AUDIT -- IF A STRINGIFIED VERSION ALIAS WAS PROVIDED
            if type(version_number) == str:
                assert version_number in latest_model_versions[model_name], f"[PIPELINE STATE] MODEL ALIAS '{version_number}' DOES NOT EXIST FOR MODEL '{model_name}'"

                # ALIAS EXISTS, UPDATE VERSION PROPS
                version_alias = version_number
                version_number = latest_model_versions[model_name][version_alias]

            # MAKE SURE MODEL VERSION STILL EXISTS
            # assert self.mlflow.model_exists(model_name, version_number), f"[PIPELINE CHANGE] MODEL '{model_name}' VERSION {version_number} DOES NOT EXISTS"

            # LOAD IN FAKE MODELS -- FOR TESTING
            with self.jaeger.create_span(f"DEPLOYING MODEL '{stock_symbol}.{pipe_name}.{model_name}' VERSION {version_number}", local_predecessor) as span:
                model = self.mlflow.load_fake_model(model_name, version_alias, version_number)
                models.append(model)
                local_predecessor = span

        # FINISHED ONE PIPE
        pipes[pipe_name] = models

########################################################################################
########################################################################################

thread_utils.start_coordinator(pipeline_component)