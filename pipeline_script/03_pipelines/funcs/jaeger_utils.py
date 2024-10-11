from funcs import constants, misc
from jaeger_client import Config
from opentracing.propagation import Format

global_config = constants.global_config()

########################################################################################################
########################################################################################################

class create_instance:
    def __init__(self, service_name: str):

        config = Config(
            config={
                'sampler': {
                    'type': 'const', 
                    'param': 1
                },
                'logging': True,
                'local_agent': {
                    'reporting_host': global_config.endpoints.host,
                    'reporting_port': global_config.endpoints.ports.jaeger,
                },
            },
            service_name=service_name,
            validate=True,
        )

        self.tracer = config.initialize_tracer()

    ########################################################################################################
    ########################################################################################################

    def create_context(self, span):
        headers = {}
        self.tracer.inject(span.context, Format.HTTP_HEADERS, headers)
        return headers
    
    ########################################################################################################
    ########################################################################################################

    def create_span(self, span_name: str, predecessor=None):
        misc.log(span_name)

        # A PREDECESSOR WAS PROVIDED
        if predecessor:

            # A PREDECESSOR HEADER WAS PROVIDED
            if type(predecessor) == dict:
                predecessor_context = self.tracer.extract(Format.HTTP_HEADERS, predecessor)
                return self.tracer.start_span(span_name, child_of=predecessor_context)
            
            # A PROPER SPAN PREDECESSOR WAS PROVIDED
            return self.tracer.start_span(span_name, child_of=predecessor)

        # NO PREDECESSOR
        return self.tracer.start_span(span_name)
    
########################################################################################################
########################################################################################################