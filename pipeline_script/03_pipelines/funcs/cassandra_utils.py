from cassandra.cluster import Cluster
import funcs.misc as misc
import funcs.constants as constants
import re

# LOAD THE GLOBAL CONFIG & STITCH TOGETHER THE CASSANDRA CONNECTION STRING
global_config = constants.global_config()
CASSANDRA_BROKERS = [(global_config.endpoints.host, global_config.endpoints.ports.cassandra)]
VERBOSE = global_config.pipeline.verbose_logging

########################################################################################################
########################################################################################################

class create_instance:
    def __init__(self, HIDE_LOGS=False):
        cluster = Cluster(CASSANDRA_BROKERS)
        self.instance = cluster.connect()

        if HIDE_LOGS:
            global VERBOSE
            VERBOSE = False

    def __del__(self):
        self.instance.shutdown()

    ########################################################################################################
    ########################################################################################################

    # FREELY EXECUTE ANY CQL QUERY
    def query(self, query):
        try:
            return self.instance.execute(query)
        
        # SAFELY CATCH ERRORS
        except Exception as raw_error:
            parsed_error = self.parse_error(raw_error)
            raise Exception(f'[CASSANDRA] QUERY ERROR => {parsed_error}')

    ########################################################################################################
    ########################################################################################################
    
    # CASSANDRA DRIVER ERRORS ARE VERY MESSY
    # THIS ATTEMPT TO PARSE THEM TO MAKE EVERYTHING MORE HUMAN-READABLE
    def parse_error(self, error):
        stringified_error = str(error)
        
        # TRY TO REGEX MATCH THE ERROR PATTERN
        match = re.search(r'message="(.+)"', stringified_error)

        # MATCH FOUND, RETURN ISOLATED ERROR MSG
        if match:
            return match.group(1)
        
        # OTHERWISE, RETURN THE WHOLE THING
        return stringified_error
    
    ########################################################################################################
    ########################################################################################################

    # READ DATA FROM THE DATABASE
    def read(self, query: str, sort_by=False) -> list[dict]:
        try:
            container = []

            # PERFORM THE TABLE QUERY
            query_result = self.instance.execute(query)

            # PARSE EACH ROW AS A DICT
            for item in query_result:
                container.append(item._asdict())

            if VERBOSE: misc.log(f'[CASSANDRA] READ {len(container)} FROM DATABASE')
            
            # SORT BY KEY WHEN REQUESTED
            if sort_by:
                return sorted(container, key=lambda x: x[sort_by])

            # OTHERWISE, RETURN UNSORTED
            return container
        
        # SAFELY CATCH ERRORS
        except Exception as raw_error:
            parsed_error = self.parse_error(raw_error)
            raise Exception(f'[CASSANDRA] READ ERROR => {parsed_error}')
    
    ########################################################################################################
    ########################################################################################################

    # COUNT TABLE ROWS -- SELECT COUNT(*) FROM ...
    def count(self, count_query: str) -> int:
        return int(self.read(count_query)[0]['count'])

    ########################################################################################################
    ########################################################################################################

    # FULL DATABASE OVERVIEW (KEYSPACES)
    def write(self, keyspace_table: str, row: dict):
        try:

            # SPLIT THE KEYS & VALUES
            columns = list(row.keys())
            values = list(row.values())

            # STITCH TOGETHER THE QUERY STRING
            query_string = f'INSERT INTO {keyspace_table} ('
            query_string += ', '.join(columns)
            query_string += ') values ('
            query_string += ', '.join(['?'] * len(columns))
            query_string += ');'
            
            # CONSTRUCT A PREPARED STATEMENT & EXECUTE THE DB WRITE
            prepared_statement = self.instance.prepare(query_string)
            self.instance.execute(prepared_statement, values)

            if VERBOSE: misc.log('[CASSANDRA] WROTE TO DATABASE')
        
        # SAFELY CATCH ERRORS
        except Exception as raw_error:
            parsed_error = self.parse_error(raw_error)
            raise Exception(f'[CASSANDRA] WRITE ERROR => {parsed_error}')