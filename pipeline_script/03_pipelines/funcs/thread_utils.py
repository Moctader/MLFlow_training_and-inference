from threading import Thread, Semaphore, Lock
from funcs import misc, constants
import time

global_config = constants.global_config()

########################################################################################################
########################################################################################################

def create_mutex():
    return Semaphore(1)

########################################################################################################
########################################################################################################

# THREAD LOCK TO KILL HELPER THREADS
class create_thread_beacon:
    def __init__(self):
        self.lock = True

    def __del__(self):
        self.kill()
    
    def is_active(self):
        return self.lock
    
    def kill(self):
        self.lock = False

########################################################################################################
########################################################################################################

# THREAD-SAFE COUNTER
class create_counter:
    def __init__(self, announce_every: int):
        self.value = 0
        self.mutex = create_mutex()

        assert isinstance(announce_every, int), '[COUNTER] ANNOUNCEMENT MUST BE OF TYPE INT'
        assert announce_every > 0, '[COUNTER] ANNOUNCEMENT BREAKPOINT MUST BE >0'
        self.announce_every: int = announce_every

    def __del__(self):
        misc.log(f'[COUNTER] ENDED AT {self.value}')
    
    def current(self):
        with self.mutex:
            return self.value
    
    def increment(self):
        with self.mutex:
            self.value += 1

            if (self.value % self.announce_every) == 0:
                misc.log(f'[COUNTER] HAS REACHED {self.value}')

########################################################################################################
########################################################################################################

def start_coordinator(pipeline_component, poll=True):

    # CREATE A PROCESS BEACON TO BIND THREAD ROUTINES
    thread_beacon = create_thread_beacon()

    try:
        misc.log('[COORDINATOR] LAUNCHED')

        # WRAP SHARED HELPER STRUCTS INTO A NAMESPACE
        helper_structs = misc.TO_NAMESPACE({
            'thread_beacon': thread_beacon,
            'global_config': global_config
        })

        # INSTANTIATE THE PIPELINE STATE
        state = pipeline_component(helper_structs)

        if poll:
            while thread_beacon.is_active():
                time.sleep(global_config.pipeline.polling_cooldown)

        misc.log('[COORDINATOR] TERMINATED')

    except AssertionError as error:
        misc.log(f'{error}')
    
    # KILL ALL HELPER-THREADS WHEN MAIN THREAD DIES
    except KeyboardInterrupt:
        misc.log('[COORDINATOR] MANUALLY INTERRUPTED..', True)
        thread_beacon.kill()

    except Exception as error:
        misc.log(f'{error}')

########################################################################################################
########################################################################################################

def start_thread(func, _args=(), _daemon=False):
    thread = Thread(target=func, args=_args)
    thread.daemon = _daemon
    thread.start()

    return thread

########################################################################################################
########################################################################################################

class thread_safe_dict:
    def __init__(self, default_value={}):
        self.dict = default_value
        self.lock = Lock()

    def keys(self):
        with self.lock:
            return list(self.dict.keys())
            
    def __getitem__(self, key):
        with self.lock:
            return self.dict[key]

    def __setitem__(self, key, value):
        with self.lock:
            self.dict[key] = value

    def __delitem__(self, key):
        with self.lock:
            del self.dict[key]

########################################################################################################
########################################################################################################