"""Keep Windows awake (no sleep, no display off) while the pipeline runs.
Run as a background process; kill it when done. Does NOT change power settings permanently.
"""
import ctypes, time, sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

ctypes.windll.kernel32.SetThreadExecutionState(
    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
print("keep_awake: sleep/display-off inhibited", flush=True)
try:
    while True:
        time.sleep(60)
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
except KeyboardInterrupt:
    pass
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print("keep_awake: released", flush=True)
