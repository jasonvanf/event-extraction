# Fake Jionlp
# Python >= 3.9 cannot install

import time


def parse_time(text, time_base=time.time(), time_type='time_span'):
    result = {
        'time': text
    }
    return result


def parse_location(text):
    return text
