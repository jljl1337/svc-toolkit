from enum import Enum

class BaseEnum(str, Enum):
    @classmethod
    def has(cls, value):
        return value in cls._value2member_map_

class Precision(str, BaseEnum):
    BF16 = 'bf16'
    FP32 = '32'

class NeglectFrequency(str, Enum):
    NYQUIST = 'nyquist'
    ZERO = 'zero'

SEED = 56615230

CSV_SONG_COLUMN = 'song'
CSV_MIXTURE_PATH_COLUMN = 'mixture_path'
CSV_STEM_PATH_COLUMN = 'stem_path'

NYQUIST = 'nyquist'
ZERO = 'zero'
NEGLECT_FREQUENCY_OPTIONS = [NYQUIST, ZERO]