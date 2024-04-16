from enum import Enum

class BaseStrEnum(str, Enum):
    @classmethod
    def has(cls, value):
        return value in cls._value2member_map_

    @classmethod
    def all(cls):
        return [member.value for name, member in cls.__members__.items()]

class Precision(BaseStrEnum):
    BF16 = 'bf16'
    FP32 = 'fp32'

class NeglectFrequency(BaseStrEnum):
    NYQUIST = 'nyquist'
    ZERO = 'zero'

class CSVColumns(BaseStrEnum):
    SONG = 'song'
    MIXTURE_PATH = 'mixture_path'
    STEM_PATH = 'stem_path'

SEED = 56615230

# CSV_SONG_COLUMN = 'song'
# CSV_MIXTURE_PATH_COLUMN = 'mixture_path'
# CSV_STEM_PATH_COLUMN = 'stem_path'

# NYQUIST = 'nyquist'
# ZERO = 'zero'
# NEGLECT_FREQUENCY_OPTIONS = [NYQUIST, ZERO]