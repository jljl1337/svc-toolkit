from enum import Enum

class BaseStrEnum(str, Enum):
    @classmethod
    def has(cls, value) -> bool:
        return value in cls._value2member_map_

    @classmethod
    def all(cls) -> list[str]:
        return [member.value for _name, member in cls.__members__.items()]

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

class ConfigKeys(BaseStrEnum):
    SAMPLE_RATE = 'sample_rate'
    WIN_LENGTH = 'win_length'
    HOP_LENGTH = 'hop_length'
    PATCH_LENGTH = 'patch_length'
    NEGLECT_FREQUENCY = 'neglect_frequency'

    EPOCHS = 'epochs'
    BATCH_SIZE = 'batch_size'
    LOADER_NUM_WORKERS = 'loader_num_workers'
    DETERMINISTIC = 'deterministic'
    PRECISION = 'precision'

    EXPAND_FACTOR = 'expand_factor'
    LEARNING_RATE = 'learning_rate'
    WEIGHT_DECAY = 'weight_decay'
    OPTIMIZER = 'optimizer'
    DEEPER = 'deeper'

class LoggerDFColumns(BaseStrEnum):
    EPOCH = 'epoch'
    TRAIN_LOSS = 'train_loss'
    VAL_LOSS = 'val_loss'

SEED = 56615230