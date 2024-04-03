import os

import numpy as np
import pandas as pd
from torchmetrics.audio import SignalDistortionRatio, ScaleInvariantSignalDistortionRatio

from vc_toolkit.separation.evaluator import Evaluator
from vc_toolkit.separation.separator import Separator
from vc_toolkit.separation.constants import CSV_SONG_COLUMN, CSV_MIXTURE_PATH_COLUMN, CSV_STEM_PATH_COLUMN

CURRENT_DIR = os.path.dirname(__file__)

# Skip testing the Evaluator constructor
class MockEvaluator(Evaluator):
    def __init__(self, *args, **kwargs) -> None:
        pass

def test_evaluator_evaluate():
    class MockSeparator:
        def load_file(self, file):
            return np.zeros((2, 44100))

        def separate(self, wave):
            return np.ones((2, 44100)), 44100

    class MockMetric:
        def __call__(self, estimate, stem):
            return 0.0

    current_dir = os.path.dirname(__file__)
    test_model_dir = os.path.join(current_dir, '../test_model/')
    evaluator = MockEvaluator(test_model_dir, 'cpu', 'bf16', last=False)
    
    evaluator.separator = MockSeparator()
    evaluator.sdr = MockMetric()
    evaluator.sisdr = MockMetric()

    df = pd.DataFrame({
        CSV_SONG_COLUMN: ['song1', 'song2'],
        CSV_MIXTURE_PATH_COLUMN: ['mixture1', 'mixture2'],
        CSV_STEM_PATH_COLUMN: ['stem1', 'stem2']
    })
    df_path = os.path.join(current_dir, 'test.csv')
    df.to_csv(df_path, index=False)

    df_result = evaluator.evaluate(df_path)

    assert len(df_result) == 2
    assert df_result[CSV_SONG_COLUMN][0] == 'song1'
    assert df_result[CSV_SONG_COLUMN][1] == 'song2'

    os.remove(df_path)

def test_evaluator_summary():
    df = pd.DataFrame({
        'song': ['song1', 'song2'],
        'SDR': [1.0, 2.0],
        'SI-SDR': [3.0, 4.0],
        'NSDR': [5.0, 6.0],
        'NSI-SDR': [7.0, 8.0]
    })
    current_dir = os.path.dirname(__file__)
    test_model_dir = os.path.join(current_dir, '../test_model/')
    evaluator = MockEvaluator(test_model_dir, 'cpu', 'bf16', last=False)
    
    summary_df = evaluator.summary(df)

    assert len(summary_df) == 4
    assert summary_df.loc['SDR']['Mean'] == 1.5
    assert summary_df.loc['SI-SDR']['Min'] == 3
    assert summary_df.loc['NSDR']['Max'] == 6
    assert summary_df.loc['NSI-SDR']['Median'] == 7.5