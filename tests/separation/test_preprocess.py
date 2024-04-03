import os
import shutil

import numpy as np

from vc_toolkit.separation.preprocess import mix_track, get_df, preprocess

CURRENT_DIR = os.path.dirname(__file__)

def test_mix_track():
    class DummyMoisesDBTrack:
        def __init__(self, id, artist, name, stems):
            self.id = id
            self.artist = artist
            self.name = name
            self.stems = stems

        def mix_stems(self, output_stems):
            waves = {}
            extra = 0
            for key in output_stems:
                waves[key] = np.ones((2, 44100 + extra))
                extra += 44100

            return waves

    mix_track(DummyMoisesDBTrack('id', 'artist', 'name', ['stem1', 'stem2']), 'stem1', CURRENT_DIR)

    assert os.path.exists(os.path.join(CURRENT_DIR, 'artist - name', 'mixture.wav'))
    assert os.path.exists(os.path.join(CURRENT_DIR, 'artist - name', 'stem1.wav'))

    shutil.rmtree(os.path.join(CURRENT_DIR, 'artist - name'))

def test_get_df():
    os.makedirs(os.path.join(CURRENT_DIR, 'test'), exist_ok=True)
    os.makedirs(os.path.join(CURRENT_DIR, 'test', 'song1'), exist_ok=True)
    os.makedirs(os.path.join(CURRENT_DIR, 'test', 'song2'), exist_ok=True)

    open(os.path.join(CURRENT_DIR, 'test', 'song1', 'mixture.wav'), 'w').close()
    open(os.path.join(CURRENT_DIR, 'test', 'song1', 'stem1.wav'), 'w').close()
    open(os.path.join(CURRENT_DIR, 'test', 'song2', 'mixture.wav'), 'w').close()
    open(os.path.join(CURRENT_DIR, 'test', 'song2', 'stem1.wav'), 'w').close()

    df = get_df(os.path.join(CURRENT_DIR, 'test'), 'stem1')

    assert len(df) == 2
    assert df.loc[0]['song'] == 'song1'
    assert df.loc[0]['mixture_path'] == os.path.join(CURRENT_DIR, 'test', 'song1', 'mixture.wav')
    assert df.loc[0]['stem_path'] == os.path.join(CURRENT_DIR, 'test', 'song1', 'stem1.wav')
    assert df.loc[1]['song'] == 'song2'
    assert df.loc[1]['mixture_path'] == os.path.join(CURRENT_DIR, 'test', 'song2', 'mixture.wav')
    assert df.loc[1]['stem_path'] == os.path.join(CURRENT_DIR, 'test', 'song2', 'stem1.wav')

    shutil.rmtree(os.path.join(CURRENT_DIR, 'test'))

def test_preprocess():
    os.makedirs(os.path.join(CURRENT_DIR, 'musdb', 'train'), exist_ok=True)
    os.makedirs(os.path.join(CURRENT_DIR, 'musdb', 'test'), exist_ok=True)

    os.makedirs(os.path.join(CURRENT_DIR, 'csv'), exist_ok=True)

    os.makedirs(os.path.join(CURRENT_DIR, 'musdb', 'train', 'song1'), exist_ok=True)
    os.makedirs(os.path.join(CURRENT_DIR, 'musdb', 'test', 'song2'), exist_ok=True)

    for i in range(50):
        os.makedirs(os.path.join(CURRENT_DIR, 'musdb', 'train', f'song{i}'), exist_ok=True)
        open(os.path.join(CURRENT_DIR, 'musdb', 'train', f'song{i}', 'mixture.wav'), 'w').close()
        open(os.path.join(CURRENT_DIR, 'musdb', 'train', f'song{i}', 'stem1.wav'), 'w').close()

    for i in range(50):
        os.makedirs(os.path.join(CURRENT_DIR, 'musdb', 'test', f'song{i}'), exist_ok=True)
        open(os.path.join(CURRENT_DIR, 'musdb', 'test', f'song{i}', 'mixture.wav'), 'w').close()
        open(os.path.join(CURRENT_DIR, 'musdb', 'test', f'song{i}', 'stem1.wav'), 'w').close()

    preprocess(os.path.join(CURRENT_DIR, 'musdb'), CURRENT_DIR, 0.2, os.path.join(CURRENT_DIR, 'csv'), 'stem1', 42)

    assert os.path.exists(os.path.join(CURRENT_DIR, 'csv', 'musdb_train.csv'))
    assert os.path.exists(os.path.join(CURRENT_DIR, 'csv', 'musdb_val.csv'))
    assert os.path.exists(os.path.join(CURRENT_DIR, 'csv', 'musdb_test.csv'))

    shutil.rmtree(os.path.join(CURRENT_DIR, 'musdb'))
    shutil.rmtree(os.path.join(CURRENT_DIR, 'csv'))