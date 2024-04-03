import os
import shutil

import pandas as pd

from svc_toolkit.separation.utility import get_best_checkpoint_path, get_last_checkpoint_path, get_checkpoint_path, save_song_list
from svc_toolkit.separation.constants import CSV_SONG_COLUMN

CURRENT_DIR = os.path.dirname(__file__)

def test_get_checkpoint_path():
    os.mkdir(os.path.join(CURRENT_DIR, 'test'))
    open(os.path.join(CURRENT_DIR, 'test', 'best_test.ckpt'), 'w').close()
    open(os.path.join(CURRENT_DIR, 'test', 'last_test.ckpt'), 'w').close()

    assert get_checkpoint_path(os.path.join(CURRENT_DIR, 'test'), 'best') == os.path.join(CURRENT_DIR, 'test', 'best_test.ckpt')
    assert get_checkpoint_path(os.path.join(CURRENT_DIR, 'test'), 'last') == os.path.join(CURRENT_DIR, 'test', 'last_test.ckpt')
    assert get_checkpoint_path(os.path.join(CURRENT_DIR, 'test'), 'random') == None

    shutil.rmtree(os.path.join(CURRENT_DIR, 'test'))

def test_get_best_checkpoint_path():
    os.mkdir(os.path.join(CURRENT_DIR, 'test'))
    open(os.path.join(CURRENT_DIR, 'test', 'best_test.ckpt'), 'w').close()
    open(os.path.join(CURRENT_DIR, 'test', 'last_test.ckpt'), 'w').close()

    assert get_best_checkpoint_path(os.path.join(CURRENT_DIR, 'test')) == os.path.join(CURRENT_DIR, 'test', 'best_test.ckpt')

    shutil.rmtree(os.path.join(CURRENT_DIR, 'test'))

def test_get_last_checkpoint_path():
    os.mkdir(os.path.join(CURRENT_DIR, 'test'))
    open(os.path.join(CURRENT_DIR, 'test', 'best_test.ckpt'), 'w').close()
    open(os.path.join(CURRENT_DIR, 'test', 'last_test.ckpt'), 'w').close()

    assert get_last_checkpoint_path(os.path.join(CURRENT_DIR, 'test')) == os.path.join(CURRENT_DIR, 'test', 'last_test.ckpt')

    shutil.rmtree(os.path.join(CURRENT_DIR, 'test'))

def test_save_song_list():

    song_list_path = os.path.join(CURRENT_DIR, 'song_list.csv')

    song_list_pd = pd.DataFrame({
        CSV_SONG_COLUMN: ['song1', 'song2']
    })

    song_list_pd.to_csv(song_list_path, index=False)

    save_song_list(song_list_path, CURRENT_DIR, 'song_list.csv')

    assert os.path.exists(os.path.join(CURRENT_DIR, 'song_list.csv'))

    os.remove(os.path.join(CURRENT_DIR, 'song_list.csv'))

