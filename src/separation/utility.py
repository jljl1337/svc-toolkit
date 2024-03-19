import os

import yaml
import pandas as pd

def load_yaml(path):
    with open(path) as file:
        data = yaml.safe_load(file)
    return data

def save_yaml(data, save_path):
    with open(save_path, 'w') as file:
        yaml.safe_dump(data, file, sort_keys=False)

def get_best_checkpoint_path(model_dir):
    return get_checkpoint_path(model_dir, 'best')

def get_last_checkpoint_path(model_dir):
    return get_checkpoint_path(model_dir, 'last')

def get_checkpoint_path(model_dir: str, prefix: str):
    for file in os.listdir(model_dir):
        if file.startswith(prefix) and file.endswith('.ckpt'):
            return os.path.join(model_dir, file)
    return None

def save_song_list(csv_path, model_dir, file_name):
    song_list_pd = pd.read_csv(csv_path)

    # Select only the song column
    song_list_pd = song_list_pd[['song']]

    song_list_pd.to_csv(os.path.join(model_dir, file_name), index=False)