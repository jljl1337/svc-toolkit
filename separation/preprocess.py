import os

import pandas as pd
from sklearn.model_selection import train_test_split
from moisesdb.dataset import MoisesDB
from moisesdb.track import MoisesDBTrack
from moisesdb.defaults import all_stems, default_sample_rate

import constants
import audio

def moisesdb_preprocess(root, save_dir, stem):
    db = MoisesDB(root)
    os.makedirs(save_dir, exist_ok=True)
    print('yo')

    track: MoisesDBTrack
    for track in db:
        if stem in track.stems:
            track_dir = f'{track.artist} - {track.name}'

            output_stems = {
                "mixture": all_stems,
                stem: [stem],
            }
            waves = track.mix_stems(output_stems)

            if waves['mixture'].shape != waves[stem].shape:
                print(f"{track.artist} - {track.name}")
                print(track.id)
                print(waves['mixture'].shape, waves[stem].shape)

            os.makedirs(os.path.join(save_dir, track_dir), exist_ok=True)
            mixture_path = os.path.join(save_dir, track_dir, 'mixture.wav')
            stem_path = os.path.join(save_dir, track_dir, f'{stem}.wav')

            audio.save(mixture_path, waves['mixture'].T, default_sample_rate)
            audio.save(stem_path, waves[stem].T, default_sample_rate)
        

def get_df(root, stem):
    df = pd.DataFrame(columns=['mixture_path', 'stem_path'])

    for dir in os.listdir(root):
        mixture_path = os.path.join(root, dir, 'mixture.wav')
        stem_path = os.path.join(root, dir, f'{stem}.wav')

        df.loc[len(df)] = [mixture_path, stem_path]
    
    return df

def main():
    # moisesdb_preprocess('/home/jljl1337/dataset/moisesdb/', '/home/jljl1337/dataset/moisesdb_wav', 'vocals')
    df = get_df('/home/jljl1337/dataset/musdb18hq/train', 'vocals')
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=constants.SEED)
    df_train.to_csv('train.csv', index=False)
    df_val.to_csv('val.csv', index=False)

if __name__ == "__main__":
    main()