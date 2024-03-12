import os
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from moisesdb.dataset import MoisesDB
from moisesdb.track import MoisesDBTrack
from moisesdb.defaults import all_stems, default_sample_rate

import constants
import audio

def mix_track(track, stem, save_dir):
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
            min_len = min(waves['mixture'].shape[1], waves[stem].shape[1])
            waves['mixture'] = waves['mixture'][:, :min_len]
            waves[stem] = waves[stem][:, :min_len]
            print(waves['mixture'].shape, waves[stem].shape)

        os.makedirs(os.path.join(save_dir, track_dir), exist_ok=True)
        mixture_path = os.path.join(save_dir, track_dir, 'mixture.wav')
        stem_path = os.path.join(save_dir, track_dir, f'{stem}.wav')

        audio.save(mixture_path, waves['mixture'].T, default_sample_rate)
        audio.save(stem_path, waves[stem].T, default_sample_rate)

def moisesdb_mix(root, save_dir, stem):
    db = MoisesDB(root)
    # db = [db[i] for i in [52, 71, 95, 113, 117, 167, 177, 220]]
    os.makedirs(save_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        futures = {executor.submit(mix_track, track, stem, save_dir): track for track in db}
        for future in tqdm(as_completed(futures), total=len(db), desc="Processing tracks"):
        # for future in as_completed(futures):
            pass

def get_df(root, stem):
    df = pd.DataFrame(columns=['song', 'mixture_path', 'stem_path'])

    for dir in os.listdir(root):
        mixture_path = os.path.join(root, dir, 'mixture.wav')
        stem_path = os.path.join(root, dir, f'{stem}.wav')

        df.loc[len(df)] = [dir, mixture_path, stem_path]
    
    return df

def main():
    # moisesdb_mix('/home/jljl1337/dataset/moisesdb/', '/home/jljl1337/dataset/moisesdb_wav', 'vocals')
    df_musdb = get_df('/home/jljl1337/dataset/musdb18hq/train', 'vocals')
    df_musdb_train, df_musdb_val = train_test_split(df_musdb, test_size=0.2, random_state=constants.SEED)
    df_musdb_train.to_csv('musdb_train.csv', index=False)
    df_musdb_val.to_csv('musdb_val.csv', index=False)
    df_musdb_test = get_df('/home/jljl1337/dataset/musdb18hq/test', 'vocals')
    df_musdb_test.to_csv('musdb_test.csv', index=False)

    df_moisesdb = get_df('/home/jljl1337/dataset/moisesdb_wav', 'vocals')
    df_moisesdb_rest, df_moisesdb_test = train_test_split(df_moisesdb, test_size=0.2, random_state=constants.SEED)
    df_moisesdb_train, df_moisesdb_val = train_test_split(df_moisesdb_rest, test_size=0.125, random_state=constants.SEED)
    df_moisesdb_train.to_csv('moisesdb_train.csv', index=False)
    df_moisesdb_val.to_csv('moisesdb_val.csv', index=False)
    df_moisesdb_test.to_csv('moisesdb_test.csv', index=False)
    # print(len(df_moisesdb))
    # print(len(df_moisesdb_train))
    # print(len(df_moisesdb_val))
    # print(len(df_moisesdb_test))

    df_train = pd.concat([df_musdb_train, df_moisesdb_train])
    df_train.to_csv('train.csv', index=False)
    df_val = pd.concat([df_musdb_val, df_moisesdb_val])
    df_val.to_csv('val.csv', index=False)
    df_test = pd.concat([df_musdb_test, df_moisesdb_test])
    df_test.to_csv('test.csv', index=False)

if __name__ == "__main__":
    main()