import os
from argparse import ArgumentParser

from svc_toolkit.separation import constants
from svc_toolkit.separation.preprocess import moisesdb_mix, preprocess

def main():
    parser = ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default='./input_csv')
    parser.add_argument('-v', '--val_size', type=float, default=0.2)
    parser.add_argument('-s', '--stem', type=str, default='vocals')
    parser.add_argument('-m', '--musdb_dir', type=str, required=True)
    parser.add_argument('-M', '--moisesdb_dir', type=str, default='')
    parser.add_argument('-w', '--moisesdb_wav_dir', type=str, required=True)
    args = parser.parse_args()

    if args.moisesdb_dir != '':
        moisesdb_mix(args.moisesdb_dir, args.moisesdb_wav_dir, args.stem)

    os.makedirs(args.output_dir, exist_ok=True)

    preprocess(args.musdb_dir, args.moisesdb_wav_dir, args.val_size, args.output_dir, args.stem, constants.SEED)

if __name__ == "__main__":
    main()