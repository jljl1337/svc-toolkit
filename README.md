# Singing Voice Conversion with GUI

![Tests](https://github.com/github/docs/actions/workflows/tests.yml/badge.svg)
[![Coverage](https://jljl1337.github.io/svc-toolkit/badges/coverage.svg)](https://github.com/jljl1337/svc-toolkit/actions)

## Useful commands for singularity container

```
singularity shell --nv --bind /public:/public ~/test.simg 
source ~/.bashrc
conda activate svc
python src/separation_train.py -t input_csv/train.csv -v input_csv/val.csv -e all
```