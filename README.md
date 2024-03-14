# Singing Voice Conversion with GUI

## Useful commands for singularity container

```
singularity shell --nv --bind /public:/public ~/test.simg 
source ~/.bashrc
conda activate svc
python src/separation_train.py -t input_csv/train.csv -v input_csv/val.csv -e all
```