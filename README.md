# Singing Voice Conversion with GUI

![Tests](https://github.com/jljl1337/svc-toolkit/actions/workflows/tests.yml/badge.svg)
![Deployment](https://github.com/jljl1337/svc-toolkit/actions/workflows/deployments.yml/badge.svg)
[![Codecov](https://codecov.io/gh/jljl1337/svc-toolkit/graph/badge.svg?token=QBM6OLIG00)](https://codecov.io/gh/jljl1337/svc-toolkit)

## Useful commands for singularity container

```
singularity shell --nv --bind /public:/public ~/test.simg 
source ~/.bashrc
conda activate svc
python src/separation_train.py -t input_csv/train.csv -v input_csv/val.csv -e all
```