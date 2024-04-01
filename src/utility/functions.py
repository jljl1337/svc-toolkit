import yaml

def load_yaml(path):
    with open(path) as file:
        data = yaml.safe_load(file)
    return data

def save_yaml(data, save_path):
    with open(save_path, 'w') as file:
        yaml.safe_dump(data, file, sort_keys=False)
