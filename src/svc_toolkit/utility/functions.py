import yaml

def load_yaml(path: str) -> dict[str, any]:
    with open(path) as file:
        data = yaml.safe_load(file)
    return data

def save_yaml(data: dict, save_path: str) -> None:
    with open(save_path, 'w') as file:
        yaml.safe_dump(data, file, sort_keys=False)
