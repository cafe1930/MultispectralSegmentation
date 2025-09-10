import argparse
from models import models_factories
import os
import pandas as pd
import json
import yaml

def create_training_environment(config_dict):
    pass

def train_lightning_module():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_config')

    sample_args = [
        '--path_to_config', ''
    ]
    args = parser.parse(sample_args)
    path_to_config = args.path_to_config

    # чтение файла конфигурации
    with open(path_to_config) as fd:
        if path_to_config.endswith('.yaml'):
            config_dict = yaml.load(fd)
        elif path_to_config.endswith('.json'):
            config_dict = json.load(fd)

