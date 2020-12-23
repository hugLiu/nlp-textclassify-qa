import os

from utils import io
from constants import DEFAULT_CONFIG_PATH

class Config:
    def __init__(self):
        super(Config, self).__init__()

    @staticmethod
    def yaml_config(config=None):
        file_config = {}
        if config is None and os.path.isfile(DEFAULT_CONFIG_PATH):
            config = DEFAULT_CONFIG_PATH

        if config is not None:
            try:
                file_config = io.read_config_file(config)
            except Exception as e:
                raise ValueError(
                    'Failed to read configuration file "{}". '
                    'Error:{}.'
                        .format(config, e)
                )

        list_config = []
        dict_config = {}
        if 'default' in file_config.keys():
            list_config.extend(file_config['default'])

        if 'run_not_debug' in file_config.keys():
            run_not_debug = file_config['run_not_debug']
            dict_config = {'run_not_debug': run_not_debug}

            if run_not_debug and 'run' in file_config.keys():
                list_config.extend(file_config['run'])
            if not run_not_debug and 'debug' in file_config.keys():
                list_config.extend(file_config['debug'])

        if list_config:
            for item in list_config:
                for key, value in item.items():
                    dict_config[key] = value
            return dict_config
        return None
