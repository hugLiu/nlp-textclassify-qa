import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs/config.yml')

DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models/pkl/')

DEFAULT_SUBMISSION_PATH = os.path.join(PROJECT_ROOT, 'results/csv/')

DEFAULT_ENCODING = 'utf-8'

YAML_VERSION = (1, 2)
