import os

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
WORKING_DIR = os.path.abspath(os.path.join(CURRENT_DIR, os.pardir))
DATA_DIR = os.path.join(WORKING_DIR, 'data')
DATA_PATH = os.path.join(DATA_DIR, 'veh_data.csv')

SUMO_CONFIG_DIR = os.path.join(WORKING_DIR, 'sumo_config')
