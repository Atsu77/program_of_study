import os
from pprint import pprint as pp
import subprocess

from config.settings import DATA_DIR, SUMO_CONFIG_DIR
from utils.processing import processing
from utils.machine_learning_model import autoencoder
from utils.simulation import create_sumo_config


DATA_PATH = os.path.join(DATA_DIR, 'veh_data.csv')

def create_train_and_test_data(time, lag):
    train_data = processing.SumoDataProcessor(
        DATA_PATH, time - lag, 20, processing.SumoDataAttribute.vehicle_acceleration)
    test_data = processing.SumoDataProcessor(
        DATA_PATH, time, 20, processing.SumoDataAttribute.vehicle_acceleration)
    return processing.DataProcessor().convert_autoencoder_input(
        train_data.accumulate_data_per_cell()), processing.DataProcessor().convert_autoencoder_input(
        test_data.accumulate_data_per_cell())


def run_sumo(simulation_time):
    cmd = [
        'sumo',
        '-n',
        os.path.join(SUMO_CONFIG_DIR, 'net.net.xml'),
        '-r',
        os.path.join(SUMO_CONFIG_DIR, 'route.rou.xml'),
        '--fcd-output.acceleration',
        '--fcd-output',
        os.path.join(DATA_DIR, 'veh_data.xml'),
        '-b',
        '0',
        '-e',
        str(simulation_time),
    ]
    subprocess.call(cmd)

def generate_config(simulation_time, obstacle_occurrence_time, obstacle_depart_pos, obstacle_depart_lane):
    simulation_time = simulation_time
    obstacle_occurrence_time = obstacle_occurrence_time
    obstacle_depart_pos = obstacle_depart_pos
    obstacle_depart_lane = obstacle_depart_lane
    edge = 'gneE0'
    row_generator = create_sumo_config.SumoRowGenerator(
        simulation_time,
        obstacle_occurrence_time,
        obstacle_depart_pos,
        obstacle_depart_lane,
        edge,
    )
    row_generator.generate_rou_file()

def xml_to_csv():
    xml2csv_path = os.path.join(os.environ['SUMO_HOME'], 'tools', 'xml', 'xml2csv.py')
    cmd = [
        'python',
        xml2csv_path,
        os.path.join(DATA_DIR, 'veh_data.xml'),
        '-s',
        ',',
        '-o',
        DATA_PATH
    ]
    subprocess.run(cmd)


def main():
    simulation_time = 1000
    obstacle_occurrence_time = simulation_time // 2
    obstacle_depart_pos = 2000
    obstacle_depart_lane = 1
    generate_config(simulation_time, obstacle_occurrence_time, obstacle_depart_pos, obstacle_depart_lane)
    run_sumo(simulation_time)
    xml_to_csv()

    time = 500
    cell_length = 20
    attribute = processing.SumoDataAttribute.vehicle_acceleration
    epochs = 1
    lag = 100
    sumo_data = processing.SumoDataProcessor(DATA_PATH, time, cell_length, attribute)

    input_dim = sumo_data.get_cell_count() * sumo_data.get_lane_count()
    hidden_dim = [100, 50, 100]
    output_dim = input_dim
    #for time in range(500, 1500, 10):
    train_data, test_data = create_train_and_test_data(time, lag)
    ae = autoencoder.Autoencoder(input_dim, hidden_dim, output_dim)
    ae.train(train_data, epochs)
    ae.predict(test_data)


if __name__ == '__main__':
    main()
