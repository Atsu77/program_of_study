import os
import random
import subprocess

import xml.etree.ElementTree as ET

import pandas as pd

from config.settings import SUMO_CONFIG_DIR, DATA_DIR, DATA_PATH
from utils.processing import processing


class SumoRowGenerator:
    MAX_SPEED_MPS = 27.78

    def __init__(self, simulation_time, obstacle_occurrence_time, obstacle_depart_pos, obstacle_depart_lane, edge):
        self.begin = "0"
        self.simulation_time = str(simulation_time)
        self.obstacle_occurrence_time = str(obstacle_occurrence_time)
        self.obstacle_depart_pos = str(obstacle_depart_pos)
        self.obstacle_depart_lane = str(obstacle_depart_lane)
        self.rou_file = "route.rou.xml"
        self.edge = str(edge)

    def generate_rou_file(self):
        root = ET.Element("routes")
        self.add_vehicle(root)
        self.add_obstacle(root)
        self.add_veh_flow(root)
        self.add_obstacle_flow(root)
        self.write_rou_file(root)

    def add_vehicle(self, root):
        v_type = ET.SubElement(root, 'vType')
        v_type.set('id', "vehicle")
        v_type.set('length', "5.0")
        v_type.set('width', "2.0")
        v_type.set('vClass', "passenger")
        v_type.set('maxSpeed', str(SumoRowGenerator.MAX_SPEED_MPS))

    def add_obstacle(self, root):
        v_type = ET.SubElement(root, 'vType')
        v_type.set('id', "obstacle")
        v_type.set('length', "5.0")
        v_type.set('width', "2.0")
        v_type.set('vClass', "passenger")
        v_type.set('maxSpeed', "0.000001")
        v_type.set('color', "1,1,0")
        v_type.set('lcStrategic', "0")
        v_type.set('lcCooperative', "0")
        v_type.set('lcKeepRight', "0")

    def add_veh_flow(self, root):
        flow = ET.SubElement(root, "flow")
        flow.set("id", "veh")
        flow.set("type", "vehicle")
        flow.set("begin", "0")
        flow.set("end", str(self.simulation_time))
        flow.set('departSpeed', "max")
        flow.set('departLane', "random")
        flow.set("probability", "1")
        ET.SubElement(flow, "route", edges=self.edge)

    def add_obstacle_flow(self, root):
        flow = ET.SubElement(root, "flow")
        flow.set("id", "obs")
        flow.set("type", "obstacle")
        flow.set("color", "1,1,0")
        flow.set("departPos", self.obstacle_depart_pos)
        flow.set("departLane", self.obstacle_depart_lane)
        flow.set("begin", self.obstacle_occurrence_time)
        flow.set("end", self.simulation_time)
        flow.set("period", "250")
        ET.SubElement(flow, "route", edges=self.edge)

    def write_rou_file(self, root):
        tree = ET.ElementTree(root)
        output_path = os.path.join(SUMO_CONFIG_DIR, self.rou_file)
        os.makedirs(os.path.dirname(SUMO_CONFIG_DIR), exist_ok=True)
        tree.write(output_path, encoding="utf-8", xml_declaration=True)

class SimurationExecutor:
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
        row_generator = SumoRowGenerator(
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

if __name__ == "__main__":
    simulation_time = 1000
    obstacle_occurrence_time = 500

    for _ in range(100):
        obstacle_depart_pos = random.randint(0, 4000)
        obstacle_depart_lane = random.randint(0, 2)
        SimurationExecutor.generate_config(simulation_time, obstacle_occurrence_time, obstacle_depart_pos, obstacle_depart_lane)
        SimurationExecutor.run_sumo(simulation_time)
        SimurationExecutor.xml_to_csv()
        df = pd.read_csv(DATA_PATH, usecols=['timestep_time', 'vehicle_id', 'vehicle_x', 'vehicle_lane', 'vehicle_acceleration'])
        processor = processing.SumoDataProcessor(df)
        processor.accumulate_data_per_cell()

