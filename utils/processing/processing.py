import enum
import math
import numpy as np
from typing import Union
import pandas as pd
from pprint import pprint as pp
from config.settings import DATA_PATH


class SumoDataAttribute:
    timestep_time = "timestep_time"
    vehicle_acceleration = "vehicle_acceleration"
    vehicle_angle = "vehicle_angle"
    vehicle_id = "vehicle_id"
    vehicle_lane = "vehicle_lane"
    vehicle_pos = "vehicle_pos"
    vehicle_slope = "vehicle_slope"
    vehicle_speed = "vehicle_speed"
    vehicle_type = "vehicle_type"
    vehicle_x = "vehicle_x"
    vehicle_y = 'vehicle_y'


class SumoDataProcessor:
    def __init__(
            self,
            veh_data: pd.DataFrame,
            cell_length: int,
            attribute: SumoDataAttribute) -> None:
        self._veh_data = veh_data
        self._veh_data = self.rename_lane()
        self._cell_length = cell_length
        self._attribute = attribute
        self._divided_road = self.divide_road_into_cells()

    @property
    def veh_data(self):
        return self._veh_data

    def rename_lane(self) -> int:
        veh_data_copy = self._veh_data.copy()
        renamed_lane_col = veh_data_copy[SumoDataAttribute.vehicle_lane].apply(
            lambda x: x[-1])
        veh_data_copy[SumoDataAttribute.vehicle_lane] = renamed_lane_col
        return veh_data_copy

    def get_simulation_time(self) -> int:
        return self._veh_data[SumoDataAttribute.timestep_time].max()

    def get_lane_count(self) -> int:
        return len(self._veh_data[SumoDataAttribute.vehicle_lane].unique())

    def get_veh_data_at_time(self, time) -> pd.DataFrame:
        return self._veh_data[self._veh_data[SumoDataAttribute.timestep_time] == time]

    def get_veh_data_by_pos_range(
            self,
            start_pos: float,
            end_pos: float) -> pd.DataFrame:
        return self._veh_data[(self._veh_data[SumoDataAttribute.vehicle_x] >= start_pos) & (
            self._veh_data[SumoDataAttribute.vehicle_x] <= end_pos)]

    def get_cell_count(self) -> int:
        return math.ceil(self.get_road_length() / self._cell_length)

    def add_cell_column(self) -> pd.DataFrame:
        veh_data_copy = self._veh_data.copy()
        veh_data_copy["cell"] = (veh_data_copy[SumoDataAttribute.vehicle_x] // self._cell_length).astype(int)
        return veh_data_copy

    def initialize_cells(self, round_road_length):
        cells = {}
        for cell, _ in enumerate(
                range(0, round_road_length, self._cell_length)):
            cells[cell] = 0
        return cells

    def initialize_data_by_cell(self) -> dict:
        round_road_length = self.get_road_length()
        lane_count = self.get_lane_count()
        data_by_cell = {}
        for lane in range(0, lane_count):
            data_by_cell[lane] = self.initialize_cells(round_road_length)
        return data_by_cell

    def get_road_length(self):
        road_length = self.get_road_length()
        round_digits = 10 ** (len(str(road_length)) - 1)
        return math.ceil(road_length / round_digits) * round_digits

    def get_obstacle_location(self):
        obstacle_data = self._veh_data[self._veh_data[SumoDataAttribute.vehicle_type]
                                       == "obstacle"].iloc[0]
        if obstacle_data.empty:
            return None, None
        cell = int(
            obstacle_data[SumoDataAttribute.vehicle_x] // self._cell_length)
        lane = int(obstacle_data[SumoDataAttribute.vehicle_lane])
        return cell, lane

    def create_obstacle_occurrence_place_true_label(self, expansion: int = 5):
        true_label = self.initialize_data_by_cell()
        cell, lane = self.get_obstacle_location()
        for i in range(-expansion, expansion + 1):
            true_label[lane][cell + i] = 1
        return true_label


class DataProcessor:
    def convert_dict_to_dataframe(self, data_by_cell: dict) -> pd.DataFrame:
        return pd.DataFrame(data_by_cell)

    def convert_dataframe_to_numpy_array(
            self, data: pd.DataFrame) -> np.ndarray:
        return data.to_numpy()

    def convert_dict_to_numpy_array(self, data_by_cell: dict) -> np.ndarray:
        data = self.convert_dict_to_dataframe(data_by_cell)
        return self.convert_dataframe_to_numpy_array(data)

    def convert_autoencoder_input(
            self, data_by_cell: Union[dict, pd.DataFrame, np.ndarray]) -> np.ndarray:
        def to_numpy(data):
            if isinstance(data, dict):
                return self.convert_dict_to_numpy_array(data)
            elif isinstance(data, pd.DataFrame):
                return self.convert_dataframe_to_numpy_array(data)
            elif isinstance(data, np.ndarray):
                pass
            else:
                raise TypeError('data_by_cell type is not supported')
        converted_data = to_numpy(data_by_cell)
        return converted_data.reshape(
            1, converted_data.shape[0] * converted_data.shape[1])


if __name__ == '__main__':
    cell_length = 20
    vehicle_attibute = SumoDataAttribute.vehicle_acceleration
    df = pd.read_csv(DATA_PATH)
    sumo_data_processor = SumoDataProcessor(df, cell_length, vehicle_attibute)
    r = sumo_data_processor.divide_road_into_cells()
    pp(r)
