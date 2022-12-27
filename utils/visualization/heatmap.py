import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Visualizer:
    def __init__(self, data: pd.DataFrame):
        self._data = data

    def draw_heatmap(self):
        sns.heatmap(self._data)
        plt.show()
