import numpy as np
from typing import Tuple
from sklearn.metrics import roc_auc_score, roc_curve


class Evaluation:
    def __init__(self, test_data, predict_data, true_label):
        self.test_data = test_data
        self.predict_data = predict_data
        self.error = self.calculate_error()

    def calculate_error(self) -> np.ndarray:
        return (self.test_data - self.predict_data) ** 2

    def calcurate_fpr_tpr(self) -> Tuple[np.ndarray, np.ndarray]:
        fpr, tpr, _ = roc_curve(self.true_label, self.test_data)
        return fpr, tpr

    def calculate_auc(self) -> float:
        return roc_auc_score(self.true_label, self.test_data)

