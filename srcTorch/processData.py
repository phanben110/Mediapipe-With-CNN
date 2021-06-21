import cv2
import os
import json
import numpy as np

json_path = './../data/processed/dataHand12ClassSize26.json'

class Dataset():
    def __init__(self,data_path=json_path):
        self.data = {
            "image": [],
            "labels": [],
            "mapping": []
        }
        with open(data_path, "r") as fp:
            data = json.load(fp)
        x = np.array(data["image"],dtype=np.float32)
        y = np.array(data["labels"])
        z = np.array(data["mapping"])
        self.data["image"] = x
        self.data["labels"] = y
        self.data["mapping"] = z

