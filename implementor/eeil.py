from data.cil_data_load import CILDatasetLoader
from data.custom_dataset import ImageDataset
from implementor.baseline import Baseline
import torch
import torch.nn as nn
import time
import os
import pandas as pd
from utils.calc_score import AverageMeter, ProgressMeter, accuracy
from utils.logger import convert_secs2time
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

class EEIL(Baseline):
    def __init__(self, model, time_data, save_path, device, configs):
        super(EEIL, self).__init__(
            model, time_data, save_path, device, configs)