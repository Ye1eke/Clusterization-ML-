import os
import numpy as np
from predict import predict
from sklearn.metrics import accuracy_score


bag = []

ROOT_DIR            = os.getcwd()

MAX_LINES = 500
MAX_TEST_LINES = 747

y_pred = []
y_true = []
