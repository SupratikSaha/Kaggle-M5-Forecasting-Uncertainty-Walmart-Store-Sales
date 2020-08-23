""" Main file to create submission for M5 Uncertainty Forecasting Competition """

from prepare_data import prepare_datasets
from train import train_model

prepare_datasets()
train_model()
