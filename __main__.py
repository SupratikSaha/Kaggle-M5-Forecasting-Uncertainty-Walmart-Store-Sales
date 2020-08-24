""" Main file to create submission for M5 Uncertainty Forecasting Competition """

from prepare_data import prepare_datasets
from train import train_model
from predict import predict_results
from predict_uncertainty import predict_uncertainties

prepare_datasets()
train_model()
predict_results()
predict_uncertainties()
