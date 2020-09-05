""" Main file to create submission for M5 Uncertainty Forecasting Competition """

from pre_process import prepare_datasets
from train import train_model
from predict import predict_results
from post_process import predict_uncertainties

prepare_datasets()
train_model()
predict_results()
predict_uncertainties()
