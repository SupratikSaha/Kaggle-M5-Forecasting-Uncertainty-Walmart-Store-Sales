# M5 Forecasting - Uncertainty

## Kaggle Competition link

https://www.kaggle.com/c/m5-forecasting-uncertainty

## Submission Details

The submission had a Weighted Scaled Pinball Loss score of 0.25547 on the private leader board 
and 0.06691 on the public leader board

# Steps to run project code

## Packages to be installed

Following packages specified in requirements.txt file need to be installed - 
keras, 
lightgbm, 
numpy, 
pandas, 
psutil, 
scikit-learn, 
scipy, 
tensorflow-gpu and 
tqdm

## Folders needed to run code

Create folder named 'data' in the project directory 
and create sub-folders named - 'lgbm_datasets', 'models', 'processed_data', 'raw_data' and 'submissions' within it

## Download Competition Data

Download the competition data files from [Kaggle Competition Data Link](https://www.kaggle.com/c/m5-forecasting-uncertainty/data) and place them in the 'raw_data' folder

Also download the 'sample_submission.csv' file from [Kaggle Competition Data Link](https://www.kaggle.com/c/m5-forecasting-accuracy/data),
rename it to 'sample_submission_accuracy.csv' and place it in the 'raw_data' folder

## Running Code

Run the \_\_main__.py file. It is advised to run the code in pieces. For reference, it took me about 3 days on my HP Spectre i5

# Model

- Initially 8 separate LGBM models are built for each store to derive point forecasts
- Model predictions of these 8 models are concatenated to form the LGBM point forecast
- 3 different Keras deep learning models are built with different embeddings to derive point forecasts
- Arithmetic average of these 3 keras models is used to create a deep learning point forecast
- A weighted average of the concatenated LGBM model and the averaged keras models in 1:3 ratio constitutes the final point forecast
- Finally uncertainty predictions are made using this this average point forecast