# Tensor-Network-QML
Freeform Project Submission for Xanadu's QHack 2021

Team Name: The Racing Scarecrow

The final report may be found in the __TN Born Machine.ipynb__ file in the form of a Jupyter Notebook. It is best viewed in Nbviewer, the link to which is as follows: 
[Final report](https://nbviewer.jupyter.org/github/Arkonaire/Tensor-Network-QML/blob/master/TN%20Born%20Machines.ipynb)

# Files
* __ansatz_layers.py__ : Implementation of the TTN and MPS layers described in [1].
* __ansatz_circuits.py__ : Discriminative and Generative circuits based on TTN and MPS layers.
* __bars_and_stripes.py__ : Implementation of Born Machine via TTN and MPS ansaetze.
* __data_processing__ : Preprocess __rain_data/weatherAUS.csv__ and store normalized features and labels in __rain_data/processed_data__.
* __rain_forecast.py__ : Rain Forecasting using __rain_data/weatherAUS.csv__.

The generative_model folder contains results of the Born Machine Smulations. See notebook for usage.
