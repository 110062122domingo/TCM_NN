# Traditional Chinese Medicine Training using Neuron Network Model

## About This Repository

This repository serve for demonstrating training a neuron network model for traditional chinese medicine prescription prediction.

Main code of the model can be found in tcm_nn_model.ipynb

## Setup Instruction

Dependent packages are listed on `requirements.txt`

install by pip
```
pip install -r requirements.txt
```

install by conda
```
conda install --yes --file requirements.txt
```

## Data Source
Training data is put under the folder /simplified_data. The data used to train the model is simplified_data/simplified_data.csv

## Code executin instructions
Training model can be run by pressing "run all" in tcm_nn_model.ipynb. Code for saving training metrices is included in the code.

## Model performance details
The f1-score of the testing data is around 0.3 ~ 0.6, mainly affected by the balance of the data.<br>
In other words, if the data of a medicine contains the 1s as many as the 0s, it tends to have a higher f1-score.

## Model outpput examples
The output of our model are organized in the results folder. Each subfolder within the results directory follows the specified format: 
**\[(inputlayer)\]_(DeleteMedThreshold)_UseWeight/NoWeight**.

For instance, \[64-32-16\]_100_UseWeight implies:
* This model has the following layer structure: input-64-32-16-output.
* It utilizes medicines that appear more than 100 times in the training data.
* The model employs class weights to address the imbalance in the dataset.

Files in Each Result Folder:
1. Model: Contains the trained neural network model (in .pb format).
2. train_f1.csv: Stores the F1 score of the training data.
3. val_f1.csv: Contains the F1 score of the validation data.

