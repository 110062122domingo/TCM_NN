# TCM_NN

## Results
The training results are organized in the results folder. Each subfolder within the results directory follows the specified format: 
**\[(inputlayer)\]_(DeleteMedThreshold)_UseWeight/NoWeight**.
For instance, \[64-32-16\]_100_UseWeight implies that this model has the following layer structure: input-64-32-16-output. Additionally:

* It utilizes medicines that appear more than 100 times in the training data.
* The model employs class weights to address the imbalance in the dataset.

Files in Each Result Folder:
1. Model: Contains the trained neural network model.
2. train_f1.csv: Stores the F1 score of the training data.
3. val_f1.csv: Contains the F1 score of the validation data.
