import numpy as np
import pandas as pd
import random

RAND_SEED = 42

def ReadData(FILENAME):
    data = pd.read_csv(FILENAME, encoding='ANSI')
    print("--------------------------------------------------------------------------------")
    print("ReadData:")
    print("Type of data:", type(data))
    print(f'Shape of data = ({data.shape[0]} rows, {data.shape[1]} cols).')
    print("End of ReadData")
    return data

def SplitXY(data, data_left_bound, label_left_bound, data_right_bound=None, label_right_bound=None):
    """
    
    label_left_bound: 第一個藥材的col no.
    
    Return
    ----
    X: pd.DataFrame
    y: pd.DataFrame
    """
    # Body status: 1~3, Diagnosis: 4~7, Symptom: 11~124
    # Prescription: 125~226
    # split_X = list(range(0, label_left_bound))
    # split_Y = list(range(label_left_bound, len(data.columns)))
    if label_right_bound is None:
        label_right_bound = len(data.columns)
    if data_right_bound is None:
        data_right_bound = label_left_bound
    X = data.iloc[1:, list(range(data_left_bound, data_right_bound))]
    y = data.iloc[1:, list(range(label_left_bound, label_right_bound))]
    
    # Debug
    print("--------------------------------------------------------------------------------")
    print("SplitXY:")
    print(f'Shape of X = ({X.shape[0]} rows, {X.shape[1]} cols).')
    # print("First 10 data of X:")
    # print(X.iloc[:10, :10])
    print(f'Shape of y = ({y.shape[0]} rows, {y.shape[1]} cols).')
    # print("First 10 data of y:")
    # print(y.iloc[:10, :10])
    print("End of SplitXY")
    return X, y

def SplitNparr(original_arr: np.ndarray, train_portion: float)->tuple:
    data_len =len(original_arr)
    train_len = int(data_len * train_portion)
    indices = list(range(data_len))
    random.shuffle(indices)
    train_idx = indices[:train_len]
    validate_idx = indices[train_len:]
    training_arr  = original_arr[train_idx]
    validation_arr = original_arr[validate_idx]

    return (training_arr,validation_arr)


def Split1Df(data: pd.DataFrame,  train_size: float, random_state: int=None):
    """
    Split the data into training and validation sets.

    Param:
    - data: any data
    - train_size: Proportion of in train set.
    - random_state: Seed 

    return:
    - data_train: data for the training set.
    - data_val: data for the validation set.

    """
    if random_state is None:
        random_state = 42
    random.seed(random_state)
    data_len = len(data)
    val_len = int(data_len * (1-train_size))
    indices = list(range(data_len))
    random.shuffle(indices)
    val_idx = indices[:val_len]
    train_idx = indices[val_len:]


    data_train, data_val = data.loc[train_idx], data.loc[val_idx]
   
    return data_train, data_val

def SplitBothXy_Df(X: pd.DataFrame, y: pd.DataFrame, train_size: float, random_state: int=None):
    """
    Split the data into training and validation sets.

    Param:
    - X: Features 
    - y: Target variable
    - train_size: Proportion of in train set.
    - random_state: Seed 

    return:
    - X_train: Features for the training set.
    - X_val: Features for the validation set.
    - y_train: Target variable for the training set.
    - y_val: Target variable for the validation set.
    """
    if random_state is None:
        random_state = 42
    random.seed(random_state)
    data_len = len(X)
    val_len = int(data_len * (1-train_size))
    indices = list(range(data_len))
    random.shuffle(indices)
    val_idx = indices[:val_len]
    train_idx = indices[val_len:]
    #print("type of val_idx", type(val_idx))
    #print("type of train_idx", type(train_idx))
    # print("--------------------------------------------------------------------------------")
    # print("SplitBothXy_Df, len of val_idx = ", len(val_idx))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert(X_train.shape[1] == X_val.shape[1])
    assert(y_train.shape[1] == y_val.shape[1])
    return X_train, X_val, y_train, y_val

def DeleteMedicine(y, threshold: int=250):
    if threshold == 0:
        return y
    """
        delete 出現次數< threshold 的藥
    """
    for col in y.columns:
        # print(f'Number of {col} is {y[col].sum()}.')
        if y[col].sum() < threshold:
            y = y.drop(col, axis=1)
    # Debug
    print("--------------------------------------------------------------------------------")
    print(f'DeleteMedicine: shape of y is {y.shape}.')
    return y


def load_data_for_1_med(del_med_thres: int=0, random_seed: int =None, n:int=None):
    """
     Param
    -------
    del_med_thres: 出現次數少於del_med_thres會被刪除,如不想delete , set this to 0
    
    """
    data_pd = ReadData("./simplified_data/simplified_data2.csv")
    first_medicine_idx = None
    #first_medicine_idx = 113
    first_medicane = '麻黃'

    for i in range(len(data_pd.columns.tolist())):
        if data_pd.columns.tolist()[i] == first_medicane:
            first_medicine_idx = i
            break
    assert(data_pd.columns[first_medicine_idx] == first_medicane)   # check del med 前第一隻藥是否'麻黃'

    if n is not None:
        right_bd = first_medicine_idx + n
    else:
        right_bd=None

    # split data into X and y
    # x= all symptoms, diagnosis, body status
    # y= all medince
    X,y = SplitXY(data_pd, data_left_bound=2, 
                  data_right_bound=first_medicine_idx, 
                  label_left_bound=first_medicine_idx,
                    label_right_bound=right_bd)

    # drop 出現次數少於threshold 的藥in y
    y = DeleteMedicine(y, threshold=del_med_thres)

    train_X, val_X, train_y, val_y = SplitBothXy_Df(X, y, 0.8, random_state=random_seed)


    print("train_X.shape: ", train_X.shape)
    print("train_y.shape: ", train_y.shape)
    
    X_np = train_X.values.astype('float64')
    X_val_np = val_X.values.astype('float64')
    print("X transformed to np array")
    print("type of X_np:", type(X_np))
    print("shape of X_np:", X_np.shape)
    print("shape of train y:", train_y.shape)
    print("type of X_val_np:", type(X_val_np))
    print("shape of X_val_np:", X_val_np.shape)
    num_col_x = X_np.shape[1]
    print("number of col in (train) x:",num_col_x)
    print("number of medicine in y: ", train_y.shape[1])

    return (X_np, X_val_np, train_y, val_y,  num_col_x)



def load_data_for_n_med(n: int=0, triain_all_med: bool=False, del_med_thres: int=0, random_seed: int =None):
    """
    Param
    -------

    n: 要train多少個藥
    triain_all_med: 是否train 所有藥, if yes則ignore n, 並放所有藥進y
    del_med_thres: 出現次數少於del_med_thres會被刪除,如不想delete , set this to 0

    Return
    ------
    tuple of all returns

    - X_np: nparr, 
    - X_val_np: nparr,
    - train_y: df, 
    - val_y: df,  
    - num_col_x: int, 
    - num_medic: int)

    """
    data_pd = ReadData("./simplified_data/simplified_data2.csv")
    first_medicine_idx = None
    #first_medicine_idx = 113
    first_medicane = '麻黃'

    for i in range(len(data_pd.columns.tolist())):
        if data_pd.columns.tolist()[i] == first_medicane:
            first_medicine_idx = i
            break

    assert(data_pd.columns[first_medicine_idx] == first_medicane)
    
    if triain_all_med:
        right_bd = None
    else:
        right_bd = first_medicine_idx + n

    # split data into X and y
    # x= all symptoms, diagnosis, body status
    # y= all medince
    X,y = SplitXY(data_pd, data_left_bound=2,
              data_right_bound=first_medicine_idx, 
              label_left_bound=first_medicine_idx, 
              label_right_bound=right_bd)
    # drop 出現次數少於threshold 的藥in y
    y = DeleteMedicine(y, threshold=del_med_thres)

    train_X, val_X, train_y, val_y = SplitBothXy_Df(X, y, 0.8, random_state=random_seed)


    print("train_X.shape: ", train_X.shape)
    print("train_y.shape: ", train_y.shape)
    
    X_np = train_X.values.astype('float64')
    X_val_np = val_X.values.astype('float64')
    y_np = train_y.values.astype('float64')
    y_val_np = val_y.values.astype('float64')

    num_medic = train_y.shape[1]
    num_col_x = X_np.shape[1]
    print("X transformed to np array")
    print("type of X_np:", type(X_np))
    print("shape of X_np:", X_np.shape)
    assert(X_np.shape == train_X.shape)
    assert(X_val_np.shape == val_X.shape)
    assert(train_y.shape[0] == train_X.shape[0])
    print("shape of train y:", train_y.shape)
    print("type of X_val_np:", type(X_val_np))
    print("shape of X_val_np:", X_val_np.shape)
    
    print("number of col in (train) x:",num_col_x)
    print("number of medicine in y: ", train_y.shape[1])

    return (X_np, train_X, X_val_np, val_X, y_np, train_y, y_val_np, val_y, num_col_x, num_medic)



def load_data_for_1_med_with_debug(del_med_thres: int=0, random_seed: int =None, n:int=None, file_name:str=None):
    """
     Param
    -------
    del_med_thres: 出現次數少於del_med_thres會被刪除,如不想delete , set this to 0
    
    """
    file_name = "./simplified_data/simplified_data2.csv" if file_name is None else file_name
    
    data_pd = ReadData(file_name)
    first_medicine_idx = None
    #first_medicine_idx = 113
    first_medicane = '麻黃'

    for i in range(len(data_pd.columns.tolist())):
        if data_pd.columns.tolist()[i] == first_medicane:
            first_medicine_idx = i
            break
    
    if n is not None:
        right_bd = first_medicine_idx + n
    else:
        right_bd=None


    # split data into X and y
    # x= all symptoms, diagnosis, body status
    # y= all medince
    X,y = SplitXY(data_pd, data_left_bound=2, data_right_bound=first_medicine_idx, label_left_bound=first_medicine_idx, label_right_bound=right_bd)
    
    print("--------------------------------------------------------------------------------")
    print("In load_data_for_1_med_with_debug of load_data.py, random_seed=", random_seed)
    #### debug no. of 0 and 1
    counts = pd.Series(y.values.flatten()).value_counts()
    num_1 = counts.get(1, 0)
    num_0 = counts.get(0, 0)
    print("After SplitXY, total number of 0, 1 in y:")
    print("Number of 0s:", num_0)
    print("Number of 1s:", num_1)
    
    # drop 出現次數少於threshold 的藥in y
    y = DeleteMedicine(y, threshold=del_med_thres)

    train_X, val_X, train_y, val_y = SplitBothXy_Df(X, y, 0.8, random_state=random_seed)

    # caluculate number of 0 and 1 in train_X
    save_num_med_of_data(train_y)

    print("Train_X.shape: ", train_X.shape)
    print("Train_y.shape: ", train_y.shape)
    
    X_np = train_X.values.astype('float64')
    X_val_np = val_X.values.astype('float64')
    # print("X transformed to np array")
    # print("type of X_np:", type(X_np))
    # print("shape of X_np:", X_np.shape)
    # print("shape of train y:", train_y.shape)
    # print("type of X_val_np:", type(X_val_np))
    # print("shape of X_val_np:", X_val_np.shape)
    num_col_x = X_np.shape[1]
    # print("number of col in (train) x:",num_col_x)
    # print("number of medicine in y: ", train_y.shape[1])
    print("\nSplit Training Validation")
    
    counts_trainy = pd.Series(train_y.values.flatten()).value_counts()
    num_1_trainy = counts_trainy.get(1, 0)
    num_0_trainy = counts_trainy.get(0, 0)

    counts_valy = pd.Series(val_y.values.flatten()).value_counts()
    num_1_valy = counts_valy.get(1, 0)
    num_0_valy = counts_valy.get(0, 0)

    print("Number of 0s in train_y:", num_0_trainy)
    print("Number of 1s train_y:", num_1_trainy)
    print("Number of 0s in val_y:", num_0_valy)
    print("Number of 1s val_y:", num_1_valy)


    print("--------------------------------------------------------------------------------")
    return (X_np, X_val_np, train_y, val_y,  num_col_x, num_1_valy, num_0_valy)


def save_num_med_of_data(train_X: pd.DataFrame):
    # create a empty df named med_cnt
    med_cnt = pd.DataFrame(columns=['med_name', 'count of 1', 'count of 0'])
    for i in range(len(train_X.columns)):
        counts = train_X.iloc[:, i].value_counts()
        num_1 = counts.get(1, 0)
        num_0 = counts.get(0, 0)
        med_cnt.loc[i] = [train_X.columns[i], num_1, num_0]
    med_cnt.to_csv('./simplified_data/med_cnt.csv', index=False)

    print("save med num done")


