import tensorflow as tf
import numpy as np
import pandas as pd
from constants import DATA_DIR, TRAIN_FILE,TEST_FILE
import os
from copy import deepcopy 
from sklearn.preprocessing import MinMaxScaler, StandardScaler




def seperate_train_val(VALIDATION_TRIP, CHANNEL):
    df = pd.read_csv(os.path.join(DATA_DIR,TRAIN_FILE))
   
    for i in df.columns:
        if len(df[i].unique()) == 1:
            df = df.drop(columns=[i])
    df = df.sort_values(by=["Driver","Trip","Time(s)"])
    # print(df.index.tolist())
    df.index = range(0,len(df))
    
    df1 = df[["Time(s)","Driver","Trip"]]
    start_index = df1[df1["Time(s)"].isin(["1"])]["Driver"].index
    driver = df1[df1["Time(s)"].isin(["1"])]["Driver"]
    end_index = start_index - 1
    start_index = start_index.tolist()
    end_index = end_index.tolist()
    end_index.append(len(df1)-1)
    end_index = end_index[1:]
    print(start_index)
    print(end_index)
    
    df_copy = deepcopy(df)
    df_copy.drop(columns=["Trip","Time(s)"],inplace=True)
    standard_cols = df_copy.columns[:-1]
    
    scaler = StandardScaler()
    scaler.fit(df_copy[standard_cols])
    df_copy[standard_cols] = scaler.transform(df_copy[standard_cols])
    data_set = {"0":[],"1":[],"2":[]}
    data_fraction = []
    count = 0

    for start,end, dr in zip(start_index,end_index, driver):
        # dirver change
        count += 1 
        for i in range(start,end+1,CHANNEL):
            if i+CHANNEL-1 > end:
                print(i,end,"not use")
            else:
                data_fraction.append(np.array(df_copy.iloc[i:i+CHANNEL])) 
        data_set[str(count%3)].append((np.asarray(data_fraction)))
        print(dr,len(data_fraction))
        data_fraction = []
        
    train_xs = []
    train_ys = []
    val_x = []
    val_y = []
    for i in (data_set.keys()):
        # set validation data
        if i == str(VALIDATION_TRIP):
            for j in range(len(data_set[i])):
                val_x.append(data_set[i][j][:,:,:-1])
                val_y.append(np.array([j]*data_set[i][j].shape[0]))
        else:
            for j in range(len(data_set[i])):
                train_xs.append(data_set[i][j][:,:,:-1])
                train_ys.append(np.array([j]*data_set[i][j].shape[0]))

    train_x = np.concatenate(train_xs)
    train_y = np.concatenate(train_ys)
    val_x = np.concatenate(val_x)
    val_y = np.concatenate(val_y)

    val_y  = np.eye(9)[val_y]
    train_y = np.eye(9)[train_y]
    print(train_x.shape)
    print(train_y.shape)

    print(val_x.shape)
    print(val_y.shape)

    np.save("train_x.npy",train_x)
    np.save("train_y.npy",train_y)
    np.save("val_x.npy",val_x)
    np.save("val_y.npy",val_y)
    
    ## parse test data set
    df_test = pd.read_csv(os.path.join(DATA_DIR,TEST_FILE))
    df_test.index = df_test["Index Number"]
    df_test= df_test[df.columns[:-1]]

    start_index = df_test[df_test["Time(s)"].isin(["1"])].index
    end_index = start_index - 1
    end_index = end_index.tolist()
    end_index.append(len(df_test))
    end_index= end_index[1:]
    
    df_test.drop(columns=["Time(s)","Trip"],inplace=True)
    scaler = StandardScaler()
    scaler.fit(df_test[standard_cols])
    df_test[standard_cols] = scaler.transform(df_test[standard_cols])
    data_set = {}

    count = 0
    for start,end in zip(start_index,end_index):

        for i in range(start,end+1,CHANNEL):
            if i+CHANNEL-1 > end:
                print(i,end,"not use")
            else:
                data_fraction.append(np.array(df_test.loc[i:i+CHANNEL-1])) 
        data_set[count] =((np.asarray(data_fraction)))
        print(count,len(data_fraction))
        count+=1
        data_fraction = []
    np.save("./test_dict.npy",data_set)
    


