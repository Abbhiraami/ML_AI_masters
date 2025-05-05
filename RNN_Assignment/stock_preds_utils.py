# Import libraries
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')\

### Packages for combinators
import itertools

## Display python tables
from IPython.display import display

### Sklearn essentials

import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

### statsmodel
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

### Tensorflow essentials

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,SimpleRNN, Dropout, LSTM, GRU,Activation

### Optimizers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.callbacks import Callback,ReduceLROnPlateau,EarlyStopping

### Extracting files from google drive
from google.colab import drive
from os import listdir


### Input time-series data ###

def window_slicing(window_size=None,window_stride=None,data=None,feats=None,company_name=None,target_columns=None,target_indicator="Close"):
  """
  window_size: lookback period
  window_stride: output variable 
  data: merged dataframe
  feats: list of indicators: 
     Multivariate time series: ['high','Low','Close']
     Univariate time series:['Close']
  company_name: Multivaraite non-cross modelling. The close price of GOOGL is will be dependent only on GOOGL indicators not other stocks
  output: X and y
  """

  X,y=[],[]
  data=data.sort_values('Date_',ascending=True)
  if company_name is None and feats is None:
    df_X=data.drop(columns="Date_")
    df_y=data[[f"{target_indicator}_{targ}" for targ in target_columns]]
  else:
    df_X=data[[f"{feat_name}_{comp_name}" for feat_name in feats for comp_name in company_name ]]
    df_y=data[[f"{target_indicator}_{comp}" for comp in company_name]]
    
  for i in range(0,df_X.shape[0]-window_size,window_stride):
    X.append(np.array(df_X.iloc[i:i+window_size]))
    y.append(np.array(df_y.iloc[i+window_size]))
  return np.array(X),np.array(y)

### Train test split ###
def train_test_split_ts(x_data=None,y_data=None,split_ratio=0.8):
  """
  x_data: windowed x set
  y_data: windowed y set
  split_ratio: train-test split ratio
  """

  indx=int(x_data.shape[0]*0.8)
  ### Train set
  X_train=x_data[:indx]
  X_test=x_data[indx:]
  ### Test set
  y_train=y_data[:indx]
  y_test=y_data[indx:]
  return X_train,X_test,y_train,y_test

### Scaling the data ###
def scaled_input(X_train,X_test,Y_train,Y_test):
  """
  X_train: train input
  X_test: test input
  Y_train: train target
  Y_test: test target
  output: scaled input and target with their respective scalers
  """

  ### Scaling the input
  scalar_x=MinMaxScaler()
  scalar_y=MinMaxScaler()
  ### features scaling
  n_samples_train,n_timeteps,n_features=X_train.shape
  n_samples_test,n_timeteps_test,n_features_test=X_test.shape
  n_samples,n_companies=Y_test.shape  
  X_train_scaled=scalar_x.fit_transform(X_train.reshape(-1,n_features)).reshape(n_samples_train,n_timeteps,n_features)
  X_test_scaled=scalar_x.transform(X_test.reshape(-1,n_features)).reshape(n_samples_test,n_timeteps,n_features)
  ### Target scaling
  Y_train_scaled=scalar_y.fit_transform(Y_train.reshape(-1,1)).reshape(-1,n_companies)
  Y_test_scaled=scalar_y.transform(Y_test.reshape(-1,1)).reshape(-1,n_companies)

  return X_train_scaled,X_test_scaled,Y_train_scaled,Y_test_scaled,scalar_x,scalar_y

# Data pre_precessing for model fitting
def data_preprocessing(data=None,split_ratio=None,window_size=None,window_stride=None,company_name=None,feats=None,target_columns=None):
  """
  data: merged dataframe
  split_ratio: ratio of train test split
  window_size:lookback period
  window_stride: output variable
  company_name: list of company names
  feats: list of indicators: Multivariate time series: ['high','Low','Close']
                             Univariate time series:['Close']
                             
  output: X and y train and test
  """
### 1. windowing the data
  X,Y=window_slicing(window_size=window_size,window_stride=window_stride,data=data,feats=feats,company_name=company_name,target_columns=target_columns,target_indicator="Close")
  ### 2. Train-test split
  X_train,X_test,y_train,y_test=train_test_split_ts(X,Y,split_ratio=0.8)
  print(f"X_train:{X_train.shape}\ny_train:{y_train.shape}")
  print(f"X_test:{X_test.shape}\ny_test:{y_test.shape}")
  ### 3. Scaling
  X_train_scaled,X_test_scaled,Y_train_scaled,Y_test_scaled,scalar_x,scalar_y=scaled_input(X_train,X_test,y_train,y_test)
  return X_train_scaled,X_test_scaled,Y_train_scaled,Y_test_scaled,scalar_x,scalar_y

### print only final epoch function ####
class FinalEpochLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch == self.params['epochs'] - 1:
            print(f"Final Epoch {epoch+1}: {logs}")

### Build Simple RNN
def create_simple_rnn_model(X=None,configurations=list,Y=None):
  ### Initializing the model
  model=Sequential()
  ### Add layers of simple RNN - input layer
  model.add(SimpleRNN(units=configurations['n_units'],return_sequences=False,kernel_initializer="glorot_uniform",input_shape=(X.shape[1],X.shape[2])))
  ## HIDDEN LAYER 1
  model.add(Dense(units=configurations['dense_units'], activation=configurations['activation']))
  model.add(Activation(configurations['activation']))
  model.add(Dropout(configurations['dropout']))
  ## Hidden layer -2
  model.add(Dense(units=configurations['dense_units'], activation=configurations['activation']))
  model.add(Activation(configurations['activation']))
  model.add(Dropout(configurations['dropout']))
  ### Output layer
  model.add(Dense(units=Y.shape[1]))
  # print(model.summary())

  return model
### Build LSTM 
def create_lstm_model(X=None,configurations=None,Y=None):
  model=Sequential()
  # Set return_sequences=True for the first LSTM layer if there are more layers to be added
  model.add(LSTM(units=configurations['units'],activation=configurations['activation'],return_sequences=configurations['layers'] > 1,input_shape=(X.shape[1],X.shape[2])))
  ## Adding layers of LSTM
  for i in range(configurations['layers'] - 1): # Subtract 1 as the first layer is already added
    # Set return_sequences=True for intermediate LSTM layers
    model.add(LSTM(units=configurations['units'],activation=configurations['activation'], return_sequences=True if i < configurations['layers'] - 2 else False))

  model.add(Dense(units=Y.shape[1]))
  return model

### Build GRU
def create_gru_model(X=None,configurations=None,Y=None):
  model=Sequential()
  # Set return_sequences=True for the first GRU layer if there are more layers to be added
  model.add(GRU(units=configurations['units'],activation=configurations['activation'],return_sequences=configurations['layers'] > 1,input_shape=(X.shape[1],X.shape[2])))
  ## Adding layers of GRU
  for i in range(configurations['layers'] - 1): # Subtract 1 as the first layer is already added
    # Set return_sequences=True for intermediate GRU layers
    model.add(GRU(units=configurations['units'],activation=configurations['activation'], return_sequences=True if i < configurations['layers'] - 2 else False))

  model.add(Dense(units=Y.shape[1]))
  return model

#### Running through several configuration to get the best of the model built

def model_fit(X_train=None,X_test=None,Y_train=None,Y_test=None,func=None,configurations=None,batch_size=32,epoch=100):
  history_list={}
  metrics_list={}
  y_preds_list={}
  model_dict={}
  for i,config in enumerate(configurations):
    # Dynamic counter
    sys.stdout.write(f"\rProcessing config {i + 1}/{len(configurations)} ...")
    sys.stdout.flush()
    # print( f"Processing {[config]}")
    model_seq=func(X=X_train,Y=Y_train,configurations=config)
    ### compile model
    opt_dict = {
      'adam': Adam(learning_rate=config['learning_rate']),
      'sgd': SGD(learning_rate=config['learning_rate']),
      'rmsprop': RMSprop(learning_rate=config['learning_rate'])
    }
    model_seq.compile(optimizer=opt_dict['adam'], loss='mean_squared_error',metrics=["mse"])
    
    ### fit the model
    history_model=model_seq.fit(X_train,Y_train,epochs=epoch,batch_size=batch_size,
                              validation_data=(X_test,Y_test),verbose=0)
    history_list[config['name']]=history_model.history

    ### Predict on the train data
    y_pred_train=model_seq.predict(X_train,verbose=0)

    ### Predict on the test data
    y_pred_test=model_seq.predict(X_test,verbose=0)
    # y_preds_dict[config['name']]=y_pred_

    ### Metrics dictionary
    y_train=Y_train.reshape(-1,Y_train.shape[1])
    y_test=Y_test.reshape(-1,Y_train.shape[1])
    y_preds_list[config['name']]={"y_pred_train":[y_pred_train],"y_pred_test":[y_pred_test]}
    ### MSE train
    mse_train=mean_squared_error(y_train,y_pred_train)
    mse_test=mean_squared_error(y_test,y_pred_test)
    ### MAE test
    mae_train=mean_absolute_error(y_train,y_pred_train)
    mae_test=mean_absolute_error(y_test,y_pred_test)
    
    ### save_model
    model_dict[config['name']]=model_seq
    ### R-squared value
    r2_train=r2_score(y_train,y_pred_train)
    r2_test=r2_score(y_test,y_pred_test)
    metrics_list[config['name']]=[mse_train,mse_test,mae_train,mae_test,r2_train,r2_test]
    if r2_test>0.7 and (r2_train-r2_test)<=0.05:
      print(f"MSE:{mse_test}\nMAE:{mae_test}\nR2:{r2_test}")
    else:
      pass
  return history_list,metrics_list,model_dict

### Get the best config
def get_best_config(metrics_dict,config_df):
  """
  Input: metrics dictionary from trained models and configuration list
  Output: best configuration name and best configuration
  """
  metrics_df=pd.DataFrame(metrics_dict).T.rename(columns={0:"MSE_train",1:"MSE_test",2:"MAE_train",3:"MAE_test",4:"R2_train",5:"R2_test"}).sort_values('R2_test',ascending=False).reset_index()
  display(metrics_df.head())
  best_fit=metrics_df.iloc[0]
  print(f'best_config:\n')
  best_config=config_df[config_df["name"]==best_fit["index"]]
  return best_fit["index"],best_config

### Retrain the model with the best config
def retrain_model_with_best_config(X_train,X_test,Y_train,Y_test,func1=None,func2=None,
                                   best_fit_name=None,best_config=None,config_df=None,batch_size=None,epoch=None,y_scalar=None):
  ### picking only the best config
 
  best_config = config_df[config_df["name"]==best_fit_name].to_dict('records')

  hist,metrics,model=func1(X_train=X_train,X_test=X_test,Y_train=Y_train,Y_test=Y_test,
                                                          func=func2,configurations=best_config,
                                                          batch_size=batch_size,epoch=epoch)

  ### Evaluate the model
  train_val=model[best_fit_name].evaluate(X_train,Y_train)
  test_val=model[best_fit_name].evaluate(X_test,Y_test)
  print(f"Train Loss:{train_val}\nTest Loss:{test_val}")
  
  ### Predict on test and train data
  y_pred_test = model[best_fit_name].predict(X_test)
  y_pred_train = model[best_fit_name].predict(X_train)

  ### R-squared values
  print(f"Rsquared-score for test:{100*r2_score(Y_test,y_pred_test):.2f}%")
  print(f"Rsquared-score for train:{100*r2_score(Y_train,y_pred_train):.2f}%")


  ### Inverse the scaling 
  y_pred_test_inv=y_scalar.inverse_transform(y_pred_test)
  y_test_inv=y_scalar.inverse_transform(Y_test)
  y_pred_train_inv=y_scalar.inverse_transform(y_pred_train)
  y_train_inv=y_scalar.inverse_transform(Y_train)

  return y_pred_test_inv,y_test_inv, y_pred_train_inv,y_train_inv
  
