import os
import sys
import numpy as np
import pickle
from os.path import dirname
from tensorflow import keras


# [SETTING WARNINGS]
#==============================================================================
import warnings
warnings.simplefilter(action='ignore', category = DeprecationWarning)
warnings.simplefilter(action='ignore', category = FutureWarning)

# [IMPORT MODULES AND CLASSES]
#==============================================================================
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from modules.components.data_classes import UserOperations
from modules.components.training_classes import PreProcessingTimeSeries, TrainingLSTM
from modules.components.validation_classes import ValidationLSTM

# [DEFINE PATHS]
#==============================================================================
if getattr(sys, 'frozen', False):
    data_path = os.path.join(os.path.dirname(dirname(sys.executable)), 'dataset')
    pretrained_path = os.path.join(os.path.dirname(dirname(sys.executable)), 'pretrained models')
else:
    data_path = os.path.join(os.path.dirname(dirname(os.path.abspath(__file__))), 'dataset')
    pretrained_path = os.path.join(os.path.dirname(dirname(os.path.abspath(__file__))), 'pretrained models')
    
if not os.path.exists(data_path):
    os.mkdir(data_path)    
if not os.path.exists(pretrained_path):
    os.mkdir(pretrained_path)
    

# [SELECT FORECASTING METHODS]
#==============================================================================
# Find files within directory using folder inspector classes
#==============================================================================
print('--------------------------------------------------------------------')
print('LSTM forecasting of timeseries')
print('--------------------------------------------------------------------')
forecasting_operations = UserOperations()
forecast_menu = {'1' : 'Perform price forecasting',
                 '2' : 'Go back to main menu'}

forecast_sel = forecasting_operations.menu_selection(forecast_menu) 
print()

# [PRICE FORECASTING]
#==============================================================================
# Find files within directory using folder inspector classes
#============================================================================== 
if forecast_sel == 1: 
    data_preprocessing = PreProcessingTimeSeries(None)      
    
    
    # [LOAD PREPROCESSED ARRAYS AND NORMALIZERS]
    #==========================================================================
    # Training the LSTM model using the functions specified in the designated class.
    # The model is saved in .h5 format at the end of the training
    #==========================================================================
    print('--------------------------------------------------------------------')
    print('Loading preprocessed data')
    print('--------------------------------------------------------------------')       
    processed_data = data_preprocessing.load_preprocessed_data(data_path)
    ts_name = data_preprocessing.ts_name
    X_train, Y_train = processed_data['X_train'], processed_data['Y_train']        
    X_test, Y_test = processed_data['X_test'], processed_data['Y_test']
    normalizer = processed_data['normalizer'] 
    
    # [LOAD TRAINED MODEL]
    #==============================================================================
    print('--------------------------------------------------------------------')
    print('Loading pretrained models')
    print('--------------------------------------------------------------------')
    training_session = TrainingLSTM(device = 'CPU')
    data_validation = ValidationLSTM()    
    LSTM_model = training_session.load_pretrained_model(pretrained_path)
    LSTM_model.summary()
    
    # [TIMESERIES PREDICTION VALIDATION]
    #==========================================================================
    # Predicting values based on trained model
    #==========================================================================
    print('--------------------------------------------------------------------')
    print('Prediction versus actual test values')
    print('--------------------------------------------------------------------')
    predictions = LSTM_model.predict(X_test) 
    predictions = normalizer.inverse_transform(predictions)
    Y_test = normalizer.inverse_transform(Y_test)     
    data_validation.predictions_vs_test(Y_test, predictions, pretrained_path, 600)
    
    # [TIMESERIES FORECASTING]
    #==========================================================================
    # Forecasting of future values
    #==========================================================================
    print('--------------------------------------------------------------------')
    print('Prediction of future prices')
    print('--------------------------------------------------------------------')
    
    
# [GO BACK TO MAIN MENU]
#==============================================================================
# Training the LSTM model using the functions specified in the designated class.
# The model is saved in .h5 format at the end of the training
#============================================================================== 
elif forecast_sel == 2: 
    pass
    
        

# [SCRIPT END]
#==============================================================================  
if __name__ == '__main__':
    pass













