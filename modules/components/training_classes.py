import os
import numpy as np
import pickle
from datetime import datetime
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Conv1D
from sklearn.preprocessing import MinMaxScaler
import modules.global_variables as GlobVar

# define model class
#==============================================================================
#==============================================================================
#==============================================================================
class PreProcessingTimeSeries:

    
    # Splits time series data into training and testing sets using TimeSeriesSplit
    #==========================================================================
    def split_timeseries(self, dataframe, test_size):
        
        """
        timeseries_split(dataframe, test_size)
        
        Splits the input dataframe into training and testing sets based on the test size.
    
        Keyword arguments:  
        
        dataframe (pd.dataframe): the dataframe to be split
        test_size (float):        the proportion of data to be used as the test set
    
        Returns:
            
        df_train (pd.dataframe): the training set
        df_test (pd.dataframe):  the testing set
        
        """
        train_size = int(len(dataframe) * (1 - test_size))
        df_train = dataframe.iloc[:train_size]
        df_test = dataframe.iloc[train_size:]

        return df_train, df_test
        
    # array scaling for the KRISK neural network, accepts an array fabricated
    # through the dataset_splitting to scale train and test dataset independently.
    # reshapes an input to 2dimensions if is unidimentional 
    #==========================================================================   
    def timeseries_normalization(self, df_train, df_test):
        
       """ 
       unidimensional_data_scaling(dataframe)
       
       Scales the input dataframe using MinMaxScaler.
    
       Keyword arguments:  
        
       dataframe (pd.dataframe): the dataframe to be scaled
    
       Returns:
            
       scaled_df (pd.dataframe): the scaled dataframe      
       
       """
       self.normalizer = MinMaxScaler(feature_range = (0, 1))       
       self.normalizer.fit(df_train)      
       train_norm = self.normalizer.transform(df_train)
       test_norm = self.normalizer.transform(df_test)      
       
       return train_norm, test_norm
    
    
    # generate n real samples with class labels; We randomly select n samples 
    # from the real data array
    #========================================================================== 
    def timeseries_labeling(self, df_train, df_test, window_size):
        
        """
        timeseries_labeling(dataframe, window_size)
    
        Labels time series data by splitting into input and output sequences using sliding window method.
    
        Keyword arguments:
            
        dataframe (pd.DataFrame): the time series data to be labeled
        window_size (int):        the number of time steps to use as input sequence
    
        Returns:
            
        X_array (np.ndarray):     the input sequence data
        Y_array (np.ndarray):     the output sequence data
        
        """        
        label_train = np.array(df_train)
        label_test = np.array(df_test)        
        X_train = [label_train[i : i + window_size] for i in range(len(label_train) - window_size)]
        Y_train = [label_train[i + window_size] for i in range(len(label_train) - window_size)]
        X_test = [label_test[i : i + window_size] for i in range(len(label_test) - window_size)]
        Y_test = [label_test[i + window_size] for i in range(len(label_test) - window_size)]        
        labeled_df = {'train_X' : np.array(X_train), 'train_Y' : np.array(Y_train),
                      'test_X' : np.array(X_test), 'test_Y' : np.array(Y_test)}        
                     
        return labeled_df
    
    
    # generate n real samples with class labels; We randomly select n samples 
    # from the real data array
    #========================================================================== 
    def save_preprocessed_data(self, path, labeled_df, normalizer):
        
        savepath = os.path.join(path, 'preprocessed_data.npz')
        np.savez(savepath, 
                 X_train = labeled_df['train_X'],
                 Y_train = labeled_df['train_Y'],
                 X_test = labeled_df['test_X'],
                 Y_test = labeled_df['test_Y'])                                          
        
        normalizer_path = os.path.join(path, 'normalizer.pkl')
        with open(normalizer_path, 'wb') as file:
            pickle.dump(normalizer, file)   

        

# Callback class inheriting from keras.Callbacks
#==============================================================================
#==============================================================================
#==============================================================================
class RealTimeHistory(keras.callbacks.Callback):   
    
    def __init__(self, plot_path):        
        super().__init__()        
        self.plot_path = plot_path
        self.epochs = []
        self.loss_hist = []
        self.loss_val_hist = []
        self.metric_hist = []
        self.metric_val_hist = []
            
    #--------------------------------------------------------------------------
    def on_epoch_end(self, epoch, logs = {}):        
        if epoch % 2 == 0:
            self.epochs.append(epoch)
            self.loss_hist.append(logs['loss'])
            self.loss_val_hist.append(logs['val_loss'])
            self.metric_hist.append(logs['mae'])
            self.metric_val_hist.append(logs['val_mae'])           
            #------------------------------------------------------------------
            fig_path = os.path.join(self.plot_path, 'training_history.jpeg')
            figure, ax = plt.subplots(nrows=2, ncols=1)
            plt.subplot(2, 1, 1)
            plt.plot(self.epochs, self.loss_hist, label = 'training loss')
            plt.plot(self.epochs, self.loss_val_hist, label = 'validation loss')
            plt.legend(loc = 'best', fontsize = 8)
            plt.title('Loss plot LSTM')
            plt.ylabel('mean squared error')
            plt.xlabel('epoch')
            plt.subplot(2, 1, 2)
            plt.plot(self.epochs, self.metric_hist, label = 'train metrics')  
            plt.plot(self.epochs, self.metric_val_hist, label = 'validation metrics') 
            plt.legend(loc = 'best', fontsize = 8)
            plt.title('metrics plot LSTM')
            plt.ylabel('mean absolute error')
            plt.xlabel('epoch')       
            plt.tight_layout()
            plt.savefig(fig_path, bbox_inches = 'tight', format = 'jpeg', dpi = 300)
            

# define model class
#==============================================================================
#==============================================================================
#==============================================================================
class TrainingModels:
    
    def __init__(self, device = 'default'):         
        self.available_devices = tf.config.list_physical_devices()
        print('----------------------------------------------------------------')
        print('The current devices are available: ')
        for dev in self.available_devices:
            print()
            print(dev)
        print()
        print('----------------------------------------------------------------')
        if device == 'GPU':
            self.physical_devices = tf.config.list_physical_devices('GPU')
            tf.config.set_visible_devices(self.physical_devices[0], 'GPU')
            print('GPU is set as active device')
            print('----------------------------------------------------------------')
            print()        
        elif device == 'CPU':
            tf.config.set_visible_devices([], 'GPU')
            print('CPU is set as active device')
            print('----------------------------------------------------------------')
            print()
    
    
    # sequential model as generator with Keras module
    #========================================================================== 
    def ConvSubModel(self, window_size, name):
        
        """ 
        LSTM_model(window_size, learning_rate)
    
        Creates a Keras Sequential model for time series prediction.

        Keyword arguments: 
        
        window_size (int):     size of the window used for prediction
        learning_rate (float): learning rate for the optimizer
                
        Returns:
        
        model (keras model): Keras sequential model for time series prediction
                  
        """              
        inputs = Input(shape = (window_size, 1), name = 'input_layer')
        #----------------------------------------------------------------------        
        BatchNorm = BatchNormalization(momentum = 0.8, name = 'batch_normalization_1')(inputs)
        #----------------------------------------------------------------------
        conv1D = Conv1D(filters = 96, activation = 'tanh', kernel_size = 3, padding = 'same',
                        name = 'convolutional_1D')(BatchNorm)
        #----------------------------------------------------------------------
        drop_1 = Dropout(0.4, name = 'dropout_1')(conv1D)
        #----------------------------------------------------------------------        
        BatchNorm_2 = BatchNormalization(momentum=0.8, name = 'batch_normalization_2')(drop_1)
        #----------------------------------------------------------------------
        LSTM_1 = LSTM(96, activation = 'tanh', return_sequences=True, use_bias=False, name='LSTM_layer')(BatchNorm_2)
        #----------------------------------------------------------------------
        drop_2 = Dropout(0.4, name = 'dropout_2')(LSTM_1)
        #----------------------------------------------------------------------
        BatchNorm_3 = BatchNormalization(momentum=0.8, name='batch_normalization_3')(drop_2)
        #----------------------------------------------------------------------
        LSTM_2 = LSTM(96, return_sequences=True, activation='tanh', use_bias = False, name='LSTM_2')(BatchNorm_3)       
        #----------------------------------------------------------------------
        drop_3 = Dropout(0.4, name = 'dropout_3')(LSTM_2)        
        
        model = Model(inputs = inputs, outputs = drop_3, name = name)
        
        return model
        
    # sequential model as generator with Keras module
    #========================================================================== 
    def Conv1DLSTM_model(self, window_size, learning_rate):
        
        """ 
        LSTM_model(window_size, learning_rate)
    
        Creates a Keras Sequential model for time series prediction.

        Keyword arguments: 
        
        window_size (int):     size of the window used for prediction
        learning_rate (float): learning rate for the optimizer
                
        Returns:
        
        model (keras model): Keras sequential model for time series prediction
                  
        """        
        
        inputs = Input(shape = (window_size, 1), name = 'input_layer')
        
        # submodels
        #----------------------------------------------------------------------
        convsubmodel = self.ConvSubModel(window_size, name = 'ConvSubModel')        
        feat_extraction = convsubmodel(inputs)
        #----------------------------------------------------------------------        
        BatchNorm = BatchNormalization(momentum = 0.8, name = 'batch_normalization')(feat_extraction)
        #----------------------------------------------------------------------
        main_LSTM = LSTM(96, return_sequences = False, activation = 'relu', 
                             use_bias = False, name = 'LSTM_branched')(BatchNorm)
        #----------------------------------------------------------------------
        drop_3 = Dropout(0.4, name = 'dropout_3')(main_LSTM)
        #----------------------------------------------------------------------        
        dense_1 = Dense(64, activation = 'relu', name = 'dense_layer_1')(drop_3)
        #----------------------------------------------------------------------
        drop_4 = Dropout(0.4, name = 'dropout_4')(dense_1)
        #----------------------------------------------------------------------
        dense_2 = Dense(32, activation = 'relu', name = 'dense_layer_2')(drop_4)
        #----------------------------------------------------------------------
        dense_output = Dense(1, activation = 'linear', name = 'output_layer')(dense_2)
        #----------------------------------------------------------------------
        
        model = Model(inputs = inputs, outputs = dense_output, name = 'LSTM_model')        
        opt = keras.optimizers.Adam(learning_rate = learning_rate)
        model.compile(loss = 'mean_squared_error', optimizer = opt, metrics = ['mae'])       
        
        return model
    
    # GAN loss and accuracy history plotting
    #==========================================================================
    def plot_history(self, epochs, loss, val_loss, metric, val_metric, path):
        
        """ 
        plot_history(dr_loss, df_loss, g_loss, dr_acc, df_acc, path)
        
        Plots the accuracy and loss of the GAN model over the iterative process.
        Metrics are plotted referring to both the discriminator and generator 
        models, indicating the performance at generating realistic fake numbers 
        and discriminating real and fake numbers. The figure is saved into the
        target folder at the end of the process.
        
        Keyword arguments: 
            
        dr_loss (numpy array): loss function of the discriminator (real numbers)
        df_loss (numpy array): loss function of the discriminator (fake numbers)
        g_loss (numpy array):  loss function of the generator (fake numbers)
        dr_acc (numpy array):  accuracy of the discriminator (real numbers)
        df_acc (numpy array):  accuracy of the discriminator (fake numbers)
        path (str):            plot saving path 
        
        Returns:
        
        None
                              
        """
        fig_path = os.path.join(path, 'training_history.jpeg')
        figure, ax = plt.subplots(2, 1)
        plt.subplot(2, 1, 1)
        plt.plot(epochs, loss, label = 'training loss')
        plt.plot(epochs, val_loss, label = 'validation loss')
        plt.legend(loc = 'best', fontsize = 8)
        plt.title('Loss plot')
        plt.ylabel('mean squared error')
        plt.xlabel('epoch')
        plt.subplot(2, 1, 2)
        plt.plot(epochs, metric, label = 'train metrics')  
        plt.plot(epochs, val_metric, label = 'validation metrics') 
        plt.legend(loc = 'best', fontsize = 8)
        plt.title('metrics plot')
        plt.ylabel('mean absolute error')
        plt.xlabel('epoch')       
        plt.tight_layout()
        plt.savefig(fig_path, bbox_inches = 'tight', format = 'jpeg', dpi = 400)  

        return figure
        
    # thread version of the training loop  
    #==========================================================================
    def Conv1DLSTM_training_thread(self, model, batch_size, num_epochs,  
                                   X_train, X_test, Y_train, Y_test, path, 
                                   window, pbar): 
        epochs = []
        loss = []
        val_loss = [] 
        metric = []
        val_metric = []             
        for id, epoch in enumerate(range(num_epochs)):
            training_session = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), verbose = 2, 
                                         batch_size = batch_size, epochs = 1)            
            loss.append(training_session.history['loss'])
            val_loss.append(training_session.history['val_loss'])
            metric.append(training_session.history['mae'])
            val_metric.append(training_session.history['val_mae'])
            epochs.append(epoch + 1)            
            if epoch % 2 == 0:
                figure = self.plot_history(epochs, loss, val_loss, metric, val_metric, path)
                if GlobVar.ML_canvas_status == True:
                    fig_canvas.get_tk_widget().pack_forget()
                    GlobVar.ML_canvas_status = False                 
                fig_canvas = FigureCanvasTkAgg(figure, master = window['-CANVAS-'].TKCanvas)
                fig_canvas.draw()
                fig_canvas.get_tk_widget().pack(side='top', fill='none', expand=False) 
                GlobVar.ML_canvas_status = True   
                GlobVar.history_plot = figure
            pbar.update(id + 1, max=num_epochs)      
        
        return training_session
    


    #==========================================================================
    def training_logger(self, parameters):        
        raw_today_datetime = str(datetime.now())
        truncated_datetime = raw_today_datetime[:-7]
        today_datetime = truncated_datetime
        for rep in ('-', ':', ' '):
            today_datetime = today_datetime.replace(rep, '_')
        parameters.update({'today_datetime' : today_datetime})
        run_report_path = os.path.join(parameters['path'], 'training_log.txt')        
        if os.path.isfile(run_report_path):
            with open(run_report_path, 'a') as f:
                f.write('------------------------------------------------------\n')
                for key, val in parameters.items():
                    f.write('{0} = {1}\n'.format(key, val))
        else:
            with open(run_report_path, 'w') as f:
                f.write('------------------------------------------------------\n')
                for key, val in parameters.items():
                    f.write('{0} = {1}\n'.format(key, val))
                    
    
    #==========================================================================
    def model_savefolder(self, path, model_name):
        
        raw_today_datetime = str(datetime.now())
        truncated_datetime = raw_today_datetime[:-10]
        today_datetime = truncated_datetime.replace(':', '')
        today_datetime = today_datetime.replace('-', '')
        today_datetime = today_datetime.replace(' ', 'H')        
        model_name = '{0}_{1}'.format(model_name, today_datetime)
        model_savepath = os.path.join(path, model_name)
        if not os.path.exists(model_savepath):
            os.mkdir(model_savepath)              
            
        return model_savepath    

    