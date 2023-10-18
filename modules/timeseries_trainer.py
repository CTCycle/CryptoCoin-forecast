# [IMPORT PACKAGES AND SETTING WARNINGS]
#==============================================================================
import os
import sys
import numpy as np
import pickle
import threading
from keras.utils.vis_utils import plot_model
import PySimpleGUI as sg
import warnings
warnings.simplefilter(action='ignore', category = DeprecationWarning)
warnings.simplefilter(action='ignore', category = FutureWarning)

# [IMPORT MODULES AND CLASSES]
#==============================================================================
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.components.training_classes import PreProcessingTimeSeries, TrainingModels
import modules.global_variables as GlobVar

# [DEFINE PATHS]
#==============================================================================
if getattr(sys, 'frozen', False):    
    pretrained_path = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'pretrained models')   
    data_path = os.path.join(os.path.dirname(sys.executable), 'dataset')          
else:    
    pretrained_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pretrained models')  
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset') 

if not os.path.exists(pretrained_path):
    os.mkdir(pretrained_path)       
  
        
# [WINDOW THEME AND OPTIONS]
#==============================================================================
sg.theme('LightGrey1')
sg.set_options(font = ('Arial', 11), element_padding = (10,10))

# [LAYOUT OF THE PREPROCESSING FRAME]
#==============================================================================
pp_text = sg.Text('Timeseries has not been preprocessed', font = ('Arial', 12), key = '-PPTEXT-')
testsize_input_text = sg.Text('Test size', size = (8,1), expand_x = True, font = ('Arial', 11))
test_size_input = sg.Input(key = '-TESTSIZE-', default_text= '0.2', size = (8,1), expand_x = True, enable_events=True)
norm_checkbox = sg.Checkbox('Normalize data', key = '-NORMCHECK-')
savedata_checkbox = sg.Checkbox('Save preprocessed data', key = '-SAVECHECK-')
preprocess_button = sg.Button('Preprocess data', key = '-PREPROCESS-', expand_x= True)
pp_frame = sg.Frame('Preprocessing parameters', layout = [[testsize_input_text, test_size_input],
                                                          [norm_checkbox],
                                                          [savedata_checkbox],                                                          
                                                          [preprocess_button],
                                                          [pp_text]], expand_x = True)

# [LAYOUT OF THE TRAINING PARAMETERS FRAME]
#==============================================================================
dev_input_text = sg.Text('Training device', size = (8,1), expand_x = True, font = ('Arial', 10))
devices_input = sg.DropDown(['CPU', 'GPU'], default_value='CPU', size = (8,1), key = '-DEVICE-', expand_x = True, enable_events=True)
model_input_text = sg.Text('AI model', size = (8,1), expand_x = True, font = ('Arial', 10))
model_input = sg.DropDown(['Conv1DLSTM', 'ConvToConv'], default_value='Conv1DLSTM', size = (8,1), key = '-MODELS-', expand_x = True, enable_events=True)
lr_input_text = sg.Text('Learning rate', size = (8,1), expand_x = True, font = ('Arial', 10))
learning_rate_input = sg.Input(key = '-LR-', default_text= '0.001', size = (8,1), expand_x = True, enable_events=True)
epochs_input_text = sg.Text('Epochs', size = (8,1), expand_x = True, font = ('Arial', 10))
epochs_input = sg.Input(key = '-EPOCHS-', size = (8,1), default_text = '100', expand_x = True, enable_events=True)
ws_input_text = sg.Text('Window size', size = (8,1),  expand_x = True, font = ('Arial', 10))
window_size_input = sg.Input(key = '-WS-', size = (8,1), default_text= '30', expand_x = True, enable_events=True)
bs_input_text = sg.Text('Batch size', size = (8,1), expand_x = True, font = ('Arial', 10))
batch_size_input = sg.Input(key = '-BS-', size = (8,1), default_text= '32', expand_x = True, enable_events=True)
pretrain_button = sg.Button('Pretrain model', key = '-PRETRAIN-', expand_x= True, disabled=True)
modelshow_button = sg.Button('Show model scheme', key = '-SHOWCASE-', expand_x= True)
pt_frame = sg.Frame('Pretraining parameters', layout = [[dev_input_text, devices_input],
                                                        [model_input_text, model_input],
                                                        [lr_input_text, learning_rate_input],
                                                        [ws_input_text, window_size_input], 
                                                        [epochs_input_text, epochs_input],                                                        
                                                        [bs_input_text, batch_size_input],
                                                        [modelshow_button],
                                                        [pretrain_button]], expand_x = True)
                                   
# [LAYOUT OF OUTPUT AND CANVAS]
#==============================================================================
output = sg.Output(size = (100, 10), key = '-OUTPUT-', expand_x = True)
canvas_object = sg.Canvas(key='-CANVAS-', size=(500, 500), expand_x=True)

# [LAYOUT OF THE WINDOW]
#==============================================================================
left_column = sg.Column(layout = [[pp_frame], [pt_frame]])
right_column = sg.Column(layout = [[canvas_object]])
progress_bar = sg.ProgressBar(100, orientation = 'horizontal', size = (50, 20), key = '-PBAR-', expand_x=True)
training_layout = [[left_column, sg.VSeparator(), right_column],
                   [sg.HSeparator()],
                   [output],
                   [progress_bar]]                           

# [WINDOW LOOP]
#==============================================================================
training_window = sg.Window('Pretraining using machine learning', training_layout, 
                            grab_anywhere = True, resizable=True, finalize = True)

while True:
    event, values = training_window.read()

    if event == sg.WIN_CLOSED:
        break                     

    # [GENERATE VIASUAL SCHEME OF THE MODEL]
    #==========================================================================
    if event == '-SHOWCASE-':
        learning_rate = float(values['-LR-'])
        if values['-WS-'].isdigit():
            window_size = int(values['-WS-'])
        else:
            window_size = 30        
        model_name = values['-MODELS-']
        model_picture_name = '{}_model.png'.format(model_name)
        model_plot_path = os.path.join(pretrained_path, model_picture_name) 
        training_routine = TrainingModels(device = 'CPU') 
        if values['-MODELS-'] ==  'Conv1DLSTM':
            ML_model = training_routine.Conv1DLSTM_model(window_size, learning_rate)
            ML_model.summary()
            plot_model(ML_model, to_file = model_plot_path, show_shapes = True, 
                        show_layer_names = True, show_layer_activations = True, 
                        expand_nested = True, rankdir = 'TB', dpi = 400)           
        

    # [PREPROCESS DATA USING ADEQUATE PIPELINE]
    #==========================================================================
    if event == '-PREPROCESS-':          
        test_size = float(values['-TESTSIZE-']) 
        if values['-WS-'].isdigit():
            window_size = int(values['-WS-'])
        else:
            window_Size = 30        
        pp_dataframe = GlobVar.dataframe        
        coin_name = GlobVar.dataframe_name
        index_pp_dataframe = pp_dataframe.set_index('Date')                 
        data_preprocessing = PreProcessingTimeSeries()
        df_train, df_test = data_preprocessing.split_timeseries(index_pp_dataframe, test_size)            
        if values['-NORMCHECK-'] == True:
            df_train, df_test = data_preprocessing.timeseries_normalization(df_train, df_test) 
            normalizer = data_preprocessing.normalizer 
            GlobVar.normalizer = normalizer       
        preprocessed_data = data_preprocessing.timeseries_labeling(df_train, df_test, window_size) 
        GlobVar.preprocessed_data = preprocessed_data      
        if values['-SAVECHECK-'] == True:
            pp_data_path = os.path.join(pretrained_path, 'preprocessed_{}'.format(coin_name))
            if not os.path.exists(pp_data_path):
                os.mkdir(pp_data_path) 
            savepath = os.path.join(pp_data_path, 'preprocessed_data.npz')
            np.savez(savepath, X_train = preprocessed_data['train_X'], Y_train = preprocessed_data['train_Y'],
                     X_test = preprocessed_data['test_X'], Y_test = preprocessed_data['test_Y'])
            if values['-NORMCHECK-'] == True:
                normalizer_path = os.path.join(pp_data_path, 'normalizer.pkl')
                with open(normalizer_path, 'wb') as file:
                    pickle.dump(normalizer, file)                                                 
        
        training_window['-PPTEXT-'].update('{} timeseries has been preprocessed'.format(coin_name.upper()))
        training_window['-PRETRAIN-'].update(disabled = False)        
        

    # [SELECT TIME SERIES USING DROPDOWN MENU]
    #==========================================================================
    if event == '-PRETRAIN-':             
        model_name = values['-MODELS-'] 
        device_name = values['-DEVICE-']
        learning_rate = float(values['-LR-'])
        if values['-WS-'].isdigit():
            window_size = int(values['-WS-'])
        else:
            window_size = 30
        if values['-BS-'].isdigit():
            batch_size = int(values['-BS-'])
        else:
            batch_size = 32
        if values['-EPOCHS-'].isdigit():
            epochs = int(values['-BS-'])
        else:
            epochs = 100       

        parameters = {'timeseries' : GlobVar.dataframe_name, 
                      'window size' : window_size, 
                      'test size' : test_size}
        
        preprocessed_data = GlobVar.preprocessed_data
        training_routine = TrainingModels(device=device_name)
        X_train, Y_train = preprocessed_data['train_X'], preprocessed_data['train_Y']
        X_test, Y_test = preprocessed_data['test_X'], preprocessed_data['test_Y'] 
        model_savepath = training_routine.model_savefolder(pretrained_path, model_name)        
        if values['-MODELS-'] == 'Conv1DLSTM':     
            Conv1DLSTM_model = training_routine.Conv1DLSTM_model(window_size, learning_rate)       
            training_thread = threading.Thread(target = training_routine.Conv1DLSTM_training_thread,
                                               args = (Conv1DLSTM_model, batch_size, epochs, 
                                                       X_train, X_test, Y_train, Y_test, 
                                                       model_savepath, training_window, progress_bar))                                               
            training_thread.start()
  

training_window.close() 
        
 


        

        

            
        
              
  

        

        
        
        
        
        

    



