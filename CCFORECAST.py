import os
import sys
import pandas as pd
import datetime
import PySimpleGUI as sg
import warnings

# set warnings
#------------------------------------------------------------------------------
import warnings
warnings.simplefilter(action='ignore', category = Warning)

# default folder path
#------------------------------------------------------------------------------
initial_folder = os.path.dirname(os.path.realpath(__file__))

# import modules and classes
#------------------------------------------------------------------------------ 
from modules.components.scraper_classes import CoingeckoScraper
from modules.components.data_classes import DataSetFinder
import modules.global_variables as GlobVar

# [DEFINE PATHS]
#==============================================================================
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')

if not os.path.exists(data_path):
    os.mkdir(data_path)     

# [GENERATE LIST OF FILES]
#==============================================================================
dataset_inspector = DataSetFinder(data_path)
list_of_files = dataset_inspector.target_files
coin_names = [x.split('_')[0] for x in list_of_files] 

# [WINDOW THEME AND OPTIONS]
#==============================================================================
sg.theme('LightGrey1')
sg.set_options(font = ('Arial', 11), element_padding = (6, 6))

# [LAYOUT OF DATA SCRAPER FRAME]
#==============================================================================
coinsearch_text = sg.Text('Type the coin name and and collect data from coingecko.com', font = ('Arial', 11),
                          expand_x = True, auto_size_text = True)
coiname_input = sg.Input(key = '-COINAME-')
search_button = sg.Button('Search', key = '-COINSEARCH-')   
dd_text = sg.Text('Select cryptocoin history data', font = ('Arial', 11), expand_x = True, auto_size_text = True)
coin_dropdown = sg.DropDown(coin_names, size = (20,1), key = '-DROPDOWN-', expand_x = True, enable_events=True)
scraper_frame = sg.Frame('Collect cryptocoin data', font = ('Arial', 12), layout = [[coinsearch_text],                                                                                                                              
                                                                                    [coiname_input, search_button],
                                                                                    [dd_text],
                                                                                    [coin_dropdown]]) 

# [LAYOUT OF MACHINE LEARNING FRAME]
#==============================================================================
ts_analysis_button = sg.Button('Timeseries analysis', key = '-TSANALYSIS-', disabled=True, expand_x=True)
pretraining_button = sg.Button('Pretraining on cryptocurrencies data', key = '-PRETRAIN-', disabled=True, expand_x=True)
forecast_button = sg.Button('Price forecast with pretrained model', key = '-FORECAST-', disabled=True, expand_x=True)                                                         
ml_frame = sg.Frame('Cryptocoin data analysis', layout = [[ts_analysis_button],
                                                          [pretraining_button],
                                                          [forecast_button]], expand_x=True)

# [LAYOUT OF THE WINDOW]
#==============================================================================
left_column = sg.Column(layout = [[scraper_frame]])
right_column = sg.Column(layout = [[ml_frame]], expand_x=True)
output_window = sg.Output(size = (60, 5), key = '-OUTPUT-', expand_x = True)
main_layout = [[left_column, sg.VSeparator(), right_column],
               [output_window]]              

# [WINDOW LOOP]
#==============================================================================
main_window = sg.Window('CryptoCoin Forecast V1.0', main_layout, 
                        grab_anywhere = True, resizable=True, finalize = True)
while True:
    event, values = main_window.read()

    if event == sg.WIN_CLOSED:
        break  

    # [COLLECT DATA USING API ACCESS]
    #==========================================================================
    if event == '-COINSEARCH-':        
        current_datetime = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        try:
            coin_name = values['-COINAME-'] 
            API_scraper = CoingeckoScraper()
            df_prices = API_scraper.fetch_coin_history(coin_name, to_datetime = True)
            file_loc = os.path.join(data_path, '{}_history.csv'.format(coin_name))    
            df_prices.to_csv(file_loc, index = False, sep = ';', encoding = 'utf-8')
            print('[{0}]: {1} data has been collected and saved in the desired folder'.format(current_datetime, coin_name))   
        except:
            del values['-COINAME-']
            print('[{0}]: Something went wrong with the API request for "{1}"'.format(current_datetime, coin_name))
        
        dataset_inspector = DataSetFinder(data_path)
        list_of_files = dataset_inspector.target_files
        coin_names = [x.split('_')[0] for x in list_of_files] 
        main_window['-DROPDOWN-'].update(values = coin_names)        
        main_window['-COINAME-'].update('')

    # [COLLECT DATA USING API ACCESS]
    #==========================================================================
    if event == '-DROPDOWN-':
        target_coin = values['-DROPDOWN-'] + '_history.csv'        
        GlobVar.dataframe_name = values['-DROPDOWN-']                
        filepath = os.path.join(data_path, target_coin)        
        df = pd.read_csv(filepath, sep= ';', encoding='utf-8')
        GlobVar.dataframe = df   
        main_window['-TSANALYSIS-'].update(disabled=False) 
        main_window['-PRETRAIN-'].update(disabled=False) 
        main_window['-FORECAST-'].update(disabled=False)    

    # [ANALYSIS OF DATA]
    #==========================================================================
    if event == '-TSANALYSIS-':
        import modules.timeseries_analyzer
        del sys.modules['modules.timeseries_analyzer']

    if event == '-PRETRAIN-':
        import modules.timeseries_trainer
        del sys.modules['modules.timeseries_trainer']
    
    if event == '-FORECAST-':
        import modules.timeseries_forecaster
        del sys.modules['modules.timeseries_forecaster']     

main_window.close()





