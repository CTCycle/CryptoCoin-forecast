# [IMPORT PACKAGES AND SETTING WARNINGS]
#==============================================================================
import os
import sys
import pandas as pd
import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.simplefilter(action='ignore', category = DeprecationWarning)
warnings.simplefilter(action='ignore', category = FutureWarning)

# [IMPORT MODULES AND CLASSES]
#==============================================================================
if __name__ == '__main__':
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from modules.components.validation_classes import ValidationLSTM
import modules.global_variables as GlobVar

# [DEFINE PATHS]
#==============================================================================
if getattr(sys, 'frozen', False):      
    out_path = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'results')  
else:      
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')

# Retrieve dataframe from global variables and set date to index
#==============================================================================
df_prices = GlobVar.dataframe   
coin_name = GlobVar.dataframe_name 
df = df_prices.set_index('Date')

# [WINDOW THEME AND OPTIONS]
#==============================================================================
sg.theme('LightGrey1')
sg.set_options(font = ('Arial', 11), element_padding = (6,6))



# [LAYOUT OF THE WINDOW]
#==============================================================================
canvas_draw = False
canvas_object = sg.Canvas(key='-CANVAS-', size=(600, 600), expand_x=True)
hist_button = sg.Button('Histogram analysis', key = '-HISTOGRAM-', expand_x= True)
TS_button = sg.Button('Time stationarity', key = '-STATIONARITY-', expand_x= True)
data_layout = [[[hist_button], [TS_button]],
                [sg.HSeparator()],                     
                [canvas_object]]                   

# [WINDOW LOOP]
#==============================================================================
data_window = sg.Window('Analysis of timeseries', data_layout, 
                        grab_anywhere = True, resizable=True, finalize = True)
while True:
    event, values = data_window.read()

    if event == sg.WIN_CLOSED:
        break    

    # [SELECT TIME SERIES USING DROPDOWN MENU]
    #==========================================================================
    if event == '-HISTOGRAM-':
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        if canvas_draw == True:
            fig_canvas.get_tk_widget().pack_forget()
            canvas_draw = False         
        data_validation = ValidationLSTM()                                
        figure = data_validation.timeseries_hist(df['Prices (usd)'], 'auto', coin_name, out_path, 600)        
        fig_canvas = FigureCanvasTkAgg(figure, master = data_window['-CANVAS-'].TKCanvas)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(side='top', fill='none', expand=False) 
        canvas_draw = True   
        
    
    # [SELECT TIME SERIES USING DROPDOWN MENU]
    #==========================================================================
    if event == '-STATIONARITY-':
        if not os.path.exists(out_path):
            os.mkdir(out_path)  
        if canvas_draw == True:
            fig_canvas.get_tk_widget().pack_forget()
            canvas_draw = False        
        data_validation = ValidationLSTM()                
        ADF, figure = data_validation.test_stationarity(df, coin_name, out_path, 600)
        ADF_statistics = ADF[0]
        P_value = ADF[1]
        critical_values = ADF[4]       
        if ADF[1] <= 0.05:
            conclusion = 'The timeseries is stationary'
        else:
            conclusion = 'The timeseries is not stationary'
        
        fig_canvas = FigureCanvasTkAgg(figure, master = data_window['-CANVAS-'].TKCanvas)
        fig_canvas.draw()
        fig_canvas.get_tk_widget().pack(side='top', fill='none', expand=False) 
        canvas_draw = True
        
            
data_window.close()

































