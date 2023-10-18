import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import adfuller


# define class for trained model validation and data comparison
#============================================================================== 
#==============================================================================
#==============================================================================
class ValidationLSTM:    
    
    
    # comparison of histograms (distributions) by superimposing plots
    #========================================================================== 
    def timeseries_hist(self, timeseries, bins, name, path, dpi):
        
        """ 
        timeseries_hist(timeseries, bins, name, path, dpi)
        
        Plots a histogram of the given time series data and saves it as a JPEG image.
    
        Keyword arguments:
            
        timeseries (pd.DataFrame): Time series data.
        bins (int):                Number of histogram bins.
        name (str):                Name of the time series.
        path (str):                Path to save the histogram figure.
        dpi (int):                 DPI for the JPEG image.
    
        Returns:
            
        None
        
        """        
        array = timeseries.values        
        fig, ax = plt.subplots()
        plt.hist(array, bins = bins, density = True, label = 'timeseries',
                 rwidth = 1, edgecolor = 'black')        
        plt.legend(loc='upper right')
        plt.title('Histogram of {}'.format(name))
        plt.xlabel('Time', fontsize = 8)
        plt.ylabel('Norm frequency', fontsize = 8) 
        plt.xticks(fontsize = 8)
        plt.yticks(fontsize = 8)
        plt.tight_layout()
        plot_loc = os.path.join(path, 'histogram_timeseries_{}.jpeg'.format(name))
        plt.savefig(plot_loc, bbox_inches='tight', format ='jpeg', dpi = dpi)
        #plt.show(block = False) 

        return fig
    
    
    # comparison of data distribution using statistical methods 
    #========================================================================== 
    def test_stationarity(self, timeseries, name, path, dpi):
            
        ADF = adfuller(timeseries)                    
        fig, ax = plt.subplots(3, sharex = True, sharey = False)                  
        ax[0].set_title('Original')
        ax[1].set_title('Rolling Mean')
        ax[2].set_title('Rolling Standard Deviation')
        plt.xlabel('Time', fontsize = 8)
        plt.xticks(fontsize = 8)
        plt.yticks(fontsize = 8)
        plt.tight_layout()                       
        plot_loc = os.path.join(path, 'timeseries_analysis_{}.jpeg'.format(name))
        plt.savefig(plot_loc, bbox_inches='tight', format ='jpeg', dpi = dpi)
        #plt.show(block = False)
        
        return ADF, fig
          
    # comparison of data distribution using statistical methods 
    #==========================================================================     
    def predictions_vs_test(self, Y_test, predictions, path, dpi):       
            
        plt.figure()
        plt.plot(Y_test, color='blue', label = 'test values')
        plt.plot(predictions, color='red', label = 'predictions')
        plt.legend(loc='best')
        plt.title('Test values vs Predictions')
        plt.xlabel('Time', fontsize=8)
        plt.ylabel('Values', fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        plot_loc = os.path.join(path, 'test_vs_predictions.jpeg')
        plt.savefig(plot_loc, bbox_inches='tight', format='jpeg', dpi = dpi)
        plt.show(block=False)
       
            
