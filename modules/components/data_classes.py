import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ...
#==============================================================================
#==============================================================================
#==============================================================================
class UserOperations:
    
    """    
    A class for user operations such as interactions with the console, directories and files
    cleaning and other maintenance operations.
      
    Methods:
        
    menu_selection(self, menu): console menu management 
   
    """    
    
    
    #==========================================================================
    def clear_all_files(self, folder_path):
        
        try:
            shutil.rmtree(folder_path)
            print('All files have been deleted.')
            
        except FileNotFoundError:
            print('Files do not exist.')
            
        except Exception as e:
            print('Error deleting files: {}'.format(e))
            
    
      
# define the class for inspection of the input folder and generation of files list.
# The extension as argument allows identifying specific files (.csv, .xlsx, .pdf, etc)
# and making a list of those than can be called with the 'target_files' method
#==============================================================================
#==============================================================================
#==============================================================================
class DataSetFinder:
    
    """ 
    DatasetFinder(path)
    
    Creates a list of files with given extension from the argument path, if the
    list is empty because no files were found, then the emtpy_folder variable is
    set to True
    
    Keyword arguments:
        
    path (str): path of the folder that has to be scanned for files    
    ext (str):  the target files extension (.csv, .jpeg, .pdf, etc)
    
    Returns:
        
    None 
    
    """
    def __init__(self, path):        
        self.path = path        
        os.chdir(path)
        self.all_files = os.listdir(path)
        self.target_files = [f for f in self.all_files if f.endswith('.csv')]        
        if len(self.target_files) == 0:
            self.target_files = ['No files have been found!']
    
          
      
    
    
   
     
# define class for correlations calculations
#==============================================================================
#==============================================================================
#==============================================================================
class MultiCorrelator:
    
    """ 
    multicorrelator(dataframe)
    
    Calculates the correlation matrix of a given dataframe using specific methods.
    The internal functions retrieves correlations based on Pearson, Spearman and Kendall
    methods. This class is also used to plot the correlation heatmap and filter correlations
    from the original matrix based on given thresholds. Returns the correlation matrix
    
    Keyword arguments: 
        
    dataframe (pd.dataframe): target dataframe
    
    Returns:
        
    df_corr (pd.dataframe): correlation matrix in dataframe form
                
    """
    def __init__(self, dataframe):
        self.dataframe = dataframe
        
    # Spearman correlation calculation
    #--------------------------------------------------------------------------
    def Spearman_corr(self, dataframe, decimals):
        self.df_corr = dataframe.corr(method = 'spearman').round(decimals)
        return self.df_corr
    
    # Kendall correlation calculation
    #--------------------------------------------------------------------------    
    def Kendall_corr(self, dataframe, decimals):
        self.df_corr = dataframe.corr(method = 'kendall').round(decimals)
        return self.df_corr
    
    # Pearson correlation calculation
    #--------------------------------------------------------------------------    
    def Pearson_corr(self, dataframe, decimals):
        self.df_corr = dataframe.corr(method = 'pearson').round(decimals)
        return self.df_corr
    
    # plotting correlation heatmap using seaborn package
    #--------------------------------------------------------------------------
    def corr_heatmap(self, matrix, path, dpi, name):
        
        """ 
        corr_heatmap(matrix, path, dpi, name)
        
        Plot the correlation heatmap using the seaborn package. The plot is saved 
        in .jpeg format in the folder that is specified through the path argument. 
        Output quality can be tuned with the dpi argument.
        
        Keyword arguments:    
            
        matrix (pd.dataframe): target correlation matrix
        path (str):            picture save path for the .jpeg file
        dpi (int):             value to set picture quality when saved (int)
        name (str):            name to be added in title and filename
        
        Returns:
            
        None
            
        """
        cmap = 'YlGnBu'
        sns.heatmap(matrix, square = True, annot = False, 
                    mask = False, cmap = cmap, yticklabels = False, 
                    xticklabels = False)
        plt.title('{}_correlation_heatmap'.format(name))
        plt.tight_layout()
        plot_loc = os.path.join(path, '{}_correlation_heatmap.jpeg'.format(name))
        plt.savefig(plot_loc, bbox_inches='tight', format ='jpeg', dpi = dpi)
        plt.show(block = False)
        plt.close()
        
    # plotting correlation heatmap of two dataframes
    #--------------------------------------------------------------------------
    def double_corr_heatmap(self, matrix_real, matrix_fake, path, dpi):        
        
        """ 
        double_corr_heatmap(matrix_real, matrix_fake, path, dpi)
        
        Plot the correlation heatmap of two dataframes using the seaborn package. 
        The plot is saved in .jpeg format in the folder specified through the path argument. 
        Output quality can be tuned with the dpi argument.
        
        Keyword arguments:
            
        matrix_real (pd.dataframe): real data correlation matrix
        matrix_fake (pd.dataframe): fake data correlation matrix
        path (str):                 picture save path for the .jpeg file
        dpi (int):                  value to set picture quality when saved (int)
        
        Returns:
            
        None
        
        """ 
        plt.subplot(2, 1, 1)
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(matrix_real, square=True, annot=False, mask = False, 
                    cmap=cmap, yticklabels=False, xticklabels=False)
        plt.title('Real data')
        plt.subplot(2, 1, 2)
        sns.heatmap(matrix_fake, square=True, annot=False, mask = False, 
                    cmap=cmap, yticklabels=False, xticklabels=False)
        plt.title('Synthetic data')
        plt.tight_layout()
        plot_loc = os.path.join(path, 'Correlation_heatmap.jpeg')
        plt.savefig(plot_loc, bbox_inches='tight', format ='jpeg', dpi = dpi)
        plt.show()
        plt.close()
     
    # filtering of correlation pairs based on threshold value. Strong, weak and null
    # pairs are isolated and embedded into output lists
    #--------------------------------------------------------------------------    
    def corr_filter(self, matrix, threshold): 
        
        """
        corr_filter(matrix, path, dpi)
        
        Generates filtered lists of correlation pairs, based on the given threshold.
        Weak correlations are those below the threshold, strong correlations are those
        above the value and zero correlations identifies all those correlation
        with coefficient equal to zero. Returns the strong, weak and zero pairs lists
        respectively.
        
        Keyword arguments:    
        matrix (pd.dataframe): target correlation matrix
        threshold (float):     threshold value to filter correlations coefficients
        
        Returns:
            
        strong_pairs (list): filtered strong pairs
        weak_pairs (list):   filtered weak pairs
        zero_pairs (list):   filtered zero pairs
                       
        """        
        self.corr_pairs = matrix.unstack()
        self.sorted_pairs = self.corr_pairs.sort_values(kind="quicksort")
        self.strong_pairs = self.sorted_pairs[(self.sorted_pairs) >= threshold]
        self.strong_pairs = self.strong_pairs.reset_index(level = [0,1])
        mask = (self.strong_pairs.level_0 != self.strong_pairs.level_1) 
        self.strong_pairs = self.strong_pairs[mask]
        
        self.weak_pairs = self.sorted_pairs[(self.sorted_pairs) >= -threshold]
        self.weak_pairs = self.weak_pairs.reset_index(level = [0,1])
        mask = (self.weak_pairs.level_0 != self.weak_pairs.level_1) 
        self.weak_pairs = self.weak_pairs[mask]
        
        self.zero_pairs = self.sorted_pairs[(self.sorted_pairs) == 0]
        self.zero_pairs = self.zero_pairs.reset_index(level = [0,1])
        mask = (self.zero_pairs.level_0 != self.zero_pairs.level_1) 
        self.zero_pairs_P = self.zero_pairs[mask]
        
        return self.strong_pairs, self.weak_pairs, self.zero_pairs
