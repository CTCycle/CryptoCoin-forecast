import os
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests as r
import json

# define the class for inspection of the input folder and generation of files list.
#==============================================================================
#==============================================================================
#==============================================================================
class WebDriver:
    
    """
    Initializes a webdriver instance with Chrome options set to disable images loading.
    
    Keyword arguments:
    
    wd_path (str): The file path to the Chrome webdriver executable
    
    Returns:
        
    None 
    
    """
    def __init__(self, wd_path):
        self.wd_path = wd_path
        self.option = webdriver.ChromeOptions()
        self.chrome_prefs = {}
        self.option.experimental_options['prefs'] = self.chrome_prefs
        self.chrome_prefs['profile.default_content_settings'] = {'images': 2}
        self.chrome_prefs['profile.managed_default_content_settings'] = {'images': 2}  
        
        self.max_page_num = 1 # how many pages you will scrape
        self.num_cryptos = self.max_page_num * 100   
        self.fetch_ht = 3 # how many coin histories you will scrape
     
      
    
    
    
    
# define the class for inspection of the input folder and generation of files list
#==============================================================================
#==============================================================================
#==============================================================================
class CoingeckoScraper:    

    """
    Initializes a webdriver instance with Chrome options set to disable images loading.
    
    Keyword arguments:
    
    wd_path (str): The file path to the Chrome webdriver executable
    
    Returns:
        
    None 
    """
        
    
    # function to retrieve HTML data
    #==========================================================================
    def HTML_tablextractor(self, driver, url):
        
        """
        Extracts table with data from HTML
        
        Keyword arguments:
        
        HTML_obj (BeautifulSoup): A webdriver instance.
                
        Returns:
            
        main_data (BeautifulSoup): BeautifulSoup object containing the HTML content of the page
            
        """ 
        driver.get(url)
        wait = WebDriverWait(driver, 3)   
        soup = BeautifulSoup(driver.page_source, 'lxml')        
        
        self.main_table = soup.find_element(By.CLASS_NAME, 'coingecko-table')
        main_data = self.main_table.find_all('tr')[1:]
        
        return main_data
    
    # sending request to coingecko website 
    #==========================================================================
    def fetch_coin_history(self, coin_name, to_datetime = False):
        
        """
        fetch_coin_history(coin_name, to_datetime = False)

        Extract historical statistics and trading volume information for a single 
        coin from CoinGecko website
        
        Keyword arguments:
        
        coin_name (str):    The name of the coin to scrape data for
        to_datetime (bool): Whether to convert the date column to datetime format
                   
        Returns:
            
        df_coin_data (pd.DataFrame): A DataFrame containing historical statistics information
        
        """
        self.coin_name = coin_name.lower().replace(' ', '-')
        vs_currency = 'usd'
        self.coin_url = f'https://api.coingecko.com/api/v3/coins/{self.coin_name}/market_chart?vs_currency={vs_currency}&days=max'
        response = r.get(self.coin_url)
        raw_data = json.loads(response.text)                
        coin_data = raw_data['prices']        
        timesteps = [pair[0] for pair in coin_data]
        prices = [pair[1] for pair in coin_data]                
        df_coin_data = pd.DataFrame({'Date' : timesteps, 'Prices (usd)' : prices})       
        if to_datetime == True:
            df_coin_data['Date'] = pd.to_datetime(df_coin_data['Date'], 
                                                  unit = 'ms', utc = True).dt.tz_localize(None)        
        
        return df_coin_data