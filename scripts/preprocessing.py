import pandas as pd
import numpy as np
import yfinance as yf

import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def data_loader():
    # Load the datasets
    logging.info("Loading Datasets...")
    df_tsla = yf.download('TSLA', start='2015-01-01', end='2025-01-31',auto_adjust=False, multi_level_index=False)
    df_sp = yf.download('SPY', start='2015-01-01', end='2025-01-31',auto_adjust=False,multi_level_index=False)
    df_bnd = yf.download('BND', start='2015-01-01', end='2025-01-31',auto_adjust=False,multi_level_index=False)
    logging.info("✅ Dataset Successfully Loaded.")

    return df_tsla, df_sp, df_bnd

def check_for_missing_values(df_tsla, df_sp, df_bnd):
    
    logging.info("Checking for missing Values...")
    print('================================================================')
    print(f"Missing Values for TSLA: {df_tsla.isnull().sum()}")
    print('================================================================')
    print(f"Missing Values for S&P: {df_sp.isnull().sum()}")
    print('================================================================')
    print(f"Missing Values for BND: {df_bnd.isnull().sum()}")
    print('================================================================')
    logging.info("✅ Checking Successful.")

def general_info(df):
    print('================================================================')
    print(f"Missing Values for TSLA: {df.info()}")
    print('================================================================')
    logging.info("✅ General Information")

def general_statistics(df):
    print('===========================================================================================================')
    print(f"Missing Values for TSLA: {df.describe()}")
    print('===========================================================================================================')
    logging.info("✅ General Statistics")