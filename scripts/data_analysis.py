import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from statsmodels.tsa.seasonal import seasonal_decompose


class DataAnalysis:

    def __init__(self,df_tsla,df_sp,df_bnd):
        self.df_tsla = df_tsla
        self.df_sp = df_sp
        self.df_bnd = df_bnd

    
    def plot_closing_price(self):
        """plots closing prices over time"""

        plt.figure(figsize=(12,8))
        plt.plot(self.df_tsla['Close'],label='TSLA', color='blue')
        plt.plot(self.df_sp['Close'],label='SPY', color='red')
        plt.plot(self.df_bnd['Close'],label='BND', color='green')
        plt.title('Closing Price Over Time',fontsize=14, fontweight='bold')
        plt.xlabel('Time',fontsize=12, fontweight='bold')
        plt.ylabel('Price',fontsize=12, fontweight='bold')
        plt.legend(fontsize=12)
        plt.show()

    def pct_change_plot(self):
        #Calculate daily percentage change
        pct_change_tsla = self.df_tsla['Adj Close'].pct_change(1).dropna()
        pct_change_sp = self.df_sp['Adj Close'].pct_change(1).dropna()
        pct_change_bnd = self.df_bnd['Adj Close'].pct_change(1).dropna()

        plt.figure(figsize=(12,8))
        plt.plot(pct_change_tsla,label='TSLA', color='blue')
        plt.plot(pct_change_sp,label='SPY', color='red')
        plt.plot(pct_change_bnd,label='BND', color='green')
        plt.title('Percentage Change Graph',fontsize=14, fontweight='bold')
        plt.xlabel('Time',fontsize=12, fontweight='bold')
        plt.ylabel('Percentage Change',fontsize=12, fontweight='bold')
        plt.legend(fontsize=12)
        plt.show()

    def rolling_mean_plot(self):
        #Calculate Rolling Std and std
        rolling_mean_tsla = self.df_tsla['Adj Close'].rolling(window=30).mean()
        rolling_mean_sp= self.df_sp['Adj Close'].rolling(window=30).mean()
        rolling_mean_bnd = self.df_bnd['Adj Close'].rolling(window=30).mean()

        #Plot Rolling Mean
        plt.figure(figsize=(12,8))
        plt.plot(rolling_mean_tsla,label='TSLA', color='blue')
        plt.plot(rolling_mean_sp,label='SPY', color='red')
        plt.plot(rolling_mean_bnd,label='BND', color='green')
        plt.title('Rolling Mean Graph',fontsize=14, fontweight='bold')
        plt.xlabel('Time',fontsize=12, fontweight='bold')
        plt.ylabel('Rolling Mean',fontsize=12, fontweight='bold')
        plt.legend(fontsize=12)
        plt.show()
        

    def rolling_std_plot(self):
        rolling_std_bnd = self.df_bnd['Adj Close'].rolling(window=30).std()
        rolling_std_tsla = self.df_tsla['Adj Close'].rolling(window=30).std()
        rolling_std_sp = self.df_sp['Adj Close'].rolling(window=30).std()

        plt.figure(figsize=(12,8))
        plt.plot(rolling_std_tsla,label='TSLA', color='blue')
        plt.plot(rolling_std_sp,label='SPY', color='red')
        plt.plot(rolling_std_bnd,label='BND', color='green')
        plt.title('Rolling Std Graph',fontsize=14, fontweight='bold')
        plt.xlabel('Time',fontsize=12, fontweight='bold')
        plt.ylabel('Rolling Std',fontsize=12, fontweight='bold')
        plt.legend(fontsize=12)
        plt.show()

    def outlier_detection(self):
        figure, axes = plt.subplots(1,3,figsize=(16,4))
        #TSLA
        sns.boxplot(self.df_tsla['Adj Close'], ax=axes[0] )
        axes[0].set_title('Box Plot for TSLA',fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Adj Close',fontsize=12, fontweight='bold')
        #S&P
        sns.boxplot(self.df_sp['Adj Close'], ax=axes[1])
        axes[1].set_title('Box Plot for S&P',fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Adj Close',fontsize=12, fontweight='bold')
        #BND
        sns.boxplot(self.df_bnd['Adj Close'], ax=axes[2])
        axes[2].set_title('Box Plot for BND',fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Adj Close',fontsize=12, fontweight='bold')

    def seasonal_decomposition_tsla(self):
        #Seasonal Decomposition
        decompose = seasonal_decompose(self.df_tsla['Close'], model='multiplicative',period=252)

        trend = decompose.trend
        seasonal = decompose.seasonal
        resid = decompose.resid

        # Create a figure and subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))

        # Plot the observed data
        ax1.plot(self.df_tsla['Close'], label='Observed', color='blue')
        ax1.legend(loc='upper left')
        ax1.set_title('Observed (TSLA)',fontsize=12, fontweight='bold')

        # Plot the trend component
        ax2.plot(trend, label='Trend', color='orange')
        ax2.legend(loc='upper left')
        ax2.set_title('Trend (TSLA)',fontsize=12, fontweight='bold')

        # Plot the seasonal component
        ax3.plot(seasonal, label='Seasonal', color='green')
        ax3.legend(loc='upper left')
        ax3.set_title('Seasonal (TSLA)',fontsize=12, fontweight='bold')

        # Plot the residual component
        ax4.plot(resid, label='Residual', color='red')
        ax4.legend(loc='upper left')
        ax4.set_title('Residual (TSLA)',fontsize=12, fontweight='bold')

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

    def seasonal_decomposition_sp(self):
        #Seasonal Decomposition
        decompose = seasonal_decompose(self.df_sp['Close'], model='multiplicative',period=252)

        trend = decompose.trend
        seasonal = decompose.seasonal
        resid = decompose.resid

        # Create a figure and subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))

        # Plot the observed data
        ax1.plot(self.df_sp['Close'], label='Observed', color='blue')
        ax1.legend(loc='upper left')
        ax1.set_title('Observed (SPY)',fontsize=12, fontweight='bold')

        # Plot the trend component
        ax2.plot(trend, label='Trend', color='orange')
        ax2.legend(loc='upper left')
        ax2.set_title('Trend (SPY)',fontsize=12, fontweight='bold')

        # Plot the seasonal component
        ax3.plot(seasonal, label='Seasonal', color='green')
        ax3.legend(loc='upper left')
        ax3.set_title('Seasonal (SPY)',fontsize=12, fontweight='bold')

        # Plot the residual component
        ax4.plot(resid, label='Residual', color='red')
        ax4.legend(loc='upper left')
        ax4.set_title('Residual (SPY)',fontsize=12, fontweight='bold')

        # Adjust layout and display
        plt.tight_layout()
        plt.show()

    def seasonal_decomposition_bnd(self):
        #Seasonal Decomposition
        decompose = seasonal_decompose(self.df_bnd['Close'], model='multiplicative',period=252)

        trend = decompose.trend
        seasonal = decompose.seasonal
        resid = decompose.resid

        # Create a figure and subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))

        # Plot the observed data
        ax1.plot(self.df_bnd['Close'], label='Observed', color='blue')
        ax1.legend(loc='upper left')
        ax1.set_title('Observed (BND)',fontsize=12, fontweight='bold')

        # Plot the trend component
        ax2.plot(trend, label='Trend', color='orange')
        ax2.legend(loc='upper left')
        ax2.set_title('Trend (BND)',fontsize=12, fontweight='bold')

        # Plot the seasonal component
        ax3.plot(seasonal, label='Seasonal', color='green')
        ax3.legend(loc='upper left')
        ax3.set_title('Seasonal (BND)',fontsize=12, fontweight='bold')

        # Plot the residual component
        ax4.plot(resid, label='Residual', color='red')
        ax4.legend(loc='upper left')
        ax4.set_title('Residual (BND)',fontsize=12, fontweight='bold')

        # Adjust layout and display
        plt.tight_layout()
        plt.show()