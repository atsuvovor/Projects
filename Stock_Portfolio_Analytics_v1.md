<a href="https://colab.research.google.com/github/atsuvovor/Projects/blob/main/Stock_Portfolio_Analytics_v1.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


# Stock Portfolio Analytics


**Toronto, August, 30 2024**  

**Autor : Atsu Vovor**

>Master of Management in Artificial Intelligence,  

>Data Analytics and Reporting Professional | Machine Learning | Data science | Quantitative Analysis |French Bilingua  



Abstract

--------

This project presents the development of an advanced stock portfolio analytics tool designed to assist portfolio managers in optimizing investment strategies. By leveraging statistical analysis, mathematical and machine learning techniques, the tool provides insights into stock asset pricing, risk assessment, asset allocation, and performance forecasting. The project outlines the methodology used, including data collection and preprocessing, explanatory datanalysis, model selection and evaluation metrics, stress testing under economic key performance indicators scenarios. Results demonstrate the tool's effectiveness in enhancing decision-making processes, potentially leading to improved portfolio performance. The findings highlight the importance of integrating modern analytics into traditional portfolio management to navigate the complexities of today's financial markets.





Introduction

------------

The growing complexity of financial instruments and risk factors places significant pressure on portfolio managers, who must navigate and analyze a vast and intricate flow of data each day. Utilizing a robust dataset comprising historical stock prices, economic indicators, and financial metrics, our goal is to develop an advanced stock portfolio analysis tool that leverages advanced statistical methods, portfolio optimization and machine learning techniques to assist portfolio managers in making informed decisions. The tool provides insights into the asset pricing, risk assessment, asset allocation, and performance forecasting.



To achieve this goal, we begin by dynamically collecting real time data of all the S&P/TSX composite constituents adjust closed prices and canadian economic factors. The methodology used involves data preprocessing to ensure accuracy and relevance, followed by exploratory data analysis (EDA) to uncover key trends and correlations. Principal Component Analysis (PCA) is applied to reduce the dimensionality of the dataset, enabling the identification of the most influential factors affecting portfolio performance. We then use correlation analysis and hierarchical clustering to categorize stocks into distinct groups, facilitating diversification and risk management.



Moreover, the project explores advenced assets pricing technics sach as Stochastic Differencial Equation and Monte Carlo Simulation combined with modern portfolio theory (MPT) to simulate the portfolio price, profit & lost, risk and construct efficient portfolios, and stress testing techniques to evaluate portfolio robustness under various economic scenarios. The results demonstrate significant improvements in risk-adjusted returns, providing actionable insights for portfolio managers and investors.



In conclusion, this project underscores the importance of integrating advanced analytics into investment decision-making processes. The findings offer a valuable framework for optimizing stock portfolios, enhancing performance, and managing risk in an increasingly complex financial environment.





Description

----------

This white paper presents an in-depth analysis of stock portfolio management through the application of advanced data analytics techniques. The project aims to address the challenges faced by investors in optimizing their portfolios by incorporating a data-driven approach to decision-making. By analyzing historical stock prices, financial indicators, and macroeconomic variables, the project seeks to develop strategies that maximize returns while minimizing risk.



**Scope of the Project**  



The scope of this project includes the following key areas:



**1. Data Collection and Preprocessing:**



  - The project begins with the collection of a comprehensive dataset that includes historical stock prices, financial ratios, and relevant economic indicators.

  - Data preprocessing steps are undertaken to clean and prepare the data, ensuring accuracy, consistency, and relevance. This includes handling missing data, normalizing variables, and filtering out noise.  



**2. Exploratory Data Analysis (EDA):**



 - EDA is conducted to uncover underlying trends, correlations, and patterns within the data. This step provides insights into the behavior of individual stocks and the market as a whole, laying the foundation for further analysis.  

 - Visualization techniques are employed to illustrate key findings and to identify potential opportunities for portfolio optimization.  



**3. Dimensionality Reduction and Portfolio Construction using Correlation Analysis, Clustering and Principal Component Analysis (PCA)**



 - **Correlation Analysis, Clustering and Portfolio Construction**

    Hierarchical clustering techniques are applied to group stocks into clusters based on their similarities in performance, risk profile, and other attributes.

    This clustering facilitates the selection of a diversified set of assets for portfolio construction, ensuring that the portfolio is balanced and less susceptible to market shocks.



 - **Principal Component Analysis (PCA)**

    To manage the complexity of the dataset and to focus on the most impactful variables, PCA is utilized to reduce the number of factors considered in the analysis.It  helps in identifying the principal components that explain the majority of the variance in the data, enabling the selection of the most relevant indicators for portfolio construction.



 - **Statcking PCA,Correlation Analysis and Clustering for Diversified Portfolio Construction**

    stacking Correlation Analysis, Clustering and Principal Component Analysis (PCA) helps to construct a well diversified portfolio



**4. Asset Pricing, Profit & Lost simation and Risk calculation***  



 - Lognormal of asset returns, Covarariance Matrix Cholesky Decomposition applied to Monte Carlo Simulation for asset pricing and Profit & Lost simulation.

 - Value at Risk(VaR) and Conditional Value at Risk(CVaR) calculation



**5. Portfolio Optimization:**



 - Modern Portfolio Theory (MPT) is implemented to construct efficient portfolios that optimize the trade-off between risk and return.

 - The optimization process involves determining the boundary random portfolios assets and weights that maximize the portfolio's expected return for a given level of risk or minimize risk for a given level of expected return or a given risk level.

 - Using Monte Carlo simulation to generate Efficient Frontier

 - Machine Learning technics are used to improve the optimization process by modelling the boundary random portfolios assets that maximize the portfolio's expected return for a given level of risk or minimize risk for a given level of expected return or a given risk level.

 - Investment strategies are bult for optimal portfolios(minimal risk portfolio, maximal return portfolio, sharpe ratio (tangent portfolio)



**6.  Investment Risk Profiles Simulation using K-Means Clustering applied to random portfolio**



 - The simulated portfolio risk is combigned with the simulated the portfolio expected return and the predicted expected return to set the randomn efficient frontier data. The randomn efficient frontier data is then used as input for the  K-means cluster models to simulate the instment risk profile and investment strategy.



**7. Stress Testing and Scenario Analysis:**



 - Stress testing is conducted to evaluate the portfolio's performance under different economic scenarios, including adverse market conditions.

 - This analysis provides insights into the portfolio’s resilience and helps in identifying potential vulnerabilities.  





**Tools and Technologies**  



The project leverages various tools and technologies, including:



 - Python: For data analysis, statistical modeling, and machine learning.

 - Pandas and NumPy: For data manipulation and numerical computations.

 - Matplotlib and Seaborn: For data visualization.

 - Scikit-learn: For machine learning, PCA, and clustering.

 - Optimization Libraries: For portfolio optimization using MPT.

 - Financial Databases: To source historical data, including stock prices and economic indicators.  



**Key Outcomes**

The project yields several key outcomes:



 - Identification of the most influential economic indicators and stock characteristics for portfolio management.

 - Creation of optimized portfolios that demonstrate improved risk-adjusted returns.

 - Insights into portfolio performance under various market conditions, aiding in risk management and strategic planning.




```python
#pip install stats-can
#conda pip install -c districtdatalabs yellowbrick on Anaconda Prompt
#conda install conda=24.5.0
#conda install conda-forge::stats_can
```

### Import Libraries


```python
import yfinance as yf
import pandas as pd
from datetime import date, timedelta
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import norm, lognorm, exponnorm, logistic, erlang,gennorm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
#from yellowbrick.cluster import KElbowVisualizer
from scipy.optimize import curve_fit
import random
from statistics import NormalDist
from scipy import stats
from fitter import Fitter, get_common_distributions, get_distributions
import matplotlib.transforms as transforms
from matplotlib.table import table
from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score
from tabulate import tabulate

#from stats_can import StatsCan as sc
from stats_can import StatsCan
sc = StatsCan()

#import pandas_datareader.data as web
```

### 1. Index Contents Data Collection and Preprocessing



In this section, we will read all the S&P/TSX composite constituents table from wikipedia(https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index).

Then we will get the tickers adjusted close prices from Yahoo Finance using yfinance library. We will clean the data by removing all the empty

rows and columns. With more than 200 remaining tickers, we will calculate the assets log return and we will remove all the assets with negative expected return. We will couple Correlation Analysis with Principle Component Analysis to reduce the volume of assets and keep only most important assets. The Correlation Analysis will be used to identify and remove redundant assets. The end result will be a well diversify portfolio.



```python

#--------------------------------------------- 1. Index Contents Data Collection and Preprocessing ------------------------------------------------
#read the index content from wikipedia and return the index content data frame
def read_index_content(content_html,web_tab_number):
    S_and_P_TSX_Composite = pd.read_html(content_html)[web_tab_number]
    index_content_df = S_and_P_TSX_Composite[['Ticker','Company','Sector [10]','Industry [10]']]
    index_content_df = index_content_df.rename(columns={"Sector [10]": "Sector", "Industry [10]": "Industry"})
    return index_content_df
    #return S_and_P_TSX_Composite[['Ticker','Company','Sector [10]','Industry [10]']].head()


#extract the index tickers
def generate_ticker_df(index_content_df):
    index_content_tickers_list = index_content_df['Ticker']
    index_content_tickers_list = index_content_tickers_list.tolist()
    new_index_content_tickers_list = []
    for item in index_content_tickers_list:
        new_index_content_tickers_list.append(str(item))
    return new_index_content_tickers_list


#--------------------------------------------------------------------------------------------------------------------
#Description:Extract adj close price for each stock on the index from Yahoo Finance web site and clean the data
#Input:start date, end date, index ticker list
#Return the index Adj close price data frame
#-----------------------------------------------------------------------------------------------------------------------
def start_date(reporting_year_period = 365*5):
    return pd.Timestamp.today() - pd.Timedelta(days = reporting_year_period)

def create_adj_close_price_df(reporting_year_period, content_ticker_list):

    #frequency = frequency_date_column[0].upper()
    #selected_asset_list = get_selected_assets_list(log_returns,correlation_coefficient_treshold)
    start_date = reporting_year_period
    end_date = date.today()
    selected_assets_yahoo_adj_close_price_data = yf.download(content_ticker_list, start_date, end_date, ['Adj Close'], period ='max')
    selected_assets_adj_close_price_df = selected_assets_yahoo_adj_close_price_data['Adj Close']
    index_adj_close_price_df = selected_assets_adj_close_price_df.dropna(axis=1)
    return index_adj_close_price_df

def asset_daily_price(price_df,number_of_asset):
    print('\nPlotting the first 5 assets daily adj closed prices\n')
    price_df.iloc[:,:number_of_asset].plot(figsize=(15,6))
    plt.show()

def plot_assets_distribution(df,xlabel, ylabel, title=''):
    # Define the number of assets
    n_assets = df.shape[1]

    # Create subplots
    fig, axes = plt.subplots(1, n_assets, figsize=(23,  3))

    if n_assets == 1:
        axes = [axes]

    # Iterate over each asset
    for i, asset in enumerate(df.columns):
        g =sns.histplot(df[asset], kde=True, ax=axes[i])
        axes[i].set_title(f'{title + asset}')
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)

        # Calculate and display statistics
        mean_return = df[asset].mean()
        std_dev = df[asset].std()
        skewness = df[asset].skew()
        kurtosis = df[asset].kurtosis()

         # Add statistics below the plot
        statistics = (f"Mean: {mean_return:.4f}\n"
                 f"Std Dev: {std_dev:.4f}\n"
                  f"Skewness: {skewness:.4f}\n"
                 f"Kurtosis: {kurtosis:.4f}")

        # Place the text under the plot
        axes[i].text(0.3, -0.3, statistics, transform=axes[i].transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"))

# Adjust layout
plt.tight_layout()
plt.show()


def normalize_asset_daily_price(price_df,number_of_asset):

    normalized_asset_daily_price_df = price_df.iloc[:,:number_of_asset]
    normalized_asset_daily_price_df = (normalized_asset_daily_price_df / normalized_asset_daily_price_df.iloc[0])*100
    normalized_asset_cols_size = len(normalized_asset_daily_price_df.columns)
    normalized_asset_daily_price_df.plot(figsize = (15, 6))
    plt.show()
    plot_assets_distribution(normalized_asset_daily_price_df, 'Adjusted Close Price','Frequency')

```

```
<Figure size 640x480 with 0 Axes>
```
```python
print('\nData collection and preprocessing\n')
index_content_df = read_index_content('https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index',3)
content_ticker_list = generate_ticker_df(index_content_df)
index_adj_close_price_df = create_adj_close_price_df( start_date(365*5), content_ticker_list )
print('\nList of companies\n')
display(index_content_df)
print('\nAdjusted Close Price Data Frame\n')
display(index_adj_close_price_df)
print('\nData structure\n')
index_adj_close_price_df.info()
print('\nData statics summary\n')
display(index_adj_close_price_df.describe().transpose())
```

```

Data collection and preprocessing

[*********************100%%**********************]  225 of 225 completed
```
```

105 Failed downloads:
['CU', 'CCL.B', 'IVN', 'IFP', 'TIH', 'DFY', 'INE', 'GEI', 'POU', 'DSG', 'KEL', 'ABX', 'TOU', 'RUS', 'AOI', 'GWO', 'WSP', 'MRU', 'WTE', 'FRU', 'KXS', 'REI.UN', 'RCH', 'EMA', 'ARX', 'ATD', 'WPK', 'WN', 'CJT', 'FIL', 'NWC', 'CPX', 'MTY', 'LUG', 'AAV', 'IFC', 'LNR', 'BBD.B', 'SRU.UN', 'EQB', 'IMG', 'CFP', 'BIR', 'EFN', 'FFH', 'TOY', 'SIA', 'LUN', 'OLA', 'NPI', 'EIF', 'DML', 'FVI', 'KNT', 'WDO', 'WCP', 'ALA', 'MATR']: Exception('%ticker%: No price data found, symbol may be delisted (1d 2019-09-04 04:00:29.864463 -> 2024-09-02)')
['HWX']: Exception("%ticker%: Period 'max' is invalid, must be one of ['1d', '5d']")
['CHP.UN', 'CNR', 'TCL.A', 'ATH', 'ONEX', 'ACO.X', 'PKI', 'BDGI', 'AP.UN', 'ATRL', 'FTT', 'BEI.UN', 'TSU', 'IPCO', 'RCI.B', 'CRR.UN', 'CSH.UN', 'CCA', 'MTL', 'CSU', 'POW', 'CAR.UN', 'NWH.UN', 'CS', 'CRT.UN', 'GRT.UN', 'CTC.A', 'BBU.UN', 'PMZ.UN', 'BEP.UN', 'TA', 'IIP.UN', 'DPM', 'QBR.B', 'TECK.B', 'KMP.UN', 'EMP.A', 'FCR.UN', 'DIR.UN', 'BIP.UN', 'HR.UN', 'GIB.A']: Exception('%ticker%: No timezone found, symbol may be delisted')
['CPG', 'ENGH', 'TCN', 'ERF']: Exception('%ticker%: No data found, symbol may be delisted')

```
```


List of companies


```
```
    Ticker                                     Company             Sector  \
0      AAV                       Advantage Energy Ltd.             Energy   
1      AOI                            Africa Oil Corp.             Energy   
2      AEM                  Agnico Eagle Mines Limited    Basic Materials   
3       AC                                  Air Canada        Industrials   
4      AGI                            Alamos Gold Inc.    Basic Materials   
..     ...                                         ...                ...   
220    WTE  Westshore Terminals Investment Corporation        Industrials   
221    WPM               Wheaton Precious Metals Corp.    Basic Materials   
222    WCP                     Whitecap Resources Inc.             Energy   
223    WPK                                 Winpak Ltd.  Consumer Cyclical   
224    WSP                             WSP Global Inc.        Industrials   

                                 Industry  
0    Oil & Gas Exploration and Production  
1    Oil & Gas Exploration and Production  
2                         Metals & Mining  
3                          Transportation  
4                         Metals & Mining  
..                                    ...  
220                        Transportation  
221                       Metals & Mining  
222  Oil & Gas Exploration and Production  
223                Packaging & Containers  
224                          Construction  

[225 rows x 4 columns]
```
```

Adjusted Close Price Data Frame


```
```
                   AC        AEM        AGI        AQN        ATS    BB  \
Date                                                                      
2019-09-04  33.602268  56.552120   6.899674  10.177608  13.890000  6.91   
2019-09-05  33.263344  54.074928   6.595137  10.124326  13.890000  7.28   
2019-09-06  33.505432  52.435226   6.328667  10.124326  13.890000  7.19   
2019-09-09  33.795937  50.874863   6.166883  10.093872  14.020000  6.98   
2019-09-10  34.357590  49.975670   6.157364  10.032975  14.020000  7.15   
...               ...        ...        ...        ...        ...   ...   
2024-08-26  33.250000  81.918884  19.570000   5.410000  27.180000  2.36   
2024-08-27  32.150002  81.869118  19.490000   5.350000  27.059999  2.32   
2024-08-28  32.720001  80.814285  19.110001   5.290000  26.740000  2.32   
2024-08-29  33.320000  81.689995  19.170000   5.360000  26.730000  2.35   
2024-08-30  33.330002  81.470001  19.280001   5.410000  26.850000  2.35   

                  BCE        BHC  BLDP        BLX  ...       TLRY        TPZ  \
Date                                               ...                         
2019-09-04  35.615284  21.330000  4.63  12.608221  ...  30.000000  11.919843   
2019-09-05  35.652538  21.719999  4.53  13.009621  ...  32.080002  11.867153   
2019-09-06  35.898422  22.180000  4.61  13.109968  ...  32.060001  11.847400   
2019-09-09  36.039982  22.090000  5.10  13.425352  ...  30.150000  11.959357   
2019-09-10  35.958023  22.490000  5.00  13.547209  ...  31.129999  11.919843   
...               ...        ...   ...        ...  ...        ...        ...   
2024-08-26  35.139999   5.970000  1.90  30.570000  ...   1.860000  18.000000   
2024-08-27  35.220001   5.960000  1.87  30.959999  ...   1.750000  18.070000   
2024-08-28  35.009998   5.900000  1.81  31.010000  ...   1.700000  17.850000   
2024-08-29  34.889999   5.870000  1.87  31.230000  ...   1.700000  18.250000   
2024-08-30  35.000000   5.930000  1.84  31.350000  ...   1.710000  18.350000   

                   TRI        TRP        TVE        VET         WCN  \
Date                                                                  
2019-09-04   61.598331  38.404881  22.843452  12.524200   88.492218   
2019-09-05   62.446735  37.961239  22.790571  12.937088   88.424744   
2019-09-06   62.552784  37.754208  22.737694  13.074717   88.578979   
2019-09-09   61.209476  37.842941  22.587868  13.539213   86.583580   
2019-09-10   60.051735  37.717232  22.587868  13.737055   86.072670   
...                ...        ...        ...        ...         ...   
2024-08-26  167.179993  45.439999  22.450001  10.350000  186.460007   
2024-08-27  170.919998  45.680000  22.490000  10.170000  186.009995   
2024-08-28  170.320007  45.450001  22.510000  10.140000  185.419998   
2024-08-29  169.570007  45.750000  22.500000  10.310000  185.770004   
2024-08-30  171.179993  46.340000  22.469999  10.280000  186.500000   

                  WFG        WPM          X  
Date                                         
2019-09-04  32.459225  28.881174  10.907244  
2019-09-05  32.262451  28.131020  11.236592  
2019-09-06  32.262451  27.005779  11.033171  
2019-09-09  33.265079  26.358767  11.846856  
2019-09-10  35.082947  26.143095  12.030904  
...               ...        ...        ...  
2024-08-26  90.309998  62.299999  37.820000  
2024-08-27  88.959999  62.430000  37.980000  
2024-08-28  88.370003  61.389999  37.389999  
2024-08-29  88.930000  61.660000  38.560001  
2024-08-30  88.519997  61.810001  37.910000  

[1257 rows x 98 columns]
```
```

Data structure

<class 'pandas.core.frame.DataFrame'>
Index: 1257 entries, 2019-09-04 00:00:00 to 2024-08-30 00:00:00
Data columns (total 98 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   AC      1257 non-null   float64
 1   AEM     1257 non-null   float64
 2   AGI     1257 non-null   float64
 3   AQN     1257 non-null   float64
 4   ATS     1257 non-null   float64
 5   BB      1257 non-null   float64
 6   BCE     1257 non-null   float64
 7   BHC     1257 non-null   float64
 8   BLDP    1257 non-null   float64
 9   BLX     1257 non-null   float64
 10  BMO     1257 non-null   float64
 11  BN      1257 non-null   float64
 12  BNS     1257 non-null   float64
 13  BTE     1257 non-null   float64
 14  BTO     1257 non-null   float64
 15  BYD     1257 non-null   float64
 16  CAE     1257 non-null   float64
 17  CCO     1257 non-null   float64
 18  CG      1257 non-null   float64
 19  CIGI    1257 non-null   float64
 20  CIX     1257 non-null   float64
 21  CLS     1257 non-null   float64
 22  CM      1257 non-null   float64
 23  CNQ     1257 non-null   float64
 24  CP      1257 non-null   float64
 25  CVE     1257 non-null   float64
 26  CWB     1257 non-null   float64
 27  DOL     1257 non-null   float64
 28  DOO     1257 non-null   float64
 29  EFR     1257 non-null   float64
 30  ELD     1257 non-null   float64
 31  ENB     1257 non-null   float64
 32  EQX     1257 non-null   float64
 33  ERO     1257 non-null   float64
 34  FM      1257 non-null   float64
 35  FNV     1257 non-null   float64
 36  FR      1257 non-null   float64
 37  FSV     1257 non-null   float64
 38  FTS     1257 non-null   float64
 39  GIL     1257 non-null   float64
 40  GOOS    1257 non-null   float64
 41  GSY     1257 non-null   float64
 42  H       1257 non-null   float64
 43  HBM     1257 non-null   float64
 44  IAG     1257 non-null   float64
 45  IGM     1257 non-null   float64
 46  IMO     1257 non-null   float64
 47  K       1257 non-null   float64
 48  KEY     1257 non-null   float64
 49  L       1257 non-null   float64
 50  LAAC    1257 non-null   float64
 51  MAG     1257 non-null   float64
 52  MFC     1257 non-null   float64
 53  MG      1257 non-null   float64
 54  MX      1257 non-null   float64
 55  NAN     1257 non-null   float64
 56  NG      1257 non-null   float64
 57  NGD     1257 non-null   float64
 58  NTR     1257 non-null   float64
 59  NXE     1257 non-null   float64
 60  OGC     1257 non-null   float64
 61  OR      1257 non-null   float64
 62  OSK     1257 non-null   float64
 63  OTEX    1257 non-null   float64
 64  PAAS    1257 non-null   float64
 65  PBH     1257 non-null   float64
 66  PD      1257 non-null   float64
 67  PEY     1257 non-null   float64
 68  PPL     1257 non-null   float64
 69  PRMW    1257 non-null   float64
 70  PSI     1257 non-null   float64
 71  PSK     1257 non-null   float64
 72  QSR     1257 non-null   float64
 73  RY      1257 non-null   float64
 74  SAP     1257 non-null   float64
 75  SHOP    1257 non-null   float64
 76  SII     1257 non-null   float64
 77  SIL     1257 non-null   float64
 78  SJ      1257 non-null   float64
 79  SLF     1257 non-null   float64
 80  SPB     1257 non-null   float64
 81  SSL     1257 non-null   float64
 82  SSRM    1257 non-null   float64
 83  STN     1257 non-null   float64
 84  SU      1257 non-null   float64
 85  T       1257 non-null   float64
 86  TD      1257 non-null   float64
 87  TFII    1257 non-null   float64
 88  TLRY    1257 non-null   float64
 89  TPZ     1257 non-null   float64
 90  TRI     1257 non-null   float64
 91  TRP     1257 non-null   float64
 92  TVE     1257 non-null   float64
 93  VET     1257 non-null   float64
 94  WCN     1257 non-null   float64
 95  WFG     1257 non-null   float64
 96  WPM     1257 non-null   float64
 97  X       1257 non-null   float64
dtypes: float64(98)
memory usage: 972.2+ KB

Data statics summary


```
```
      count        mean        std        min        25%         50%  \
AC   1257.0   36.512448   3.092432  25.417492  34.379246   36.194168   
AEM  1257.0   53.623814   9.305847  32.276482  47.323544   52.089771   
AGI  1257.0    9.447117   3.169251   3.714411   7.307484    8.300611   
AQN  1257.0    9.912172   2.794989   4.757608   6.779181   10.848129   
ATS  1257.0   28.857884  10.614868  10.000000  16.709999   31.440001   
..      ...         ...        ...        ...        ...         ...   
VET  1257.0   11.536508   5.510147   1.587390   7.014824   11.788893   
WCN  1257.0  123.783909  25.419136  69.163414  99.404480  127.301590   
WFG  1257.0   67.408003  18.936303  14.714417  55.634064   73.254677   
WPM  1257.0   40.842845   7.997292  22.332952  37.146351   41.381927   
X    1257.0   23.187595  10.819288   4.768804  14.710779   23.290730   

            75%         max  
AC    38.320412   61.728195  
AEM   58.276974   82.386589  
AGI   11.715064   19.910000  
AQN   12.408038   14.397228  
ATS   38.150002   48.730000  
..          ...         ...  
VET   14.338865   28.071140  
WCN  137.661545  186.500000  
WFG   80.781425   98.821220  
WPM   45.544857   62.430000  
X     29.508034   49.411922  

[98 rows x 8 columns]
```
### 2. Exploratory Data Analysis (EDA)


```python
def plot_assets_distribution(df,xlabel, ylabel, title=''):
    # Define the number of assets
    n_assets = df.shape[1]

    # Create subplots
    fig, axes = plt.subplots(1, n_assets, figsize=(23,  3))

    if n_assets == 1:
        axes = [axes]

    # Iterate over each asset
    for i, asset in enumerate(df.columns):
        g =sns.histplot(df[asset], kde=True, ax=axes[i])
        axes[i].set_title(f'{title + asset}')
        axes[i].set_xlabel(xlabel)
        axes[i].set_ylabel(ylabel)

        # Calculate and display statistics
        mean_return = df[asset].mean()
        std_dev = df[asset].std()
        skewness = df[asset].skew()
        kurtosis = df[asset].kurtosis()

         # Add statistics below the plot
        statistics = (f"Mean: {mean_return:.4f}\n"
                 f"Std Dev: {std_dev:.4f}\n"
                  f"Skewness: {skewness:.4f}\n"
                 f"Kurtosis: {kurtosis:.4f}")

        # Place the text under the plot
        axes[i].text(0.3, -0.3, statistics, transform=axes[i].transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightgrey"))

# Adjust layout
plt.tight_layout()
plt.show()


def normalize_asset_daily_price(price_df,number_of_asset):

    normalized_asset_daily_price_df = price_df.iloc[:,:number_of_asset]
    normalized_asset_daily_price_df = (normalized_asset_daily_price_df / normalized_asset_daily_price_df.iloc[0])*100
    normalized_asset_cols_size = len(normalized_asset_daily_price_df.columns)
    normalized_asset_daily_price_df.plot(figsize = (15, 6))
    plt.show()
    plot_assets_distribution(normalized_asset_daily_price_df, 'Adjusted Close Price','Frequency')

```

```
<Figure size 640x480 with 0 Axes>
```
```python
print('\nExploratory Data Analysis (EDA)\n')
number_of_asset =5
index_adj_close_price_df.iloc[0] # first row
asset_daily_price(index_adj_close_price_df,number_of_asset) #Plotting the first 5 assets daily adj closed prices
normalize_asset_daily_price(index_adj_close_price_df,number_of_asset)  #Normalization of adj closed prices to 100

```

```

Exploratory Data Analysis (EDA)


Plotting the first 5 assets daily adj closed prices


```
```
<Figure size 1500x600 with 1 Axes>
```
```
<Figure size 1500x600 with 1 Axes>
```
```
<Figure size 2300x300 with 5 Axes>
```
      

      




#### Assets log return Volatility Calculation

In this section, we will calculate the assets log return instead of arithmetic return. The arithmetic return is the percentage change in the asset's price from one period to the next where as the log return of an asset over a period is calculated as the natural logarithm of the ratio of the ending price to the starting price.



$$ Arithmetic Return:

R = \frac{P_t - P_{t-1}}{P_{t-1}}

$$.

$$ Log Return:

\text{Log Return} = \ln\left(\frac{P_t}{P_{t-1}}\right)

$$.



Throughout this project, we will use asset log returns instead of arithmetic returns, simply because, in the upcoming sections we will perform stochastic simulation of the stock prices to calculate Profit & Lost, VaR, CVaR and stress testing. Log returns are commonly used in the financial literature to perform financial modeling like asset prices modeling over time,  as prices cannot be negative but can increase indefinitely. Log returns are normally distributed with Fat-Tailed that make them more likely to predict extreme returns than assuming arithmetic returns to be normally distributed. As we know, stocks are traded with very high frequency over very short period of time and the form of their distributions are unknown as we can see in the plottings above. This leads to use log returns witch naturally account for continuous compounding and more accurate instead of arithmetic returns witch are based on simple interest. Furthermore, as opposed to arithmetic returns, log returns are additive meaning that you can add log returns over multiple periods to get the total log return.



```python
def calculate_stock_price_log_return(index_adj_close_price_df):
    log_returns = np.log(index_adj_close_price_df / index_adj_close_price_df.shift(1))
    log_returns = log_returns.dropna(how = 'all')
    return log_returns

#removing asset with negative expected return
def removing_assets_with_negative_expected_return(log_returns,threshold):
    # Calculate the correlation matrix
    #corr_matrix = expected_returns.corr()
    # Create a list to store uncorrelated assets
    assets_with_positive_expected_return = []
    # Iterate through the correlation matrix
    for asset in log_returns.columns:
        # Check if the asset is uncorrelated with all other assets
        #for other_assets in corr_matrix.columns:
            if log_returns.mean()[asset] > threshold:
                assets_with_positive_expected_return.append(asset)
    assets_with_positive_expected_return_list = list(dict.fromkeys(assets_with_positive_expected_return))
    return assets_with_positive_expected_return_list


def positive_assets_log_returns_df(log_returns_df, positive_assets_list):
    return log_returns_df[positive_assets_list]

def stocks_initial_price(positive_assets_list):
    return index_adj_close_price_df.iloc[0][positive_assets_list]

def generate_asset_volatility(frequency_date_column, log_return_df):
    #selected_content_ticker_list = get_selected_assets_list(log_returns,correlation_coefficient_treshold)

    frequency = frequency_date_column[0].upper()


    assets_volatility_df = log_return_df.rolling(center=False,window= 252).std() * np.sqrt(252)
    for col in list(assets_volatility_df.columns):
        assets_volatility_df = assets_volatility_df.rename(columns={col: col+' Volatility'})

    assets_volatility_df = assets_volatility_df.dropna(axis=0)

    assets_volatility_df[frequency_date_column] = pd.to_datetime(assets_volatility_df.index, format = '%m/%Y')
    assets_volatility_df[frequency_date_column] = assets_volatility_df[frequency_date_column].dt.to_period(frequency)

    assets_volatility_df.set_index(frequency_date_column, inplace=True)
    assets_volatilities = assets_volatility_df.groupby(frequency_date_column).mean()
    assets_volatilities = round(assets_volatilities,1)
    assets_volatilities = assets_volatilities.dropna(axis=0)
    return assets_volatilities


def portfolio_arihtmetics(log_returns,stocks_initial_prices):
    return pd.DataFrame({'mu expected_return':log_returns.mean(),
                         'variance':log_returns.var(),
                         'Sigmas(volatilities)':log_returns.std(),
                         'modifiy shape(Er)/𝝈':log_returns.mean()/log_returns.std(),
                         'initial price': stocks_initial_prices}).transpose()


stock_price_log_return = calculate_stock_price_log_return(index_adj_close_price_df)
log_returns = positive_assets_log_returns_df(stock_price_log_return,
                                             removing_assets_with_negative_expected_return(stock_price_log_return,0))
asset_volatility_df = generate_asset_volatility('Quater', log_returns)
positive_assets_list = removing_assets_with_negative_expected_return(stock_price_log_return,0)
stocks_initial_prices = stocks_initial_price(positive_assets_list)
portfolio_arihtmetics_df = portfolio_arihtmetics(log_returns,stocks_initial_prices).transpose()

```

```python
print('\nAssets log return data frame\n')
display(log_returns)
print('\nAssets volatility data frame\n')
display(asset_volatility_df)
print('\nPortfolio arithmetics\n')
display(portfolio_arihtmetics_df)
plot_assets_distribution(log_returns.iloc[:,:number_of_asset], 'log_returns','Frequency')
plot_assets_distribution(asset_volatility_df.iloc[:,:number_of_asset], 'Volatility','Frequency')
```

```

Assets log return data frame


```
```
                 AEM       AGI       ATS       BLX       BMO        BN  \
Date                                                                     
2019-09-05 -0.044792 -0.045142  0.000000  0.031340  0.016443  0.012316   
2019-09-06 -0.030792 -0.041243  0.000000  0.007684  0.008908  0.007131   
2019-09-09 -0.030210 -0.025896  0.009316  0.023772  0.013638 -0.008073   
2019-09-10 -0.017833 -0.001545  0.000000  0.009036  0.018039 -0.011756   
2019-09-11  0.009306  0.000000  0.000000  0.039424  0.002630  0.007411   
...              ...       ...       ...       ...       ...       ...   
2024-08-26 -0.005693 -0.002552  0.001841  0.009201  0.005535  0.005483   
2024-08-27 -0.000608 -0.004096 -0.004425  0.012677 -0.063600  0.008469   
2024-08-28 -0.012968 -0.019690 -0.011896  0.001614 -0.017194 -0.006649   
2024-08-29  0.010778  0.003135 -0.000374  0.007069  0.013104  0.004437   
2024-08-30 -0.002697  0.005722  0.004479  0.003835  0.007924  0.011804   

                 BNS       BTE       BTO       BYD  ...         T        TD  \
Date                                                ...                       
2019-09-05  0.007420  0.030772  0.028079  0.023600  ...  0.004748  0.009185   
2019-09-06  0.010479  0.029853 -0.002579 -0.004174  ...  0.009981  0.010911   
2019-09-09  0.008377  0.043172  0.027067  0.042978  ...  0.014787  0.005411   
2019-09-10  0.005426  0.000000  0.018675  0.024145  ...  0.021246  0.011802   
2019-09-11  0.007547 -0.007067  0.011954  0.035725  ...  0.030401  0.001599   
...              ...       ...       ...       ...  ...       ...       ...   
2024-08-26  0.003505  0.019152  0.009959 -0.001834  ...  0.001519 -0.002868   
2024-08-27  0.027805 -0.019152 -0.011477  0.000000  ... -0.005582  0.008581   
2024-08-28 -0.021241 -0.011111  0.009674 -0.008379  ...  0.008614 -0.004365   
2024-08-29  0.006318  0.024829  0.002104  0.008546  ... -0.003032  0.000336   
2024-08-30  0.013320 -0.030431  0.004194  0.001501  ...  0.007060  0.007875   

                TFII       TPZ       TRI       TRP       WCN       WFG  \
Date                                                                     
2019-09-05  0.000000 -0.004430  0.013679 -0.011619 -0.000763 -0.006081   
2019-09-06  0.041314 -0.001666  0.001697 -0.005469  0.001743  0.000000   
2019-09-09  0.000000  0.009406 -0.021709  0.002348 -0.022784  0.030604   
2019-09-10  0.000000 -0.003310 -0.019096 -0.003327 -0.005918  0.053207   
2019-09-11  0.022345 -0.003875 -0.017369 -0.018200 -0.003029  0.005594   
...              ...       ...       ...       ...       ...       ...   
2024-08-26  0.009018  0.005571  0.005638  0.008176  0.000161  0.006889   
2024-08-27 -0.015835  0.003881  0.022125  0.005268 -0.002416 -0.015061   
2024-08-28 -0.003695 -0.012250 -0.003517 -0.005048 -0.003177 -0.006654   
2024-08-29  0.001883  0.022162 -0.004413  0.006579  0.001886  0.006317   
2024-08-30 -0.004782  0.005465  0.009450  0.012814  0.003922 -0.004621   

                 WPM         X  
Date                            
2019-09-05 -0.026317  0.029748  
2019-09-06 -0.040822 -0.018269  
2019-09-09 -0.024250  0.071156  
2019-09-10 -0.008216  0.015416  
2019-09-11  0.006792  0.063179  
...              ...       ...  
2024-08-26  0.003054  0.018144  
2024-08-27  0.002085  0.004222  
2024-08-28 -0.016799 -0.015656  
2024-08-29  0.004388  0.030812  
2024-08-30  0.002430 -0.017001  

[1256 rows x 76 columns]
```
```

Assets volatility data frame


```
```
        AEM Volatility  AGI Volatility  ATS Volatility  BLX Volatility  \
Quater                                                                   
2020Q3             0.5             0.7             0.4             0.7   
2020Q4             0.5             0.7             0.4             0.7   
2021Q1             0.5             0.7             0.4             0.6   
2021Q2             0.4             0.5             0.4             0.4   
2021Q3             0.4             0.5             0.4             0.3   
2021Q4             0.3             0.4             0.4             0.2   
2022Q1             0.3             0.4             0.3             0.2   
2022Q2             0.4             0.4             0.4             0.2   
2022Q3             0.4             0.4             0.4             0.2   
2022Q4             0.4             0.5             0.5             0.3   
2023Q1             0.4             0.4             0.5             0.3   
2023Q2             0.4             0.4             0.4             0.3   
2023Q3             0.4             0.4             0.4             0.3   
2023Q4             0.3             0.3             0.3             0.3   
2024Q1             0.3             0.3             0.3             0.3   
2024Q2             0.3             0.3             0.3             0.3   
2024Q3             0.3             0.3             0.3             0.3   

        BMO Volatility  BN Volatility  BNS Volatility  BTE Volatility  \
Quater                                                                  
2020Q3             0.5            0.5             0.4             1.1   
2020Q4             0.5            0.5             0.5             1.1   
2021Q1             0.5            0.5             0.4             1.1   
2021Q2             0.3            0.3             0.2             0.8   
2021Q3             0.2            0.3             0.2             0.7   
2021Q4             0.2            0.3             0.2             0.7   
2022Q1             0.2            0.3             0.2             0.6   
2022Q2             0.2            0.3             0.2             0.6   
2022Q3             0.2            0.3             0.2             0.7   
2022Q4             0.3            0.3             0.2             0.7   
2023Q1             0.3            0.4             0.2             0.7   
2023Q2             0.3            0.3             0.2             0.6   
2023Q3             0.2            0.3             0.2             0.5   
2023Q4             0.2            0.3             0.2             0.5   
2024Q1             0.2            0.3             0.2             0.4   
2024Q2             0.2            0.3             0.2             0.4   
2024Q3             0.2            0.3             0.2             0.4   

        BTO Volatility  BYD Volatility  ...  T Volatility  TD Volatility  \
Quater                                  ...                                
2020Q3             0.8             0.9  ...           0.3            0.4   
2020Q4             0.8             0.9  ...           0.4            0.4   
2021Q1             0.8             0.9  ...           0.3            0.4   
2021Q2             0.4             0.5  ...           0.2            0.2   
2021Q3             0.3             0.4  ...           0.2            0.2   
2021Q4             0.3             0.4  ...           0.2            0.2   
2022Q1             0.3             0.4  ...           0.2            0.2   
2022Q2             0.4             0.4  ...           0.2            0.2   
2022Q3             0.4             0.4  ...           0.3            0.2   
2022Q4             0.4             0.4  ...           0.3            0.2   
2023Q1             0.4             0.4  ...           0.3            0.2   
2023Q2             0.4             0.3  ...           0.3            0.2   
2023Q3             0.4             0.3  ...           0.3            0.2   
2023Q4             0.4             0.3  ...           0.3            0.2   
2024Q1             0.4             0.3  ...           0.3            0.2   
2024Q2             0.3             0.3  ...           0.2            0.2   
2024Q3             0.3             0.3  ...           0.2            0.2   

        TFII Volatility  TPZ Volatility  TRI Volatility  TRP Volatility  \
Quater                                                                    
2020Q3              0.5             0.8             0.3             0.5   
2020Q4              0.5             0.8             0.3             0.5   
2021Q1              0.5             0.7             0.3             0.5   
2021Q2              0.4             0.3             0.2             0.3   
2021Q3              0.4             0.2             0.2             0.2   
2021Q4              0.4             0.2             0.2             0.2   
2022Q1              0.4             0.2             0.2             0.2   
2022Q2              0.4             0.2             0.2             0.2   
2022Q3              0.4             0.2             0.2             0.2   
2022Q4              0.4             0.2             0.2             0.3   
2023Q1              0.4             0.2             0.2             0.3   
2023Q2              0.4             0.2             0.2             0.3   
2023Q3              0.4             0.2             0.2             0.3   
2023Q4              0.3             0.1             0.2             0.3   
2024Q1              0.3             0.1             0.2             0.2   
2024Q2              0.3             0.1             0.2             0.2   
2024Q3              0.3             0.1             0.2             0.2   

        WCN Volatility  WFG Volatility  WPM Volatility  X Volatility  
Quater                                                                
2020Q3             0.3             0.7             0.5           0.8  
2020Q4             0.3             0.7             0.5           0.8  
2021Q1             0.3             0.7             0.5           0.8  
2021Q2             0.2             0.5             0.4           0.8  
2021Q3             0.1             0.4             0.4           0.7  
2021Q4             0.1             0.4             0.4           0.7  
2022Q1             0.2             0.4             0.3           0.6  
2022Q2             0.2             0.4             0.3           0.5  
2022Q3             0.2             0.4             0.3           0.5  
2022Q4             0.2             0.5             0.4           0.6  
2023Q1             0.2             0.5             0.4           0.5  
2023Q2             0.2             0.4             0.4           0.5  
2023Q3             0.2             0.3             0.3           0.5  
2023Q4             0.2             0.3             0.3           0.5  
2024Q1             0.2             0.3             0.3           0.5  
2024Q2             0.2             0.3             0.3           0.5  
2024Q3             0.2             0.3             0.3           0.4  

[17 rows x 76 columns]
```
```

Portfolio arithmetics


```
```
     mu expected_return  variance  Sigmas(volatilities)  modifiy shape(Er)/𝝈  \
AEM            0.000291  0.000604              0.024567             0.011831   
AGI            0.000818  0.000922              0.030356             0.026952   
ATS            0.000525  0.000568              0.023823             0.022028   
BLX            0.000725  0.000564              0.023755             0.030529   
BMO            0.000341  0.000350              0.018697             0.018251   
..                  ...       ...                   ...                  ...   
TRP            0.000150  0.000363              0.019042             0.007853   
WCN            0.000594  0.000192              0.013856             0.042837   
WFG            0.000799  0.000809              0.028440             0.028086   
WPM            0.000606  0.000530              0.023014             0.026322   
X              0.000992  0.001458              0.038187             0.025974   

     initial price  
AEM      56.552120  
AGI       6.899674  
ATS      13.890000  
BLX      12.608221  
BMO      54.471703  
..             ...  
TRP      38.404881  
WCN      88.492218  
WFG      32.459225  
WPM      28.881174  
X        10.907244  

[76 rows x 5 columns]
```
```
<Figure size 2300x300 with 5 Axes>
```
```
<Figure size 2300x300 with 5 Axes>
```
    




### **3. Dimensionality Reduction & Portfolio Construction using Correlation Analysis, Clustering and PCA**



In this section, we will stack Correlation Analysis and the  Principal Component Analysis (PCA) to create a diversified portfolio containing only the most important assets with less correlation.

The Principal Component Analysis (PCA) is a dimensionality reduction technique aimed at reducing the number of assets. The PCA process will take the log returns of the assets as input and will produce a correlation matrix as output by transforming the original set of assets into a smaller set of uncorrelated variables called principal components. These components capture the majority of the variance in the data. The correlation analysis process will use the correlation matrix produced by  PCA and will analyze the correlation between the most important assets selected by PCA. the highly correlated assets that may be redundant will be dropped. The remaining assets are expected to maintain a well-diversified portfolio.


#### Correlation Analysis


```python
def generate_correlation_matrix(log_returns):
    return log_returns.corr(method='pearson')

def uncorrelated_assets_returns_log_returns_df(log_returns_df, uncorrelated_assets_list):
    return log_returns_df[uncorrelated_assets_list]

def selecting_important_assets_treshold_covariance_method(df,correlation_coefficient_treshold):

    return df[(df < correlation_coefficient_treshold).any(axis=1)].index.to_list()

def get_selected_assets_list(log_returns, correlation_coefficient_treshold):
    corr_mat = generate_correlation_matrix(log_returns)
    return selecting_important_assets_treshold_covariance_method(corr_mat,correlation_coefficient_treshold)


def get_selected_assets_log_return( frequency_date_column, log_returns, correlation_coefficient_treshold):
    frequency = frequency_date_column[0].upper()
    #selected_asset_list = selecting_uncorrelated_assets(log_returns,threshold)
    selected_asset_list = get_selected_assets_list(log_returns, correlation_coefficient_treshold)
    #display(selected_asset_list)
    selected_assets_log_returns_df = uncorrelated_assets_returns_log_returns_df(log_returns, selected_asset_list)

    for col in range(len(selected_asset_list)):
        selected_assets_log_returns_df = selected_assets_log_returns_df.rename(columns={selected_assets_log_returns_df.columns[col]:
                                                                                                                selected_asset_list[col]+' Log return'})

    selected_assets_log_returns_df[frequency_date_column] = pd.to_datetime(selected_assets_log_returns_df.index, format = '%m/%Y')
    selected_assets_log_returns_df[frequency_date_column] = selected_assets_log_returns_df[frequency_date_column].dt.to_period(frequency)

    selected_assets_log_returns_frequency_df = selected_assets_log_returns_df
    selected_assets_log_returns_frequency_df.set_index(frequency_date_column, inplace=True)
    selected_assets_log_returns = selected_assets_log_returns_frequency_df.groupby(frequency_date_column).mean()

    return selected_assets_log_returns

def selected_assets_log_return_var_covar_mat( frequency_date_column, log_returns, correlation_coefficient_treshold):
    selected_assets_log_return_df  = get_selected_assets_log_return( frequency_date_column, log_returns, correlation_coefficient_treshold)

    return generate_correlation_matrix(selected_assets_log_return_df)

def get_selected_assets_corr_mat_clustermap( frequency_date_column, log_returns, correlation_coefficient_treshold):

    selected_assets_log_return_df = get_selected_assets_log_return(frequency_date_column, log_returns,
                                                                   correlation_coefficient_treshold)

    g = sns.clustermap(selected_assets_log_return_df.corr(),  method = 'complete', cmap   = 'RdBu',  annot  = True,  annot_kws = {'size': 8})
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=60)


def get_selected_assets_volatility(assets_volatility_df, selected_content_ticker_list):

    for col in list(assets_volatility_df.columns):
        assets_volatility_df = assets_volatility_df.rename(columns={col: col.replace(' Volatility', '')})

    return assets_volatility_df[selected_content_ticker_list]


#---------------------------------------------- MAIN FUNCTION ---------------------------------------------------------------------------

def portfolio_diversification_and_assets_volatility_Corr_Analysis_main_function():
    #cumulative_variance_treshold = 1.0
    #threshold_for_highest_loadings = 0.5
    correlation_coefficient_treshold = 0.045

    selected_assets_list = get_selected_assets_list(log_returns,correlation_coefficient_treshold)


    print('\n                           ************************************************************\n'+
             '                                            All the Initial Assets Log Returns \n'+
              '                           ************************************************************\n')
    display(log_returns)

    print('\n                   ***************************************************************************\n'+
             '                       Diversified Portfolio Assets Log Returns- Correlation Analysis Methode \n'+
              '                   ***************************************************************************\n')
    selected_assets_log_return_df = get_selected_assets_log_return( 'quarter', log_returns, correlation_coefficient_treshold)
    display(selected_assets_log_return_df)

    print('\n                  *************************************************************************\n'+
             '                  Diversified Portfolio Assets Volatility - Correlation Analysis Methode   \n'+
              '                  *************************************************************************\n')
    selected_assets_volatility_df = get_selected_assets_volatility(asset_volatility_df, selected_assets_list)
    display(selected_assets_volatility_df)


    print('\n                      *************************************************************************\n'+
             '                            Diversified Portfolio Assets correlation Matrix Cluster Map \n'+
              '                      *************************************************************************\n')
    get_selected_assets_corr_mat_clustermap( 'day', log_returns, correlation_coefficient_treshold)





portfolio_diversification_and_assets_volatility_Corr_Analysis_main_function()
```

```

                           ************************************************************
                                            All the Initial Assets Log Returns 
                           ************************************************************


```
```
                 AEM       AGI       ATS       BLX       BMO        BN  \
Date                                                                     
2019-09-05 -0.044792 -0.045142  0.000000  0.031340  0.016443  0.012316   
2019-09-06 -0.030792 -0.041243  0.000000  0.007684  0.008908  0.007131   
2019-09-09 -0.030210 -0.025896  0.009316  0.023772  0.013638 -0.008073   
2019-09-10 -0.017833 -0.001545  0.000000  0.009036  0.018039 -0.011756   
2019-09-11  0.009306  0.000000  0.000000  0.039424  0.002630  0.007411   
...              ...       ...       ...       ...       ...       ...   
2024-08-26 -0.005693 -0.002552  0.001841  0.009201  0.005535  0.005483   
2024-08-27 -0.000608 -0.004096 -0.004425  0.012677 -0.063600  0.008469   
2024-08-28 -0.012968 -0.019690 -0.011896  0.001614 -0.017194 -0.006649   
2024-08-29  0.010778  0.003135 -0.000374  0.007069  0.013104  0.004437   
2024-08-30 -0.002697  0.005722  0.004479  0.003835  0.007924  0.011804   

                 BNS       BTE       BTO       BYD  ...         T        TD  \
Date                                                ...                       
2019-09-05  0.007420  0.030772  0.028079  0.023600  ...  0.004748  0.009185   
2019-09-06  0.010479  0.029853 -0.002579 -0.004174  ...  0.009981  0.010911   
2019-09-09  0.008377  0.043172  0.027067  0.042978  ...  0.014787  0.005411   
2019-09-10  0.005426  0.000000  0.018675  0.024145  ...  0.021246  0.011802   
2019-09-11  0.007547 -0.007067  0.011954  0.035725  ...  0.030401  0.001599   
...              ...       ...       ...       ...  ...       ...       ...   
2024-08-26  0.003505  0.019152  0.009959 -0.001834  ...  0.001519 -0.002868   
2024-08-27  0.027805 -0.019152 -0.011477  0.000000  ... -0.005582  0.008581   
2024-08-28 -0.021241 -0.011111  0.009674 -0.008379  ...  0.008614 -0.004365   
2024-08-29  0.006318  0.024829  0.002104  0.008546  ... -0.003032  0.000336   
2024-08-30  0.013320 -0.030431  0.004194  0.001501  ...  0.007060  0.007875   

                TFII       TPZ       TRI       TRP       WCN       WFG  \
Date                                                                     
2019-09-05  0.000000 -0.004430  0.013679 -0.011619 -0.000763 -0.006081   
2019-09-06  0.041314 -0.001666  0.001697 -0.005469  0.001743  0.000000   
2019-09-09  0.000000  0.009406 -0.021709  0.002348 -0.022784  0.030604   
2019-09-10  0.000000 -0.003310 -0.019096 -0.003327 -0.005918  0.053207   
2019-09-11  0.022345 -0.003875 -0.017369 -0.018200 -0.003029  0.005594   
...              ...       ...       ...       ...       ...       ...   
2024-08-26  0.009018  0.005571  0.005638  0.008176  0.000161  0.006889   
2024-08-27 -0.015835  0.003881  0.022125  0.005268 -0.002416 -0.015061   
2024-08-28 -0.003695 -0.012250 -0.003517 -0.005048 -0.003177 -0.006654   
2024-08-29  0.001883  0.022162 -0.004413  0.006579  0.001886  0.006317   
2024-08-30 -0.004782  0.005465  0.009450  0.012814  0.003922 -0.004621   

                 WPM         X  
Date                            
2019-09-05 -0.026317  0.029748  
2019-09-06 -0.040822 -0.018269  
2019-09-09 -0.024250  0.071156  
2019-09-10 -0.008216  0.015416  
2019-09-11  0.006792  0.063179  
...              ...       ...  
2024-08-26  0.003054  0.018144  
2024-08-27  0.002085  0.004222  
2024-08-28 -0.016799 -0.015656  
2024-08-29  0.004388  0.030812  
2024-08-30  0.002430 -0.017001  

[1256 rows x 76 columns]
```
```

                   ***************************************************************************
                       Diversified Portfolio Assets Log Returns- Correlation Analysis Methode 
                   ***************************************************************************


```
```
         AGI Log return  BLX Log return  BTO Log return  BYD Log return  \
quarter                                                                   
2019Q3        -0.012309        0.006967        0.004450        0.001329   
2019Q4         0.000609        0.001378        0.002133        0.003524   
2020Q1        -0.002932       -0.011437       -0.010657       -0.011784   
2020Q2         0.010017        0.002108        0.003883        0.005891   
2020Q3        -0.000956        0.001191       -0.000692        0.006003   
2020Q4        -0.000071        0.004429        0.005725        0.005241   
2021Q1        -0.001812       -0.000481        0.004112        0.005205   
2021Q2        -0.000283        0.000512       -0.000175        0.000667   
2021Q3        -0.000897        0.002296        0.002156        0.000443   
2021Q4         0.001082       -0.000641        0.001621        0.000560   
2022Q1         0.001512       -0.000763       -0.002177        0.000089   
2022Q2        -0.002877       -0.002303       -0.001007       -0.004458   
2022Q3         0.000897        0.000017       -0.002257       -0.000625   
2022Q4         0.004971        0.003669        0.001159        0.002182   
2023Q1         0.003108        0.001354       -0.001074        0.002657   
2023Q2        -0.000382        0.004074       -0.001725        0.001307   
2023Q3        -0.000829       -0.000459        0.000063       -0.002045   
2023Q4         0.002830        0.002637        0.002506        0.000499   
2024Q1         0.001518        0.003247        0.000090        0.001233   
2024Q2         0.000995        0.000298       -0.000492       -0.003130   
2024Q3         0.004697        0.001639        0.004034        0.001944   

         CIX Log return  ELD Log return  ERO Log return  FNV Log return  \
quarter                                                                   
2019Q3        -0.003165       -0.000018       -0.000722       -0.005514   
2019Q4         0.000353        0.000880        0.003530        0.001993   
2020Q1         0.000771       -0.002962       -0.014102       -0.000564   
2020Q2        -0.001401        0.001453        0.010335        0.005408   
2020Q3         0.001339        0.000101        0.000006        0.000021   
2020Q4        -0.000659        0.001615        0.001666       -0.001652   
2021Q1         0.004082       -0.001170        0.000851        0.000033   
2021Q2         0.002409        0.000476        0.003291        0.002358   
2021Q3         0.000144       -0.000482       -0.002642       -0.001691   
2021Q4         0.001362       -0.000474       -0.002331        0.001012   
2022Q1         0.000921       -0.000921       -0.000701        0.002337   
2022Q2        -0.000048       -0.001185       -0.008903       -0.003068   
2022Q3        -0.004366       -0.000655        0.004158       -0.001466   
2022Q4         0.002257        0.001198        0.003542        0.002147   
2023Q1        -0.000145        0.000956        0.004018        0.001106   
2023Q2         0.003240        0.000503        0.002210       -0.000320   
2023Q3        -0.002345       -0.000659       -0.002539       -0.001010   
2023Q4         0.005045        0.001270       -0.001395       -0.002905   
2024Q1         0.005166       -0.000193        0.003274        0.001243   
2024Q2        -0.005030       -0.000390        0.001641       -0.000038   
2024Q3         0.005704        0.000898       -0.000900        0.000686   

         GSY Log return  H Log return  K Log return  KEY Log return  \
quarter                                                               
2019Q3         0.000208      0.000759      0.000879        0.004676   
2019Q4         0.000097      0.003116      0.001263        0.002122   
2020Q1        -0.000254     -0.010083     -0.002142       -0.010603   
2020Q2         0.000431      0.000773      0.001669        0.002803   
2020Q3         0.000064      0.000929     -0.000225       -0.000096   
2020Q4         0.000051      0.005159     -0.000441        0.005160   
2021Q1        -0.000002      0.001767      0.000442        0.003378   
2021Q2         0.000024     -0.001002      0.000397        0.000650   
2021Q3         0.000010     -0.000109      0.000044        0.000856   
2021Q4        -0.000033      0.003409      0.000264        0.001188   
2022Q1        -0.000131     -0.000076      0.000161       -0.000407   
2022Q2        -0.000053     -0.004125      0.001763       -0.004059   
2022Q3         0.000021      0.001424     -0.000246       -0.000972   
2022Q4         0.000162      0.001759      0.000486        0.001502   
2023Q1         0.000196      0.003417     -0.000858       -0.005146   
2023Q2         0.000192      0.000420      0.000250       -0.004569   
2023Q3         0.000218     -0.001202     -0.001821        0.002719   
2023Q4         0.000322      0.003298      0.000182        0.004898   
2024Q1         0.000224      0.003330      0.000564        0.001769   
2024Q2         0.000218     -0.000769      0.000256       -0.001467   
2024Q3         0.000304      0.000023      0.007607        0.004431   

         LAAC Log return  SHOP Log return  TPZ Log return  
quarter                                                    
2019Q3         -0.002319        -0.011955        0.000107  
2019Q4          0.000603         0.003804       -0.000589  
2020Q1         -0.002829         0.000766       -0.013729  
2020Q2          0.010238         0.013059        0.004576  
2020Q3          0.012647         0.001169       -0.000649  
2020Q4          0.001515         0.001582        0.004346  
2021Q1          0.004053        -0.000373        0.001637  
2021Q2         -0.001264         0.004411        0.002179  
2021Q3          0.006384        -0.001168       -0.000340  
2021Q4          0.004148         0.000247        0.000651  
2022Q1          0.004500        -0.011481        0.001025  
2022Q2         -0.010455        -0.012449       -0.001722  
2022Q3          0.004136        -0.002314        0.000346  
2022Q4         -0.005160         0.004022        0.000717  
2023Q1          0.002230         0.005208        0.000497  
2023Q2         -0.001192         0.004811        0.000840  
2023Q3         -0.002736        -0.002678        0.000545  
2023Q4         -0.001273         0.005650        0.001124  
2024Q1         -0.002609        -0.000154        0.002107  
2024Q2         -0.008276        -0.002470        0.000527  
2024Q3         -0.004372         0.002605        0.003693  
```
```

                  *************************************************************************
                  Diversified Portfolio Assets Volatility - Correlation Analysis Methode   
                  *************************************************************************


```
```
        AGI  BLX  BTO  BYD  CIX  ELD  ERO  FNV  GSY    H    K  KEY  LAAC  \
Quater                                                                     
2020Q3  0.7  0.7  0.8  0.9  0.6  0.2  0.7  0.4  0.0  0.6  0.3  0.7   0.8   
2020Q4  0.7  0.7  0.8  0.9  0.7  0.2  0.7  0.4  0.0  0.6  0.3  0.7   1.0   
2021Q1  0.7  0.6  0.8  0.9  0.6  0.2  0.7  0.4  0.0  0.6  0.3  0.7   1.1   
2021Q2  0.5  0.4  0.4  0.5  0.6  0.1  0.5  0.3  0.0  0.5  0.2  0.5   1.0   
2021Q3  0.5  0.3  0.3  0.4  0.6  0.1  0.5  0.3  0.0  0.4  0.2  0.4   1.0   
2021Q4  0.4  0.2  0.3  0.4  0.6  0.1  0.5  0.3  0.0  0.4  0.2  0.3   0.9   
2022Q1  0.4  0.2  0.3  0.4  0.5  0.1  0.5  0.3  0.0  0.3  0.2  0.3   0.8   
2022Q2  0.4  0.2  0.4  0.4  0.4  0.1  0.5  0.3  0.0  0.4  0.2  0.3   0.8   
2022Q3  0.4  0.2  0.4  0.4  0.4  0.1  0.6  0.3  0.0  0.4  0.2  0.3   0.8   
2022Q4  0.5  0.3  0.4  0.4  0.4  0.1  0.6  0.3  0.0  0.4  0.2  0.3   0.8   
2023Q1  0.4  0.3  0.4  0.4  0.4  0.1  0.7  0.3  0.0  0.4  0.2  0.4   0.7   
2023Q2  0.4  0.3  0.4  0.3  0.5  0.1  0.7  0.3  0.0  0.4  0.2  0.5   0.6   
2023Q3  0.4  0.3  0.4  0.3  0.5  0.1  0.6  0.3  0.0  0.3  0.2  0.6   0.5   
2023Q4  0.3  0.3  0.4  0.3  0.4  0.1  0.5  0.3  0.0  0.3  0.2  0.6   0.6   
2024Q1  0.3  0.3  0.4  0.3  0.6  0.1  0.5  0.3  0.0  0.3  0.2  0.6   0.6   
2024Q2  0.3  0.3  0.3  0.3  0.7  0.1  0.5  0.3  0.0  0.3  0.2  0.4   0.6   
2024Q3  0.3  0.3  0.3  0.3  0.8  0.1  0.5  0.3  0.0  0.3  0.2  0.4   0.6   

        SHOP  TPZ  
Quater             
2020Q3   0.7  0.8  
2020Q4   0.7  0.8  
2021Q1   0.7  0.7  
2021Q2   0.6  0.3  
2021Q3   0.5  0.2  
2021Q4   0.5  0.2  
2022Q1   0.6  0.2  
2022Q2   0.7  0.2  
2022Q3   0.9  0.2  
2022Q4   0.9  0.2  
2023Q1   0.9  0.2  
2023Q2   0.8  0.2  
2023Q3   0.7  0.2  
2023Q4   0.6  0.1  
2024Q1   0.6  0.1  
2024Q2   0.5  0.1  
2024Q3   0.5  0.1  
```
```

                      *************************************************************************
                            Diversified Portfolio Assets correlation Matrix Cluster Map 
                      *************************************************************************


```
```
<Figure size 1000x1000 with 4 Axes>
```
####  Principal Components Analysis(PCA)


```python

#Selecting most important economic factors
#-------------------------------------------------------------------------------
 #Principal Components Analysis(PCA) to select most importance assets
#-------------------------------------------------------------------------------



def selecting_important_item_PCA_treshold_method(matrix,threshold):

    return matrix[(matrix.abs() > threshold).any(axis=1)].index.to_list()

def selecting_important_item_corr_treshold_method(matrix,threshold):

    return matrix[(matrix < threshold).any(axis=1)].index.to_list()

def setting_PCA_for_assets_selection(log_returns_df):
    # economic indicators dataset

    # Standardizing the data
    scaler = StandardScaler()
    scaled_data_df = scaler.fit_transform(log_returns_df)

    # Applying PCA
    all_pca = PCA(n_components=None)  # Use all components to find the best number of important indicators
    all_principal_components = all_pca.fit_transform(scaled_data_df)

    # Explained variance
    explained_variance = all_pca.explained_variance_ratio_

    # Principal Component Loadings(coefficients)
    loadings_matrix = all_pca.components_

    # Create a DataFrame for loadings
    loadings_matrix_df = pd.DataFrame(loadings_matrix.T, columns=[f'PC{i+1}' for i in range(loadings_matrix.shape[0])],
                                      index=log_returns.columns)


    return loadings_matrix_df, explained_variance

#----------------------

def get_num_components(explained_variance,cumulative_variance_treshold = 0.9):
    # Determine the number of components explaining the cumulative varience treshold of the variance
    cumulative_variance = explained_variance.cumsum()
    return  (cumulative_variance <= cumulative_variance_treshold).sum() + 1

def select_top_components_df(loadings_matrix_df, num_components, threshold_for_high_loadings = 0.5):
    # Select top components
    return loadings_matrix_df.iloc[:, :num_components]

def select_top_indicators_df(loadings_matrix_df, num_components, threshold_for_high_loadings = 0.5):
    # Select top components
    selected_components_df = loadings_matrix_df.iloc[:, :num_components]
    # Find indicators with high loadings
    return selected_components_df[(selected_components_df.abs() > threshold_for_high_loadings).any(axis=1)]


def plot_explained_variance_for_assets_selection(loadings_matrix_df, explained_variance):

    # Print explained variance

    explained_variance_df = pd.DataFrame(explained_variance).T
    explained_variance_df.columns = loadings_matrix_df.columns
    print('\nexplained_variance_df\n')
    display(explained_variance_df)

    # Plotting the explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='individual explained variance')
    plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', label='cumulative explained variance')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.legend(loc='best')
    plt.show()


#----------------------
def print_explained_variance(loadings_matrix_df, explained_variance,cumulative_variance_treshold, num_components, threshold_for_highest_loadings):
     # Print explained variance
    print('\nloadings_matrix_df\n')
    display(loadings_matrix_df)
    num_components = get_num_components(explained_variance,cumulative_variance_treshold)
    top_components_df = select_top_components_df(loadings_matrix_df, num_components, threshold_for_highest_loadings)
    print('\ntop_components_df\n')
    display(top_components_df)
    print('\nMost important assets with top components\n')
    top_indicators_df = select_top_indicators_df(loadings_matrix_df, num_components, threshold_for_highest_loadings)
    display(top_indicators_df)



def get_all_assets_corr_matrix(log_returns_df, cumulative_variance_treshold = 1, threshold_for_highest_loadings = 0.5 ):

    all_assets_matrix =  generate_correlation_matrix(log_returns_df)
    return all_assets_matrix


def get_most_important_assets_list_PCA(log_returns_df, cumulative_variance_treshold = 1, threshold_for_highest_loadings = 0.5 ):

    loadings_matrix_df, explained_variance = setting_PCA_for_assets_selection(log_returns_df)
    #print('\nloadings_matrix_df\n')
    #display(loadings_matrix_df)
    num_components = get_num_components(explained_variance,cumulative_variance_treshold)
    top_components_df = select_top_components_df(loadings_matrix_df, num_components, threshold_for_highest_loadings)
    #print('\ntop_components_df\n')
    #display(top_components_df)
    #print('\ntop_indicators_df\n')
    top_indicators_df = select_top_indicators_df(loadings_matrix_df, num_components, threshold_for_highest_loadings)
    #display(top_indicators_df)
    most_important_assets_list = selecting_important_item_PCA_treshold_method(top_indicators_df, threshold_for_highest_loadings)

    return most_important_assets_list


def get_most_important_assets_log_returns_df_PCA(log_returns_df, cumulative_variance_treshold = 1, threshold_for_highest_loadings = 0.5 ):

    most_important_assets_log_returns_list = get_most_important_assets_list_PCA(log_returns_df,
                                                                    cumulative_variance_treshold, threshold_for_highest_loadings)

    most_important_assets_log_returns_df = log_returns_df[most_important_assets_log_returns_list]
    #print('\n most_important_assets_df\n')
    #display(most_important_assets_log_returns_df)

    return most_important_assets_log_returns_df


def get_most_important_assets_corr_matrix_PCA(log_returns_df, cumulative_variance_treshold = 1, threshold_for_highest_loadings = 0.5 ):

    most_important_assets_log_returns_df = get_most_important_assets_log_returns_df_PCA(log_returns_df,
                                                                    cumulative_variance_treshold , threshold_for_highest_loadings)

    #PCA couple with covarience matrice to select most important portfolio assets
    most_important_assets_matrix =  generate_correlation_matrix(most_important_assets_log_returns_df)

    return most_important_assets_matrix

#----------------------------------------------------------------------------------------------------------
 #Stack Correlation Analysis and Principal Components Analysis(PCA) to select most divesified assets
#-------------------------------------------------------------------------------------------------------------

def get_most_diversify_portfolio_asset_log_return_df_stack_corr_PCA(log_returns_df,most_diversify_portfolio_assets_list):
    return log_returns_df[most_diversify_portfolio_assets_list]


def using_PCA__and_corr_matrix_to_diversify_portfolio(log_returns_df, cumulative_variance_treshold = 1, threshold_for_highest_loadings = 0.5,
                                              correlation_coefficient_treshold= 0.40):

    most_important_assets_corr_matrix = get_most_important_assets_corr_matrix_PCA(log_returns_df, cumulative_variance_treshold,
                                                                                          threshold_for_highest_loadings)

    most_diversify_portfolio_assets_list = selecting_important_item_corr_treshold_method(most_important_assets_corr_matrix,
                                                                                              correlation_coefficient_treshold)

    most_diversify_portfolio_assets_df = log_returns_df[most_diversify_portfolio_assets_list]
    most_diversify_portfolio_assets_corr_matrix = generate_correlation_matrix(most_diversify_portfolio_assets_df)

    return most_diversify_portfolio_assets_corr_matrix

def plotting_selected_assets_corr_mat_clustermap(assets_matrix, title, dendrogram = True):

    g = sns.clustermap(assets_matrix,  method = 'ward', metric='euclidean', cmap   = 'RdBu',  annot  = True,  annot_kws = {'size': 8},
                      row_cluster=dendrogram, col_cluster=dendrogram)
    g.fig.suptitle('Diversified Portfolio Assets Log Returns Correlation Matrix Cluster Map using PCA', y=0.9, fontsize=12)
    plt.subplots_adjust(top=0.85)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=360)
    g.cax.set_position([1.02, 0.2, 0.03, 0.4])  # [left, bottom, width, height]
    g.cax.set_ylabel('Correlation Coefficient', rotation=270, labelpad=15)  # Rotate label
    g.fig.suptitle(title, y=0.9, fontsize=12)


 #----------------------------------------------------------------------------------------------------------
 #                              Most divesified Assets Daily Volatility
#-------------------------------------------------------------------------------------------------------------
#selected assets daily volatility
def get_selected_assets_volatility_df_from_Stack_Corr_PCA_method(selected_assets_adj_close_price_log_return_df, frequency_date_column = 'day'):

    frequency = frequency_date_column[0].upper()

    #selected_assets_adj_close_price_log_return_df = get_most_important_assets_log_returns_df_PCA(log_returns)

    #Market volatility

    selected_assets_volatility_df = selected_assets_adj_close_price_log_return_df.rolling(center=False,window= 252).std() * np.sqrt(252)
    for col in list(selected_assets_volatility_df.columns):
        selected_assets_volatility_df = selected_assets_volatility_df.rename(columns={col: col+' Volatility'})

    selected_assets_volatility_df = selected_assets_volatility_df.dropna(axis=0)

    if frequency == 'D':
        selected_assets_volatilities = selected_assets_volatility_df
    else:
        selected_assets_volatility_df[frequency_date_column] = pd.to_datetime(selected_assets_volatility_df.index, format = '%m/%Y')
        selected_assets_volatility_df[frequency_date_column] = selected_assets_volatility_df[frequency_date_column].dt.to_period(frequency)

        #market_adj_close_price_log_return_frequency_df = market_volatility_df
        selected_assets_volatility_df.set_index(frequency_date_column, inplace=True)
        selected_assets_volatilities = selected_assets_volatility_df.groupby(frequency_date_column).mean()
        selected_assets_volatilities = round(selected_assets_volatilities,1)
        selected_assets_volatilities = selected_assets_volatilities.dropna(axis=0)

    return selected_assets_volatilities

#--------------------------------------------------MAIN FUNCION --------------------------------------------------------------
def select_most_important_Portfolio_assets_and_diversification_stack_corr_PCA_mathod_main_function():
    cumulative_variance_treshold = 1.0
    threshold_for_highest_loadings = 0.5
    correlation_coefficient_treshold = 0.3
    #-------------------------------------
    loadings_matrix_df, explained_variance  = setting_PCA_for_assets_selection(log_returns)
    num_components = get_num_components(explained_variance,cumulative_variance_treshold)
    top_components_df = select_top_components_df(loadings_matrix_df, num_components, threshold_for_highest_loadings)
    top_indicators_df = select_top_indicators_df(loadings_matrix_df, num_components, threshold_for_highest_loadings)
    most_important_assets_list = selecting_important_item_PCA_treshold_method(top_indicators_df, threshold_for_highest_loadings)
    #------------------------------------
    plot_explained_variance_for_assets_selection(loadings_matrix_df, explained_variance)
    print_explained_variance(loadings_matrix_df, explained_variance,cumulative_variance_treshold, num_components, threshold_for_highest_loadings)
    most_important_assets_log_returns_df_PCA = get_most_important_assets_log_returns_df_PCA(log_returns, cumulative_variance_treshold ,
                                                                                        threshold_for_highest_loadings)

    most_important_assets_corr_matrix_PCA = get_most_important_assets_corr_matrix_PCA(log_returns, cumulative_variance_treshold,
                                                                                  threshold_for_highest_loadings )

    most_diversify_portfolio_assets_list = selecting_important_item_corr_treshold_method(most_important_assets_corr_matrix_PCA,
                                                                                              correlation_coefficient_treshold)

    most_diversify_portfolio_assets_log_returns_df = get_most_diversify_portfolio_asset_log_return_df_stack_corr_PCA(log_returns,
                                                                                                         most_diversify_portfolio_assets_list)

    most_diversify_portfolio_assets_corr_matrix = using_PCA__and_corr_matrix_to_diversify_portfolio(log_returns, cumulative_variance_treshold ,
                                                  threshold_for_highest_loadings, correlation_coefficient_treshold )

    selected_assets_volatility_df_stack_corr_PCA_method = \
                    get_selected_assets_volatility_df_from_Stack_Corr_PCA_method(most_diversify_portfolio_assets_log_returns_df,
                                                                                            frequency_date_column = 'day')

#-------graphs and tables--------------------------------------------------
    print('\nMost Important Assets Log returns  using PCA\n')
    display(most_important_assets_log_returns_df_PCA)
    plotting_selected_assets_corr_mat_clustermap(most_important_assets_corr_matrix_PCA, 'Most Important Assets Correlation Matrix PCA Method')
    print('\nMost Diversified Assets Log returns  using Stack Correlation Matrix/PCA Method\n')
    display(most_diversify_portfolio_assets_log_returns_df)
    plotting_selected_assets_corr_mat_clustermap(most_diversify_portfolio_assets_corr_matrix,
                                                 'Most Diversified Assets Correlation Matrix - stacking Correlation Analysis/PCA Method')
    print('\nDiversified Portfolio Assets Volatility \n')
    display(selected_assets_volatility_df_stack_corr_PCA_method)

select_most_important_Portfolio_assets_and_diversification_stack_corr_PCA_mathod_main_function()
```

```

explained_variance_df


```
```
        PC1       PC2       PC3       PC4       PC5       PC6       PC7  \
0  0.413571  0.092048  0.042029  0.033425  0.028264  0.022387  0.016672   

        PC8       PC9      PC10  ...      PC67      PC68      PC69    PC70  \
0  0.016169  0.012889  0.012505  ...  0.001654  0.001622  0.001451  0.0014   

       PC71      PC72      PC73      PC74      PC75      PC76  
0  0.001346  0.001299  0.000955  0.000908  0.000633  0.000318  

[1 rows x 76 columns]
```
```
<Figure size 1000x600 with 1 Axes>
```
```

loadings_matrix_df


```
```
          PC1       PC2       PC3       PC4       PC5       PC6       PC7  \
AEM -0.061928 -0.300673 -0.018603 -0.033758 -0.043640  0.041924  0.078728   
AGI -0.054214 -0.317213 -0.019314 -0.045093 -0.045763  0.090384 -0.002040   
ATS -0.074365  0.000179  0.016299  0.124022  0.071638 -0.140825 -0.232579   
BLX -0.107064  0.056913 -0.040209  0.044130 -0.129863  0.300048  0.074065   
BMO -0.154090  0.072810 -0.055718 -0.052644  0.043137 -0.065054 -0.137062   
..        ...       ...       ...       ...       ...       ...       ...   
TRP -0.135361  0.025675 -0.085641 -0.110914 -0.078325 -0.136525  0.097596   
WCN -0.101385  0.003086  0.169273 -0.183968  0.079093 -0.059743  0.243405   
WFG -0.111772 -0.004493 -0.017282  0.011134  0.003261 -0.047796 -0.191205   
WPM -0.066090 -0.309043  0.011015 -0.047076  0.025778 -0.029112 -0.026209   
X   -0.092889 -0.002042 -0.105657  0.064158  0.117769  0.187691 -0.118618   

          PC8       PC9      PC10  ...      PC67      PC68      PC69  \
AEM  0.048601 -0.074202 -0.031446  ... -0.150138  0.010027 -0.125825   
AGI -0.005456 -0.035734  0.009407  ...  0.181376 -0.121596  0.062707   
ATS  0.015172 -0.192035 -0.326613  ... -0.005813  0.006769  0.005296   
BLX -0.026251 -0.141236  0.094920  ...  0.030163 -0.038492  0.002709   
BMO  0.124146 -0.044911  0.083070  ... -0.036197  0.120071  0.275386   
..        ...       ...       ...  ...       ...       ...       ...   
TRP  0.112444  0.060010  0.160832  ... -0.105272  0.342310  0.055739   
WCN -0.027527  0.027130  0.038142  ... -0.057424 -0.042657  0.041802   
WFG  0.068690  0.132980 -0.215008  ...  0.005610  0.005589  0.007621   
WPM  0.010502 -0.065610 -0.020476  ...  0.194700  0.047612  0.001477   
X   -0.192140  0.180294 -0.017633  ... -0.004554 -0.021461  0.017062   

         PC70      PC71      PC72      PC73      PC74      PC75      PC76  
AEM -0.016254  0.082313  0.017581 -0.009547 -0.003375 -0.011353  0.018102  
AGI -0.123947 -0.047106  0.035094 -0.061341 -0.021900 -0.012963 -0.019846  
ATS  0.004956 -0.003904 -0.010756  0.026051 -0.019479 -0.010712  0.000526  
BLX -0.034415  0.018790  0.019224 -0.004271 -0.044736  0.023852 -0.015881  
BMO -0.716781  0.185395 -0.152984 -0.059495 -0.056711  0.032720 -0.002581  
..        ...       ...       ...       ...       ...       ...       ...  
TRP  0.120079  0.022928  0.040598  0.035714  0.035335 -0.039161 -0.005182  
WCN  0.003885 -0.026465  0.054365 -0.008726  0.002437  0.000136 -0.012959  
WFG -0.025308  0.021077  0.000838 -0.010475 -0.021437  0.006819  0.010910  
WPM -0.209939  0.020964  0.072836  0.010258 -0.005362 -0.329228 -0.018788  
X    0.020064 -0.034882 -0.005765  0.006582 -0.034042 -0.000958  0.008777  

[76 rows x 76 columns]
```
```

top_components_df


```
```
          PC1       PC2       PC3       PC4       PC5       PC6       PC7  \
AEM -0.061928 -0.300673 -0.018603 -0.033758 -0.043640  0.041924  0.078728   
AGI -0.054214 -0.317213 -0.019314 -0.045093 -0.045763  0.090384 -0.002040   
ATS -0.074365  0.000179  0.016299  0.124022  0.071638 -0.140825 -0.232579   
BLX -0.107064  0.056913 -0.040209  0.044130 -0.129863  0.300048  0.074065   
BMO -0.154090  0.072810 -0.055718 -0.052644  0.043137 -0.065054 -0.137062   
..        ...       ...       ...       ...       ...       ...       ...   
TRP -0.135361  0.025675 -0.085641 -0.110914 -0.078325 -0.136525  0.097596   
WCN -0.101385  0.003086  0.169273 -0.183968  0.079093 -0.059743  0.243405   
WFG -0.111772 -0.004493 -0.017282  0.011134  0.003261 -0.047796 -0.191205   
WPM -0.066090 -0.309043  0.011015 -0.047076  0.025778 -0.029112 -0.026209   
X   -0.092889 -0.002042 -0.105657  0.064158  0.117769  0.187691 -0.118618   

          PC8       PC9      PC10  ...      PC67      PC68      PC69  \
AEM  0.048601 -0.074202 -0.031446  ... -0.150138  0.010027 -0.125825   
AGI -0.005456 -0.035734  0.009407  ...  0.181376 -0.121596  0.062707   
ATS  0.015172 -0.192035 -0.326613  ... -0.005813  0.006769  0.005296   
BLX -0.026251 -0.141236  0.094920  ...  0.030163 -0.038492  0.002709   
BMO  0.124146 -0.044911  0.083070  ... -0.036197  0.120071  0.275386   
..        ...       ...       ...  ...       ...       ...       ...   
TRP  0.112444  0.060010  0.160832  ... -0.105272  0.342310  0.055739   
WCN -0.027527  0.027130  0.038142  ... -0.057424 -0.042657  0.041802   
WFG  0.068690  0.132980 -0.215008  ...  0.005610  0.005589  0.007621   
WPM  0.010502 -0.065610 -0.020476  ...  0.194700  0.047612  0.001477   
X   -0.192140  0.180294 -0.017633  ... -0.004554 -0.021461  0.017062   

         PC70      PC71      PC72      PC73      PC74      PC75      PC76  
AEM -0.016254  0.082313  0.017581 -0.009547 -0.003375 -0.011353  0.018102  
AGI -0.123947 -0.047106  0.035094 -0.061341 -0.021900 -0.012963 -0.019846  
ATS  0.004956 -0.003904 -0.010756  0.026051 -0.019479 -0.010712  0.000526  
BLX -0.034415  0.018790  0.019224 -0.004271 -0.044736  0.023852 -0.015881  
BMO -0.716781  0.185395 -0.152984 -0.059495 -0.056711  0.032720 -0.002581  
..        ...       ...       ...       ...       ...       ...       ...  
TRP  0.120079  0.022928  0.040598  0.035714  0.035335 -0.039161 -0.005182  
WCN  0.003885 -0.026465  0.054365 -0.008726  0.002437  0.000136 -0.012959  
WFG -0.025308  0.021077  0.000838 -0.010475 -0.021437  0.006819  0.010910  
WPM -0.209939  0.020964  0.072836  0.010258 -0.005362 -0.329228 -0.018788  
X    0.020064 -0.034882 -0.005765  0.006582 -0.034042 -0.000958  0.008777  

[76 rows x 76 columns]
```
```

Most important assets with top components


```
```
          PC1       PC2       PC3       PC4       PC5       PC6       PC7  \
AGI -0.054214 -0.317213 -0.019314 -0.045093 -0.045763  0.090384 -0.002040   
ATS -0.074365  0.000179  0.016299  0.124022  0.071638 -0.140825 -0.232579   
BMO -0.154090  0.072810 -0.055718 -0.052644  0.043137 -0.065054 -0.137062   
BN  -0.148970  0.045362  0.071590 -0.013753  0.065388 -0.014771 -0.054065   
BTO -0.129136  0.092535 -0.022307 -0.017024 -0.121852  0.114454 -0.197616   
CIX -0.055212 -0.016288  0.002035  0.029668 -0.086048  0.204812 -0.014785   
CNQ -0.129724  0.049238 -0.268474  0.045882  0.044639 -0.134181  0.188874   
CWB -0.143854  0.000884  0.170220  0.152204  0.038359 -0.093760  0.020776   
DOL -0.161772  0.002449  0.044724 -0.026729  0.082273 -0.017104 -0.027028   
DOO -0.158596  0.001864  0.037617 -0.037232  0.076613  0.000041 -0.053165   
ENB -0.146013  0.035765 -0.100768 -0.085941 -0.050137 -0.157815  0.101209   
IGM -0.128902 -0.003623  0.249091  0.142736  0.183396 -0.038799  0.104929   
PEY -0.151227  0.051928 -0.020777 -0.197706  0.037814  0.166523 -0.050712   
SIL -0.081061 -0.311668 -0.026162  0.018422 -0.015973  0.034826 -0.056791   
SLF -0.153174  0.048941 -0.005763 -0.088314  0.053784 -0.044194 -0.094464   
TD  -0.150345  0.067267 -0.058720 -0.102146  0.065792 -0.045129 -0.167000   
WFG -0.111772 -0.004493 -0.017282  0.011134  0.003261 -0.047796 -0.191205   

          PC8       PC9      PC10  ...      PC67      PC68      PC69  \
AGI -0.005456 -0.035734  0.009407  ...  0.181376 -0.121596  0.062707   
ATS  0.015172 -0.192035 -0.326613  ... -0.005813  0.006769  0.005296   
BMO  0.124146 -0.044911  0.083070  ... -0.036197  0.120071  0.275386   
BN   0.124413  0.021003  0.043324  ... -0.031771  0.060318 -0.138004   
BTO  0.060695  0.035904  0.007855  ... -0.080556  0.037866 -0.024051   
CIX -0.145424  0.258110  0.389522  ...  0.001290 -0.008863  0.013643   
CNQ -0.090676 -0.033049 -0.005931  ...  0.162870  0.072036  0.008063   
CWB -0.135562  0.059152  0.034551  ...  0.292793 -0.227813 -0.215333   
DOL -0.073629 -0.150492  0.079942  ... -0.021253  0.046153 -0.002184   
DOO -0.080575 -0.155516  0.069618  ... -0.059032  0.056226 -0.015360   
ENB  0.051884 -0.003982  0.117162  ...  0.146058 -0.584737  0.018011   
IGM -0.125032 -0.036014  0.091382  ...  0.080119  0.009791 -0.067273   
PEY -0.111970  0.039265 -0.035241  ... -0.034462 -0.054046 -0.090142   
SIL -0.013075 -0.063445 -0.010083  ...  0.017741  0.066615 -0.052449   
SLF  0.034118 -0.024130  0.021328  ... -0.232536  0.048768 -0.353892   
TD   0.058571 -0.045571  0.099347  ... -0.009191  0.039243 -0.091276   
WFG  0.068690  0.132980 -0.215008  ...  0.005610  0.005589  0.007621   

         PC70      PC71      PC72      PC73      PC74      PC75      PC76  
AGI -0.123947 -0.047106  0.035094 -0.061341 -0.021900 -0.012963 -0.019846  
ATS  0.004956 -0.003904 -0.010756  0.026051 -0.019479 -0.010712  0.000526  
BMO -0.716781  0.185395 -0.152984 -0.059495 -0.056711  0.032720 -0.002581  
BN   0.000106  0.006984 -0.144015 -0.124427 -0.053054  0.019647  0.005267  
BTO  0.022130 -0.059100 -0.022345  0.011791 -0.055302 -0.027939 -0.018902  
CIX -0.003692 -0.022874 -0.021235  0.007475 -0.010454 -0.023243 -0.000405  
CNQ  0.088833 -0.010372 -0.659676 -0.072624  0.160442  0.091158 -0.003918  
CWB  0.106362  0.269248  0.111149 -0.184335  0.084492 -0.041895 -0.019573  
DOL  0.018226 -0.002938 -0.015618 -0.053428 -0.044170  0.003260  0.749315  
DOO  0.031245  0.031729 -0.045749  0.029585  0.007027 -0.057994 -0.649662  
ENB -0.156185  0.016373 -0.028183 -0.084224 -0.046819  0.035148 -0.015616  
IGM -0.052730  0.005293 -0.142015  0.713820 -0.330229  0.085619  0.009130  
PEY -0.114874  0.082819  0.068073  0.378847  0.774294 -0.107697  0.067495  
SIL  0.025368 -0.006615  0.115071 -0.014371  0.117720  0.807479 -0.038236  
SLF -0.070497  0.116209 -0.161623 -0.026336  0.033871  0.026828  0.012075  
TD   0.040564 -0.494215 -0.009929 -0.007028  0.038986 -0.001102 -0.027592  
WFG -0.025308  0.021077  0.000838 -0.010475 -0.021437  0.006819  0.010910  

[17 rows x 76 columns]
```
```

Most Important Assets Log returns  using PCA


```
```
                 AGI       ATS       BMO        BN       BTO       CIX  \
Date                                                                     
2019-09-05 -0.045142  0.000000  0.016443  0.012316  0.028079 -0.053481   
2019-09-06 -0.041243  0.000000  0.008908  0.007131 -0.002579  0.006930   
2019-09-09 -0.025896  0.009316  0.013638 -0.008073  0.027067 -0.002766   
2019-09-10 -0.001545  0.000000  0.018039 -0.011756  0.018675  0.020563   
2019-09-11  0.000000  0.000000  0.002630  0.007411  0.011954  0.066275   
...              ...       ...       ...       ...       ...       ...   
2024-08-26 -0.002552  0.001841  0.005535  0.005483  0.009959 -0.012112   
2024-08-27 -0.004096 -0.004425 -0.063600  0.008469 -0.011477  0.042652   
2024-08-28 -0.019690 -0.011896 -0.017194 -0.006649  0.009674  0.003180   
2024-08-29  0.003135 -0.000374  0.013104  0.004437  0.002104  0.037727   
2024-08-30  0.005722  0.004479  0.007924  0.011804  0.004194  0.014500   

                 CNQ       CWB       DOL       DOO       ENB       IGM  \
Date                                                                     
2019-09-05  0.026257  0.003223  0.005728  0.006324  0.012107  0.020621   
2019-09-06 -0.016591  0.000946  0.001317  0.001078  0.007894 -0.003107   
2019-09-09  0.033720 -0.002840  0.004378  0.000667  0.001746 -0.005562   
2019-09-10  0.037696 -0.001139  0.005664  0.006024  0.009259 -0.007008   
2019-09-11 -0.010176  0.006622  0.005199  0.006077  0.001439  0.010132   
...              ...       ...       ...       ...       ...       ...   
2024-08-26  0.026545 -0.000945 -0.003348 -0.002865  0.005783 -0.009842   
2024-08-27 -0.012643  0.001214  0.005204  0.003964 -0.005530  0.003863   
2024-08-28 -0.011435 -0.004189 -0.003714 -0.003118 -0.007591 -0.013042   
2024-08-29  0.010893  0.002570  0.003343  0.001844  0.004814 -0.000217   
2024-08-30 -0.019418  0.003371  0.001667  0.000102  0.015551  0.010579   

                 PEY       SIL       SLF        TD       WFG  
Date                                                          
2019-09-05  0.003369 -0.046697  0.018389  0.009185 -0.006081  
2019-09-06  0.005032 -0.030040  0.008717  0.010911  0.000000  
2019-09-09  0.011093 -0.019287  0.010036  0.005411  0.030604  
2019-09-10  0.010971  0.006131  0.004865  0.011802  0.053207  
2019-09-11  0.013547  0.013826  0.003461  0.001599  0.005594  
...              ...       ...       ...       ...       ...  
2024-08-26  0.003697  0.000887  0.001284 -0.002868  0.006889  
2024-08-27 -0.006943 -0.001183  0.002563  0.008581 -0.015061  
2024-08-28  0.002783 -0.032780 -0.002906 -0.004365 -0.006654  
2024-08-29  0.002775  0.005792  0.004255  0.000336  0.006317  
2024-08-30  0.006904 -0.005181  0.006440  0.007875 -0.004621  

[1256 rows x 17 columns]
```
```

Most Diversified Assets Log returns  using Stack Correlation Matrix/PCA Method


```
```
                 AGI       ATS       BMO        BN       BTO       CIX  \
Date                                                                     
2019-09-05 -0.045142  0.000000  0.016443  0.012316  0.028079 -0.053481   
2019-09-06 -0.041243  0.000000  0.008908  0.007131 -0.002579  0.006930   
2019-09-09 -0.025896  0.009316  0.013638 -0.008073  0.027067 -0.002766   
2019-09-10 -0.001545  0.000000  0.018039 -0.011756  0.018675  0.020563   
2019-09-11  0.000000  0.000000  0.002630  0.007411  0.011954  0.066275   
...              ...       ...       ...       ...       ...       ...   
2024-08-26 -0.002552  0.001841  0.005535  0.005483  0.009959 -0.012112   
2024-08-27 -0.004096 -0.004425 -0.063600  0.008469 -0.011477  0.042652   
2024-08-28 -0.019690 -0.011896 -0.017194 -0.006649  0.009674  0.003180   
2024-08-29  0.003135 -0.000374  0.013104  0.004437  0.002104  0.037727   
2024-08-30  0.005722  0.004479  0.007924  0.011804  0.004194  0.014500   

                 CNQ       CWB       DOL       DOO       ENB       IGM  \
Date                                                                     
2019-09-05  0.026257  0.003223  0.005728  0.006324  0.012107  0.020621   
2019-09-06 -0.016591  0.000946  0.001317  0.001078  0.007894 -0.003107   
2019-09-09  0.033720 -0.002840  0.004378  0.000667  0.001746 -0.005562   
2019-09-10  0.037696 -0.001139  0.005664  0.006024  0.009259 -0.007008   
2019-09-11 -0.010176  0.006622  0.005199  0.006077  0.001439  0.010132   
...              ...       ...       ...       ...       ...       ...   
2024-08-26  0.026545 -0.000945 -0.003348 -0.002865  0.005783 -0.009842   
2024-08-27 -0.012643  0.001214  0.005204  0.003964 -0.005530  0.003863   
2024-08-28 -0.011435 -0.004189 -0.003714 -0.003118 -0.007591 -0.013042   
2024-08-29  0.010893  0.002570  0.003343  0.001844  0.004814 -0.000217   
2024-08-30 -0.019418  0.003371  0.001667  0.000102  0.015551  0.010579   

                 PEY       SIL       SLF        TD       WFG  
Date                                                          
2019-09-05  0.003369 -0.046697  0.018389  0.009185 -0.006081  
2019-09-06  0.005032 -0.030040  0.008717  0.010911  0.000000  
2019-09-09  0.011093 -0.019287  0.010036  0.005411  0.030604  
2019-09-10  0.010971  0.006131  0.004865  0.011802  0.053207  
2019-09-11  0.013547  0.013826  0.003461  0.001599  0.005594  
...              ...       ...       ...       ...       ...  
2024-08-26  0.003697  0.000887  0.001284 -0.002868  0.006889  
2024-08-27 -0.006943 -0.001183  0.002563  0.008581 -0.015061  
2024-08-28  0.002783 -0.032780 -0.002906 -0.004365 -0.006654  
2024-08-29  0.002775  0.005792  0.004255  0.000336  0.006317  
2024-08-30  0.006904 -0.005181  0.006440  0.007875 -0.004621  

[1256 rows x 17 columns]
```
```

Diversified Portfolio Assets Volatility 


```
```
            AGI Volatility  ATS Volatility  BMO Volatility  BN Volatility  \
Date                                                                        
2020-09-02        0.722088        0.380698        0.494260       0.510384   
2020-09-03        0.720578        0.383126        0.494137       0.510851   
2020-09-04        0.719338        0.383447        0.494077       0.511048   
2020-09-08        0.719311        0.383335        0.494701       0.511587   
2020-09-09        0.719990        0.389331        0.494570       0.511666   
...                    ...             ...             ...            ...   
2024-08-26        0.336985        0.328255        0.216709       0.289643   
2024-08-27        0.336736        0.327967        0.225907       0.289722   
2024-08-28        0.334901        0.328120        0.226021       0.289590   
2024-08-29        0.334237        0.327515        0.226337       0.288030   
2024-08-30        0.334262        0.327472        0.226044       0.288189   

            BTO Volatility  CIX Volatility  CNQ Volatility  CWB Volatility  \
Date                                                                         
2020-09-02        0.780097        0.622903        0.815006        0.255488   
2020-09-03        0.779591        0.620621        0.814673        0.258030   
2020-09-04        0.779606        0.620899        0.814549        0.258336   
2020-09-08        0.779410        0.621174        0.819958        0.260213   
2020-09-09        0.779173        0.620827        0.819753        0.260624   
...                    ...             ...             ...             ...   
2024-08-26        0.260475        0.772576        0.276750        0.084813   
2024-08-27        0.260503        0.773628        0.276778        0.084779   
2024-08-28        0.260545        0.773491        0.276080        0.084880   
2024-08-29        0.260516        0.774304        0.275127        0.084525   
2024-08-30        0.260478        0.774409        0.275850        0.084472   

            DOL Volatility  DOO Volatility  ENB Volatility  IGM Volatility  \
Date                                                                         
2020-09-02        0.307450        0.295969        0.471170        0.351598   
2020-09-03        0.307956        0.296476        0.471263        0.355219   
2020-09-04        0.307954        0.296514        0.471448        0.355847   
2020-09-08        0.308082        0.296812        0.471670        0.358251   
2020-09-09        0.308533        0.297460        0.471982        0.358998   
...                    ...             ...             ...             ...   
2024-08-26        0.121523        0.121095        0.172435        0.208262   
2024-08-27        0.121467        0.121064        0.172460        0.208210   
2024-08-28        0.121071        0.120928        0.172492        0.208574   
2024-08-29        0.120509        0.120158        0.172256        0.207535   
2024-08-30        0.120504        0.120157        0.172824        0.207681   

            PEY Volatility  SIL Volatility  SLF Volatility  TD Volatility  \
Date                                                                        
2020-09-02        0.387438        0.545365        0.445792       0.435493   
2020-09-03        0.387536        0.543424        0.446122       0.435766   
2020-09-04        0.387522        0.542502        0.446038       0.435661   
2020-09-08        0.387716        0.542682        0.446219       0.436037   
2020-09-09        0.387579        0.543947        0.446900       0.436161   
...                    ...             ...             ...            ...   
2024-08-26        0.165861        0.333416        0.180709       0.187160   
2024-08-27        0.166029        0.333366        0.180717       0.187258   
2024-08-28        0.165680        0.334371        0.180722       0.186212   
2024-08-29        0.164891        0.333743        0.179847       0.185355   
2024-08-30        0.164875        0.333793        0.179833       0.185476   

            WFG Volatility  
Date                        
2020-09-02        0.686560  
2020-09-03        0.689481  
2020-09-04        0.689529  
2020-09-08        0.690172  
2020-09-09        0.688212  
...                    ...  
2024-08-26        0.302854  
2024-08-27        0.303123  
2024-08-28        0.303210  
2024-08-29        0.303163  
2024-08-30        0.303192  

[1005 rows x 17 columns]
```
```
<Figure size 1000x1000 with 4 Axes>
```
```
<Figure size 1000x1000 with 4 Axes>
```
        

        

        

        




## **4. Asset Pricing, Profit & Lost simation and Risk calculation**

##### In  this section, we will focus on :

 - ##### Monte Carlo simulation of stock price using cov matrix and cholesky decomposition

 - ##### Profit & Lost simulation

 - ##### VaR/CVaR calculation under current macroeconomic factors

                            

Before going forward with our analysis, it is crucial to understand the form of the data distribution, here stock price and asset returns distribution over time in other to choose the appropriate model.

The daily adjust closed prices charts above show that the stock prices movement and their returns over time folow an independent random process, with stock prices always positive and ncrease indefinitely. The distribution exhibits positive skewnes.The future price movement doesn't depend on its history, but it is determined by  both its current state and some inherent randomness such as economic indicators, company performance, investor sentiment, geopolitical events, and unforeseen news. Therefore, the stock prices are considered stochastic. In general we don’t know the distribution of the stock prices, we only know that it is closed to the brownan motion stockastic process. Typically, the logarithm of the stock price follows a Brownian motion with drift.



Stochastic differential equation (SDE) of the stock price 𝑆(𝑡):

$$

dS(t) = \mu S(t) \, d(t) + \sigma S(t) \, dW(t)

$$

where:

- $dS(t)$  is the variation (absolute change) of the stock price over the time interval d(t)

- $\mu$ is the drift term (expected return),

- $\sigma$ is the volatility of the stock,

- $W(t)$ is a Wiener process (Geometric Brownian motion).



In a simple term, $dS(t) = S(t+d(t))$ , with $S(t)$ representing the stock price at the time $t$.

logarithmicly the continuous compounding return of the stock over the interval $d(t)$ is $ r(t) = \log\left(\frac{S(t+d(t))}{S(t)}\right)$.  

**The volatility** $\sigma$ is the square root of the variance. It provides a measure of the risk or uncertainty of the stock price. $\sigma$ is a key parameter of the Geometric Brownian motion that determines the stochastic variation of the stock price.  

**The variance** of the stock price over a given period is a measure of the magnitude of expected price fluctuations. It is the mathematical expectation of the squared deviation between the price and its mean.  

**$W(t)$ is a random variable  that follows a Wiener process.** It is the random component of the stock price movement and is related to the variation of time d(t). The mathematical expression of this relationship is : $dW(t)=\epsilon \sqrt{dt}$.

In this expression, the term "epsilon" represents a random variable whose distribution is normal with expected value of zero(mean zero) and variance equal 1. It's mathematical expection is $E(dW(t))=\sqrt{dt}E(epsilon)=0$ with the variance $Var(dW(t))=d(t)Var(epsilon)=d(t)$.  

The variations of $W(t)$ are independent over time. In the case the company associated with that stock does not distribute a portion of its profits to shareholders in the form of dividend payments,  the stochastic equation for the return of a stock is:$\frac{dS(t)}{S(t)} = \mu d(t) + \sigma dW(t)$.

Therefore, it makes sense that the return on a stock does not depend on the price of the stock.  





Let's now focus on the stochastic equation for the return. We will dig into this yow part : $\mu d(t) and \sigma dW(t)$.  

The first part, $\mu d(t)$, is deterministic meaning that, using the historical data, we can calculate the expected change in the stock price over the small time interval $dt$, assuming no randomness. Essentially, $mu$ is the expected rate of return per unit time, and when multiplied by the stock price $S(t)$ and the time interval $dt$, it gives the expected change in the stock price due to predictable factors like steady growth, interest rates, or dividends.  

Tthe second part, $\sigma dW(t)$, is the Stochastic or randomness part of the stock rate of return. It takes in to considaration the unpredictable fluctuations in the stock price due to various factors like market volatility, company-Specific news, or economic factors, geopolitical events,natural disasters and pandemic, investor behavior and sentiment, technological advances and disruptions, global economic interdependencies. The $dW(t)$ represents the random shock to the stock price, and $sigma$ scales this shock, making it more or less volatile. Let's look inside the solution of the stochatic equation of the stock price.



Quation1:

  $S(t) = S(0) \exp \left( \left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W(t) \right)$



or in more details



Quation2:  $S(t) = S(0) \exp \left( \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma \phi \sqrt{dt} \right)$



In the equation2, the term $ \phi$ represent correlated normal distributions with standard deviation equal 1  and expected value of zero(mean zero). But from Geometric Brownian Motion prostective, the stock price movement is independent over time(uncorrelated) and follow a log-normal distribution with a mean of zero and a standard deviation that depends on the time interval dt. In order to come out of this situation, we will procide as follow:



- Calculate log returns of the stock prices

- Calculate the expected return, the variance and the volatility of each stock

- Calculate the variance-covariance matrix

- Calculate cholesky decomposition matrix.

- Simulate an uncorrelated random normal distribution with $mu = 0 and sigma = 1$(Z distribution)

- Apply the cholesky matrix to the uncorrelated random normal distribution(Z distribution) in order to get a correlated random normal normal distribution with with $mu = 0  and  sigma = 1$.

- Use correlated random normal normal distribution as input for the stock price function.



After then, we will simulate the portfolio Profit & Lost and finanly we will calculate the portfolio VaR(value at Rick and the CVaR(conditional Value at Risk)



```python
# Data collection-
cumulative_variance_treshold = 1.0
threshold_for_highest_loadings = 0.5
correlation_coefficient_treshold = 0.3
#-------------------------------------
loadings_matrix_df, explained_variance  = setting_PCA_for_assets_selection(log_returns)
num_components = get_num_components(explained_variance,cumulative_variance_treshold)
top_components_df = select_top_components_df(loadings_matrix_df, num_components, threshold_for_highest_loadings)
top_indicators_df = select_top_indicators_df(loadings_matrix_df, num_components, threshold_for_highest_loadings)
most_important_assets_list = selecting_important_item_PCA_treshold_method(top_indicators_df, threshold_for_highest_loadings)
#------------------------------------
#plot_explained_variance_for_assets_selection(loadings_matrix_df, explained_variance)
#print_explained_variance(loadings_matrix_df, explained_variance,cumulative_variance_treshold, num_components, threshold_for_highest_loadings)
most_important_assets_log_returns_df_PCA = get_most_important_assets_log_returns_df_PCA(log_returns, cumulative_variance_treshold ,
                                                                                        threshold_for_highest_loadings)

most_important_assets_corr_matrix_PCA = get_most_important_assets_corr_matrix_PCA(log_returns, cumulative_variance_treshold,
                                                                                  threshold_for_highest_loadings )

most_diversify_portfolio_assets_list = selecting_important_item_corr_treshold_method(most_important_assets_corr_matrix_PCA,
                                                                                              correlation_coefficient_treshold)

most_diversify_portfolio_assets_log_returns_df = get_most_diversify_portfolio_asset_log_return_df_stack_corr_PCA(log_returns,
                                                                                                       most_diversify_portfolio_assets_list)

most_diversify_portfolio_assets_corr_matrix = using_PCA__and_corr_matrix_to_diversify_portfolio(log_returns, cumulative_variance_treshold ,
                                                  threshold_for_highest_loadings, correlation_coefficient_treshold )

selected_assets_volatility_df_stack_corr_PCA_method = \
                    get_selected_assets_volatility_df_from_Stack_Corr_PCA_method(most_diversify_portfolio_assets_log_returns_df,
                                                                                            frequency_date_column = 'day')

most_diversify_portfolio_assets_initial_prices =  stocks_initial_prices[most_diversify_portfolio_assets_list]
#most_diversify_portfolio_assets_list


```

#### Correlation - Covariance & Cholesky decomposition



- **Covariance**: Covariance measures the degree to which two variables (e.g., asset returns) move together. It tells us whether the returns of two assets tend to rise and fall together (positive covariance) or move in opposite directions (negative covariance). Zero Covariance means that there is  no linear relationship between the assets' returns. A mix of assets with low or negative covariances can reduce overall portfolio risk.

- **Mathematical Formula**:  

$$

  Cov(X, Y) =

[

\frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X})(Y_i - \bar{Y})

]

$$

Where:



  - $( X_i )$ and $( Y_i )$ are the returns of assets $(X)$ and $(Y)$.

  - $( \bar{X} )$ and $( \bar{Y} )$ are the mean returns of $(X)$ and $(Y)$.  

  - \$( n )$ is the number of observations.  

  





 - **Correlation**: Correlation is a normalized version of covariance, which measures the strength and direction of the linear relationship between two variables (asset returns). Unlike covariance, correlation is dimensionless and always ranges between -1 and 1. Correlation is used to measure the degree of diversification in a portfolio. Combining assets with low or negative correlations can significantly reduce portfolio risk. Portfolio managers use correlation to understand how different assets are likely to behave relative to one another under various market conditions.



- **Mathematical Formula**:

$$

Correlation(X, Y) =

[

\frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}

]

$$

Where:



  - $Cov(X, Y)$ is the covariance of $( X )$ and $( Y )$.

  - $( \sigma_X )$  and $( \sigma_Y )$ are the standard deviations of $( X )$ and $( Y )$.  

  

  

 - **Cholesky decomposition**: Cholesky decomposition is a mathematical technique used of decomposing a positive-definite matrix into the product of a lower triangular matrix and its transpose. We will use Cholesky decomposition methode to decompose  the covariance matrix into the product of a lower triangular matrix and its transpose(cholesky Matrix). This will help us to generate correlated asset returns and the stock price.



- **For a positive-definite matrix $( A )$, the Cholesky decomposition is expressed as:**

$$

A = LL^\top

$$

Where:





 - $( L )$ is a lower triangular matrix.

 - $( L^\top )$ is the transpose of $( L )$.  





 - **Suppose you have a covariance matrix of asset returns:**

$$

\Sigma =

\begin{pmatrix}

\sigma_{11} & \sigma_{12} & \dots & \sigma_{1n} \\

\sigma_{21} & \sigma_{22} & \dots & \sigma_{2n} \\

\vdots & \vdots & \ddots & \vdots \\

\sigma_{n1} & \sigma_{n2} & \dots & \sigma_{nn}

\end{pmatrix}

$$

The Cholesky decomposition would allow you to write:



$$\Sigma = LL^\top$$



Where:





 - \$( L )$ is used to generate correlated random variables from uncorrelated normal variables, which is essential for realistic financial simulations.



```python
def plotting_heatmap_for_correlation_matrix(log_returns, title):
    plt.figure(figsize=(20, 8))
    #sns.heatmap(log_returns.corr(), annot=True)
    sns.heatmap(log_returns.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, fmt=".2f")
    plt.yticks(rotation=360)
    plt.title(title, pad= 20)

def plotting_heatmap_for_covariance_matrix(covariance_matrix, title):
    plt.figure(figsize=(20, 8))
    sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt=".5f",
            linewidths=0.5, vmin=covariance_matrix.min().min(), vmax=covariance_matrix.max().max())
    plt.yticks(rotation=360)
    plt.title(title, pad= 20)


def variance_covariance_matrix(log_returns):
    return log_returns.cov()


#-----------------------------------------------------------------------------------------------------------------------------------------------
#create cholesky matrice: let's apply cholesky decomposition to the covarience matrix
# Input: covarience matrice
# output: cholesky matrice data frame
#------------------------------------------------------------------------------------------------------------------------------------------------
def create_cholesky_matrix(covar_mat):
    cholesky_matrix_data = np.linalg.cholesky(covar_mat)
    return pd.DataFrame(cholesky_matrix_data[0:,0:], columns=covar_mat.columns.tolist(), index=covar_mat.columns.tolist())


covar_mat = variance_covariance_matrix(most_diversify_portfolio_assets_log_returns_df)
cholesky_matrix_data_df = create_cholesky_matrix(covar_mat)

plotting_heatmap_for_correlation_matrix(most_diversify_portfolio_assets_log_returns_df,
                                        'Correlation Matrix of the Most Diversified portfolio Asset Log Returns')

plotting_heatmap_for_covariance_matrix(covar_mat, 'Covariance Matrix of the Most Diversified portfolio  Asset Log Returns')
plotting_heatmap_for_covariance_matrix(cholesky_matrix_data_df, 'Cholesky Matrix of the Most Diversified portfolio Asset Log Returns')
```

```
<Figure size 2000x800 with 2 Axes>
```
```
<Figure size 2000x800 with 2 Axes>
```
```
<Figure size 2000x800 with 2 Axes>
```
    


#### Uncorelated normal distribution


Uncorrelated Normal(epsilon T)

Two random variables are said to be uncorrelated if their covariance is zero.

Two variables that are uncorrelated are not necessarily independent, as is simply exemplified by the fact that X

 and X2  are uncorrelated but not independent. However, two variables that are uncorrelated AND jointly normally distributed are guaranteed to be independent

https://stats.stackexchange.com/questions/376229/uncorrelatedness-joint-normality-independence-why-intuition-and-mechanics#:~:text=Two%20variables%20that%20are%20uncorrelated,are%20guaranteed%20to%20be%20independent.



https://stackoverflow.com/questions/20626994/how-to-calculate-the-inverse-of-the-normal-cumulative-distribution-function-in-p



```python
#--------------------------------------------------------------------------------------------------------
# here let's simulate 10000 uncorelated normal distribution  iterations to calculate the stock price.
#input:covariance matrice and number of iteration
#output 10000 Z score for each stock price: uncorelated normal z core array and  it's  data frame
# here we simulate 10000 uncorelated normal distribution  iterations to calculate the stock price.
#t_intervals = 250
#number_of_assets = len(covar_mat.columns.tolist())
#Z = norm.ppf(np.random.rand(iterations,number_of_assets ))
#--------------------------------------------------------------------------------------------------------
def simulate_uncorelated_normal_distribution(covar_mat,iterations):
    number_of_assets = len(covar_mat.columns.tolist())
    #z score array
    Z = norm.ppf(np.random.rand(iterations,number_of_assets ))
    Z_df = pd.DataFrame(data=Z[0:,0:],index=[i for i in range(Z.shape[0])], columns=covar_mat.columns.tolist())
    return Z,Z_df

Z, Z_df = simulate_uncorelated_normal_distribution(covar_mat,10000)
display(Z_df)

```

```
           AGI       ATS       BMO        BN       BTO       CIX       CNQ  \
0    -0.014251  0.984673 -0.640661  0.352552 -0.755954  0.140241  1.938027   
1    -1.584557  0.132888 -0.022475 -0.359053  0.433362  0.642457  0.848172   
2     1.339836 -0.074981  0.286637 -1.427705  0.887084  1.096874  0.877216   
3     2.042191  2.554160 -0.092553 -0.940519 -0.539154  0.871144 -0.814800   
4     0.460667  1.242499 -0.543992  0.617708 -0.294869 -0.574476 -1.302641   
...        ...       ...       ...       ...       ...       ...       ...   
9995  2.285962  1.143190  1.666773  0.532412  1.102743 -0.710522 -1.217958   
9996 -0.933452  1.226422 -0.991998  1.561508 -0.183356 -0.794913 -1.173983   
9997 -0.328184  0.758332 -0.102508  0.295453 -0.330564  0.955908  0.539927   
9998 -0.293886  0.107442  1.428256 -0.851825 -0.274065 -2.167839  0.613583   
9999 -0.819552 -0.773844 -0.987715 -0.167020 -1.913879 -0.357072 -1.020579   

           CWB       DOL       DOO       ENB       IGM       PEY       SIL  \
0     0.351559  1.425819  0.566410 -1.130777 -1.427924  0.119983 -0.035151   
1    -0.120482  0.403287 -0.235679  1.367996 -0.608351  0.223716  0.449225   
2     1.518152 -0.023004 -0.807409 -0.540780  0.993497 -0.064737 -1.635076   
3    -1.159940  0.683020  1.625901 -0.168646 -0.418753 -0.831418 -0.267564   
4    -0.514947 -0.179331 -0.729605 -0.315258 -0.551196 -0.196510  0.440043   
...        ...       ...       ...       ...       ...       ...       ...   
9995 -1.082297 -1.439800 -0.390220 -0.258851  1.963762 -0.245368  0.245116   
9996  0.004532  0.444666 -0.384535  1.112723 -1.895295  0.312740  0.413843   
9997  0.505346 -0.646064 -1.538100  1.222877  0.495787 -1.100979 -0.748121   
9998  1.315447 -1.104972  0.858727  0.894246  1.060963  0.020778  0.757783   
9999  0.788049 -1.103704  1.080072 -1.159892  2.201218  0.115906  0.669816   

           SLF        TD       WFG  
0     1.431731  0.253041 -0.757660  
1     0.142766  1.121109  0.280437  
2     0.120613  0.783371 -0.411260  
3    -0.108139  1.265697 -0.745670  
4     0.884956 -0.572972 -1.587416  
...        ...       ...       ...  
9995 -0.007518 -0.323493  1.080169  
9996 -1.040315 -0.258575  2.573085  
9997 -1.073875 -1.654978  1.593181  
9998  0.646817 -0.243252  0.293175  
9999  0.807538  0.674188  0.170408  

[10000 rows x 17 columns]
```
#### Correlated normal distribution


```python
#-----------------------------------------------------------------------------------------------------
#Description: generate correlated normal distribution using transposed cholesky matrix and uncorelated
#normal Z score distribution
#=MMULT(unCorrelated_normal_distribution,TRANSPOSE(cholesky_matrix))
#------------------------------------------------------------------------------------------------------
def generate_correlated_normal_distribution(cholesky_matrix_data_df,Z):
    Correlated_Normals_Z = np.matmul(Z, cholesky_matrix_data_df.T)
    Correlated_Normals_Z_arr = np.array(Correlated_Normals_Z)
    return Correlated_Normals_Z, Correlated_Normals_Z_arr

Correlated_Normals_Z, Correlated_Normals_Z_arr= generate_correlated_normal_distribution(cholesky_matrix_data_df,Z)
display(Correlated_Normals_Z)

```

```
           AGI       ATS       BMO        BN       BTO       CIX       CNQ  \
0    -0.000433  0.023327 -0.004859  0.003156 -0.017646  0.000687  0.040785   
1    -0.048101 -0.000311 -0.002368 -0.010031  0.005826  0.014631  0.015728   
2     0.040672  0.001150  0.006916 -0.012391  0.018651  0.047821  0.028297   
3     0.061993  0.065054  0.018510  0.011930  0.007635  0.043610  0.006362   
4     0.013984  0.030482 -0.000659  0.012116 -0.002435 -0.018094 -0.027576   
...        ...       ...       ...       ...       ...       ...       ...   
9995  0.069393  0.032115  0.040559  0.049526  0.068828  0.008496  0.021225   
9996 -0.028336  0.027053 -0.011077  0.014160 -0.006581 -0.034278 -0.035863   
9997 -0.009962  0.017272  0.002515  0.007524 -0.001475  0.032531  0.016739   
9998 -0.008921  0.001906  0.025102  0.008800  0.017954 -0.070304  0.037551   
9999 -0.024879 -0.020148 -0.023700 -0.026080 -0.066007 -0.033995 -0.053334   

           CWB       DOL       DOO       ENB       IGM       PEY       SIL  \
0     0.004622  0.011653  0.012442 -0.001786 -0.002495  0.003362  0.007600   
1    -0.003324 -0.001681 -0.002503  0.013993 -0.013458  0.001131 -0.025045   
2     0.013226  0.008036  0.005036  0.005002  0.020782  0.004158  0.011082   
3     0.001888  0.014788  0.019384  0.009638  0.001728  0.006743  0.046047   
4    -0.000943 -0.000407 -0.002153 -0.007351 -0.002267 -0.003426  0.015101   
...        ...       ...       ...       ...       ...       ...       ...   
9995  0.011909  0.014665  0.013923  0.022675  0.033419  0.024448  0.052189   
9996  0.000139 -0.000937 -0.002007  0.001828 -0.012249 -0.001147 -0.012758   
9997  0.006982  0.000924 -0.003587  0.017175  0.011131 -0.008088 -0.012801   
9998  0.014037  0.004767  0.006670  0.024769  0.022313  0.008877  0.008923   
9999 -0.007765 -0.021662 -0.018028 -0.038104  0.009916 -0.024527 -0.017775   

           SLF        TD       WFG  
0     0.017540  0.005148 -0.008301  
1     0.000725  0.009813  0.002893  
2     0.007770  0.010744  0.000395  
3     0.013391  0.023005  0.005917  
4     0.005626 -0.004179 -0.032156  
...        ...       ...       ...  
9995  0.026274  0.026996  0.062650  
9996 -0.012302 -0.006405  0.053131  
9997 -0.007647 -0.011772  0.031650  
9998  0.019311  0.015381  0.028035  
9999 -0.018233 -0.020770 -0.022374  

[10000 rows x 17 columns]
```
#### Daily returns simulation

$Daily returns = e^{\left( \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma \phi \sqrt{dt} \right)}$


```python
#--------------------------------------------------------------------------------------------------------------
#Description: Daily returns simulation (returns simulation = 𝒆^(((𝝁𝒊−(𝟏/𝟐)𝝈𝒊𝟐)(𝒕𝟐−𝒕𝟏)+𝝈𝒊√((𝒕𝟐−𝒕𝟏) ) 𝝓)))
#Inputs:
#   𝝓 : Correlated_Normals_Z
#   𝝓_arr : Correlated_Normals_Z_array
#   𝝁 : log_returns.mean()
#   variance: log_returns.var()
#   𝝈 : log_returns.std()
#   𝓢1 : initial_prices
#   delta_t = 1
#output:
 #--------------------------------------------------------------------------------------------------------------
def simulate_daily_returns(𝝓,𝝓_arr, 𝝁,𝝈,delta_t):
    daily_returns_list_df = np.zeros_like(𝝓_arr)
    daily_returns_list_df = np.exp((𝝁 - 0.5 * 𝝈** 2) * delta_t + 𝝈* delta_t ** 0.5 *𝝓)
    return daily_returns_list_df


#daily_returns_list_df = simulate_daily_returns(Correlated_Normals_Z,Correlated_Normals_Z_arr, log_returns.mean(),log_returns.std(),1)
daily_returns_df = simulate_daily_returns(Correlated_Normals_Z,Correlated_Normals_Z_arr,
                                               most_diversify_portfolio_assets_log_returns_df.mean(),
                                               most_diversify_portfolio_assets_log_returns_df.std(),1)
daily_returns_df
```

```
           AGI       ATS       BMO        BN       BTO       CIX       CNQ  \
0     1.000344  1.000797  1.000076  1.000315  0.999432  1.000139  1.001916   
1     0.998898  1.000234  1.000122  1.000018  1.000120  1.000653  1.001140   
2     1.001593  1.000268  1.000296  0.999965  1.000496  1.001876  1.001529   
3     1.002242  1.001792  1.000513  1.000513  1.000173  1.001721  1.000850   
4     1.000782  1.000968  1.000154  1.000517  0.999878  0.999448  0.999801   
...        ...       ...       ...       ...       ...       ...       ...   
9995  1.002467  1.001007  1.000925  1.001361  1.001969  1.000427  1.001310   
9996  0.999497  1.000886  0.999959  1.000563  0.999756  0.998852  0.999545   
9997  1.000055  1.000653  1.000213  1.000414  0.999906  1.001312  1.001171   
9998  1.000087  1.000286  1.000636  1.000442  1.000475  0.997529  1.001816   
9999  0.999602  0.999761  0.999723  0.999656  0.998016  0.998863  0.999006   

           CWB       DOL       DOO       ENB       IGM       PEY       SIL  \
0     1.000359  1.000358  1.000242  1.000212  1.000569  1.000283  0.999917   
1     1.000273  1.000196  1.000063  1.000489  1.000378  1.000250  0.999066   
2     1.000451  1.000314  1.000153  1.000331  1.000976  1.000295  1.000008   
3     1.000329  1.000396  1.000325  1.000413  1.000643  1.000333  1.000921   
4     1.000299  1.000212  1.000067  1.000115  1.000573  1.000183  1.000113   
...        ...       ...       ...       ...       ...       ...       ...   
9995  1.000437  1.000394  1.000260  1.000641  1.001196  1.000593  1.001081   
9996  1.000310  1.000205  1.000068  1.000276  1.000399  1.000217  0.999386   
9997  1.000384  1.000228  1.000050  1.000545  1.000807  1.000115  0.999385   
9998  1.000460  1.000274  1.000173  1.000678  1.001002  1.000364  0.999952   
9999  1.000225  0.999954  0.999876  0.999576  1.000786  0.999873  0.999256   

           SLF        TD       WFG  
0     1.000518  1.000202  1.000158  
1     1.000237  1.000280  1.000477  
2     1.000355  1.000296  1.000406  
3     1.000448  1.000502  1.000563  
4     1.000319  1.000046  0.999480  
...        ...       ...       ...  
9995  1.000663  1.000569  1.002178  
9996  1.000020  1.000008  1.001907  
9997  1.000097  0.999918  1.001295  
9998  1.000547  1.000374  1.001192  
9999  0.999921  0.999768  0.999758  

[10000 rows x 17 columns]
```
```python
#--------------------------------------------------------------------------------------------------------------------------
# Description: Stock price simulation
def stock_prices_simulation(initial_prices,daily_returns_list_df ):
    𝓢1_list = []
    𝓢1_list = initial_prices.values
    expo_r = daily_returns_list_df
    future_stock_price_list_df= pd.DataFrame(data=daily_returns_list_df[0:0:],
                                             index=[i for i in range(daily_returns_list_df.shape[0])],
                                             columns=expo_r.columns.tolist())
    for (index, column) in enumerate(expo_r):
        future_stock_price_list_df[column] = pd.DataFrame(data=𝓢1_list[index]*expo_r[column].values)
    return future_stock_price_list_df

future_stock_price_df = stock_prices_simulation(most_diversify_portfolio_assets_initial_prices,daily_returns_df)
future_stock_price_df
```

```
           AGI        ATS        BMO         BN        BTO        CIX  \
0     6.902050  13.901071  54.475821  26.914950  20.777522  11.056767   
1     6.892070  13.893245  54.478358  26.906952  20.791823  11.062444   
2     6.910668  13.893729  54.487816  26.905521  20.799642  11.075969   
3     6.915142  13.914896  54.499628  26.920272  20.792926  11.074252   
4     6.905071  13.903441  54.480098  26.920385  20.786788  11.049125   
...        ...        ...        ...        ...        ...        ...   
9995  6.916696  13.903982  54.522101  26.943091  20.830260  11.059946   
9996  6.896206  13.902305  54.469488  26.921625  20.784262  11.042543   
9997  6.900054  13.899066  54.483332  26.917599  20.787374  11.069737   
9998  6.900272  13.893979  54.506346  26.918374  20.799217  11.027908   
9999  6.896930  13.886681  54.456634  26.897222  20.748085  11.042659   

           CNQ        CWB        DOL        DOO        ENB        IGM  \
0     8.764450  46.878844  37.546200  34.983736  24.397128  35.655474   
1     8.757666  46.874833  37.540140  34.977469  24.403878  35.648653   
2     8.761068  46.883187  37.544556  34.980630  24.400031  35.669959   
3     8.755132  46.877464  37.547626  34.986648  24.402015  35.658101   
4     8.745954  46.876035  37.540719  34.977615  24.394748  35.655616   
...        ...        ...        ...        ...        ...        ...   
9995  8.759154  46.882523  37.547570  34.984357  24.407593  35.677826   
9996  8.743715  46.876581  37.540478  34.977676  24.398674  35.649406   
9997  8.757940  46.880035  37.541323  34.977014  24.405240  35.663953   
9998  8.763574  46.883597  37.543070  34.981315  24.408489  35.670912   
9999  8.738995  46.872591  37.531060  34.970959  24.381598  35.663196   

            PEY        SIL        SLF         TD        WFG  
0     14.198505  30.371363  34.527311  43.431187  32.464362  
1     14.198040  30.345503  34.517628  43.434585  32.474700  
2     14.198671  30.374122  34.521684  43.435263  32.472392  
3     14.199210  30.401847  34.524921  43.444193  32.477492  
4     14.197089  30.377308  34.520450  43.424395  32.442345  
...         ...        ...        ...        ...        ...  
9995  14.202903  30.406720  34.532342  43.447101  32.529936  
9996  14.197565  30.355234  34.510128  43.422774  32.521131  
9997  14.196117  30.355199  34.512808  43.418867  32.501269  
9998  14.199655  30.372411  34.528331  43.438640  32.497928  
9999  14.192690  30.351260  34.506714  43.412316  32.451371  

[10000 rows x 17 columns]
```
#### Stock prices simulation - Protfolio Profit and Lost calculation

$S(t) = S(0) \exp \left( \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma \phi \sqrt{dt} \right)$



```python
#Initial portfolio price
def calculate_initial_portfolio_price(𝓢1):
    return 𝓢1.values.sum()

#Portfolio price simulation
def simulated_portfolio_price(future_stock_price_df):
    simulated_portfolio_price_row_sum = []
    for i in range(len(future_stock_price_df)):
        simulated_portfolio_price_row_sum.append(future_stock_price_df.iloc[i].sum())
    simulated_portfolio_price_df = pd.DataFrame(data = simulated_portfolio_price_row_sum, columns=['portfolio_prices'])
    return simulated_portfolio_price_df

#-----------------------------------------------------------------------------------------------------------
#Description: Portfolio Profit and lost calculation
# input:simulated_portfolio_price_df,portfolio_initial_price
# output :portfolio_profit_and_lost_df
#-----------------------------------------------------------------------------------------------------------
def calculate_prtfolio_profit_and_lost(simulated_portfolio_price_df, portfolio_initial_price):
    portfolio_profit_and_lost_df = simulated_portfolio_price_df - portfolio_initial_price
    portfolio_profit_and_lost_df.columns = ['profit_&_lost']
    return portfolio_profit_and_lost_df

def set_portfolio_price_profit_and_Lost_simulation_df(simulated_portfolio_price_df, portfolio_profit_and_lost_df):
    return  pd.DataFrame({'simulated_portfolio_price': simulated_portfolio_price_df['portfolio_prices'].values,
                          'Simulated Portfolio Profit & Lost': portfolio_profit_and_lost_df['profit_&_lost'].values})


#index=[i for i in range(portfolio_profit_and_lost_df.shape[0])],
#                        columns=simulated_portfolio_price_df.columns, portfolio_profit_and_lost_df.columns
```

```python
simulated_portfolio_price_df= simulated_portfolio_price(future_stock_price_df)
initial_portfolio_prices = calculate_initial_portfolio_price(most_diversify_portfolio_assets_initial_prices)
portfolio_profit_and_lost_df = calculate_prtfolio_profit_and_lost(simulated_portfolio_price_df, initial_portfolio_prices)

simulated_portfolio_price_profit_and_Lost_df = set_portfolio_price_profit_and_Lost_simulation_df(simulated_portfolio_price_df,
                                                                                                portfolio_profit_and_lost_df)


print('initial_portfolio_prices')
display(initial_portfolio_prices)
print('\nSimulated Portfolio Prices - Profit & Lost')
display(simulated_portfolio_price_profit_and_Lost_df)

```

```
initial_portfolio_prices

```
```
477.11676597595215
```
```

Simulated Portfolio Prices - Profit & Lost

```
```
      simulated_portfolio_price  Simulated Portfolio Profit & Lost
0                    477.246739                           0.129973
1                    477.197986                           0.081220
2                    477.314910                           0.198144
3                    477.391765                           0.274999
4                    477.197181                           0.080415
...                         ...                                ...
9995                 477.554099                           0.437333
9996                 477.209791                           0.093025
9997                 477.266926                           0.150160
9998                 477.334018                           0.217252
9999                 477.000961                          -0.115805

[10000 rows x 2 columns]
```
#### Portfolio VaR (Value at Risk) and CVaR calculation

While VaR represents a worst-case loss associated with a probability and a time horizon, CVaR is the expected loss if that worst-case threshold is ever crossed. CVaR, in other words, quantifies the expected losses that occur beyond the VaR breakpoint. CVaR is the average loss over a specified time period of unlikely scenarios beyond the confidence level.

https://www.investopedia.com/terms/c/conditional_value_at_risk.asp#:~:text=While%20VaR%20represents%20a%20worst,occur%20beyond%20the%20VaR%20breakpoint.


```python
#-------------------------------------------------------------------------------------------------------------------------
# Description:sorting profit and lost ascendante; confifence level rank; Var calculation;CVar calculation
# Input:
# Output:
#-------------------------------------------------------------------------------------------------------------------------

def calculate_portfolio_Var_and_CVar(portfolio_profit_and_lost_df, confidence_level):
    #sorting profit and lost ascendante
    lportfolio_profit_and_lost_df = portfolio_profit_and_lost_df.sort_values(by='profit_&_lost', ascending=True)
    lportfolio_profit_and_lost_df = portfolio_profit_and_lost_df.reset_index(drop=True)
    #confifence level rank ( 95% confidence lavel)
    rank = int((1-confidence_level)*len(lportfolio_profit_and_lost_df))-1
    #Var calculation
    VaR = portfolio_profit_and_lost_df.iloc[rank]['profit_&_lost']
    #CVar calculation
    port_folio_lost_beyond_VaR = portfolio_profit_and_lost_df[:rank]
    CVaR = np.average(port_folio_lost_beyond_VaR)
    return VaR, CVaR

#--------------------------------------------------------------------------------------------------------------------------
#Decription: Profit and lost summary statistics. Minimum lost, maximum lost, mean(moderate lost)lost standart deviation,
#            Value at risk(Var),Conditional value-at-risk (CVaR)
#Input :portfolio_profit_and_lost_df,VaR,CVaR
#Output:
#--------------------------------------------------------------------------------------------------------------------------
def profit_and_lost_summary_statistics(portfolio_profit_and_lost_df,VaR,CVaR):
    VaR_and_CVaR_df = pd.DataFrame([{'VaR':VaR, 'CVaR':CVaR}]).transpose()
    VaR_and_CVaR_df = VaR_and_CVaR_df.rename(columns={0:'profit_&_lost'})
    portfolio_profit_and_lost_stat_df =  portfolio_profit_and_lost_df.agg(['min', 'max', 'mean', 'std'])
    return pd.concat([portfolio_profit_and_lost_stat_df,VaR_and_CVaR_df], ignore_index=False)

# Plot a histogram
def profit_lost_summary(portfolio_profit_and_lost_df):
    fig, ax = plt.subplots(figsize=(8, 4))
    portfolio_profit_and_lost_df.plot.kde(ax=ax, legend=True, title='Histogram: Profit & Lost')
    portfolio_profit_and_lost_df.plot.hist(density=True, ax=ax)
    ax.set_ylabel('Probability')
    ax.grid(axis='y')
    ax.set_facecolor('#d8dcd6')
    plt.show()

def summary_statistics_graph_and_table(portfolio_profit_and_lost_df):
    profit_lost_summary(portfolio_profit_and_lost_df)
    profit_and_lost_summary_statistics_df = profit_and_lost_summary_statistics(portfolio_profit_and_lost_df,VaR,CVaR)
    display(profit_and_lost_summary_statistics_df)


VaR, CVaR = calculate_portfolio_Var_and_CVar(portfolio_profit_and_lost_df, 0.95)
profit_and_lost_summary_statistics_df = profit_and_lost_summary_statistics(portfolio_profit_and_lost_df,VaR,CVaR)
summary_statistics_graph_and_table(portfolio_profit_and_lost_df)
```

```
<Figure size 800x400 with 1 Axes>
```
```
      profit_&_lost
min       -0.379296
max        0.550121
mean       0.102792
std        0.135651
VaR        0.146044
CVaR       0.094615
```
## Portfolio Optimization


#### Portfolio Expected Return and Volatility Simulation - Random Efficient Frontier


```python
def portfolio_random_weight_array_df(assets_returns_df):
    #random portfolio weigh simulation
    number_of_assets = len(assets_returns_df.columns.tolist())
    random_array = np.random.rand(1,number_of_assets )
    random_array_df = pd.DataFrame(random_array, columns = assets_returns_df.columns.tolist())
    random_weight_df = random_array_df/random_array_df.values.sum()
    return random_weight_df

def portfolio_expected_Return(random_weight_df,log_returns):
    assets_expected_returns = log_returns.mean()
    weited_expected_returns = assets_expected_returns * random_weight_df
    portfolio_expected_return_ = weited_expected_returns.values.sum()
    return 100*portfolio_expected_return_

def  portfolio_volatility(varcovar,w):
    transpose_w = w.T
    σp = np.sqrt(np.matmul(np.matmul(w,varcovar),w.T))
    return 100*σp[0][0]

```

```python
def efficient_frontiere_plot(portfolio_trails_simulation_df):
    display(portfolio_trails_simulation_df)
    #fig, ax = plt.subplots()
    portfolio_trails_simulation_df.plot(x='σp', y='E_rp', kind='scatter', figsize=(10, 6));
    plt.xlabel('Expected Volatility')
    plt.ylabel('Expected Return')
    plt.title('Random portfolios Efficient Frontier')

#efficient_frontiere_plot(portfolio_trails_simulation_df)
```

```python
def generate_excess_return(log_returns_df):
    𝝁 = log_returns_df.mean()
    𝝁_list = []
    𝝁_list = 𝝁.values
    X_df= pd.DataFrame(data=log_returns_df[0:0:],
                       index=log_returns_df.index.to_list(), #[i for i in range(log_returns_df.shape[0])],
                       columns=log_returns_df.columns.tolist())

    assets_list = log_returns_df.columns.tolist()
    for index in range(len(assets_list)):
            X_df[assets_list[index]] =  log_returns_df[assets_list[index]].values - 𝝁_list[index]
    return X_df

#Portfolio Statistics
def portfolio_arihtmetics(log_returns_df,index_adj_close_price_df):
    return pd.DataFrame({'mu expected_return':log_returns_df.mean(),
                        'variance':log_returns_df.var(),
                        'Sigmas(volatilities)':log_returns_df.std(),
                        'modifiy shape(Er)/𝝈':log_returns_df.mean()/log_returns_df.std(),
                        'initial price':index_adj_close_price_df.iloc[0]}).transpose()


def excess_return_varcovar(X_df):
    return X_df.cov()

def get_uncorrelated_assets_index_adj_close_price_df(index_adj_close_price_df, uncorrelated_assets_list):
    return index_adj_close_price_df[uncorrelated_assets_list]

def uncorrelated_assets_arithmetics_summary(index_adj_close_price_df, log_returns,
                                            most_diversify_portfolio_assets_list, top_modify_shape_ratio):

    #uncorrelated_assets_index_adj_close_price_df = get_uncorrelated_assets_index_adj_close_price_df(index_adj_close_price_df,
    #                                                                selecting_uncorrelated_assets(log_returns,threshold))

    uncorrelated_assets_index_adj_close_price_df = get_uncorrelated_assets_index_adj_close_price_df(index_adj_close_price_df,
                                                                     most_diversify_portfolio_assets_list)

    #uncorrelated_assets_log_returns_df = uncorrelated_assets_returns_log_returns_df(log_returns, selecting_uncorrelated_assets(log_returns,threshold))
    uncorrelated_assets_log_returns_df = uncorrelated_assets_returns_log_returns_df(log_returns, most_diversify_portfolio_assets_list)

    portfolio_arihtmetics_df = portfolio_arihtmetics(uncorrelated_assets_log_returns_df, uncorrelated_assets_index_adj_close_price_df)
    portfolio_arihtmetics_df_T = portfolio_arihtmetics_df.transpose()
    portfolio_arihtmetics_df_T = portfolio_arihtmetics_df_T.sort_values(by='modifiy shape(Er)/𝝈',ascending=False)
    modify_shape_ratio_sort_assets_ticker_list = portfolio_arihtmetics_df_T.index.tolist()
    uncorrelated_assets_index_adj_close_price_by_return_df = get_uncorrelated_assets_index_adj_close_price_df(index_adj_close_price_df,
                                                                                                         modify_shape_ratio_sort_assets_ticker_list)

    return portfolio_arihtmetics_df_T, uncorrelated_assets_index_adj_close_price_by_return_df
#--------------------------------------------

```

```python
#------execution time approx 5mn
def uncorelated_portfolio_trals_simulation(log_returns, most_diversify_portfolio_assets_list, trial):

    σp_list = []
    E_rp_list = []
    random_weight_array_df_rows_list = []
    excess_return_df = generate_excess_return(log_returns[most_diversify_portfolio_assets_list])

    for i in  range(0, trial):
        random_weight_array_df = portfolio_random_weight_array_df(uncorrelated_assets_returns_log_returns_df(log_returns,
                                                                                        most_diversify_portfolio_assets_list))

        random_weight_array_df_rows_list.append(random_weight_array_df)

        E_rp_list.append(portfolio_expected_Return(random_weight_array_df,uncorrelated_assets_returns_log_returns_df(log_returns,
                                                                                               most_diversify_portfolio_assets_list)))

        σp_list.append(portfolio_volatility(excess_return_varcovar(excess_return_df),random_weight_array_df))

    uncorelated_portfolio_trails_simulation_df =  pd.DataFrame({'σp':σp_list,'E_rp':E_rp_list}, index=[i for i in range(0,trial)])
    σp = uncorelated_portfolio_trails_simulation_df['σp']
    E_rp = uncorelated_portfolio_trails_simulation_df['E_rp']
    sharpes_rat = E_rp/σp
    uncorelated_portfolio_trails_simulation_sharpes_ratio_df = pd.DataFrame({'σp':σp,'E_rp':E_rp,'sharpes_ratio':sharpes_rat})

    random_weight_array_all_rows_df = pd.concat(random_weight_array_df_rows_list, axis=0,ignore_index=True)
    uncorelated_weighted_portfolio_trails_simulation_df = uncorelated_portfolio_trails_simulation_sharpes_ratio_df.merge(random_weight_array_all_rows_df,
                                                                                                                         left_index=True, right_index=True)

    return uncorelated_portfolio_trails_simulation_df,uncorelated_portfolio_trails_simulation_sharpes_ratio_df, \
                                        random_weight_array_all_rows_df,uncorelated_weighted_portfolio_trails_simulation_df

```

```python

uncorelated_portfolio_trails_simulation_df,uncorelated_portfolio_trails_simulation_sharpes_ratio_df, random_weight_array_all_rows_df, \
                                   uncorelated_weighted_portfolio_trails_simulation_df = \
                                   uncorelated_portfolio_trals_simulation(log_returns, most_important_assets_list, 10000)


X_df =generate_excess_return(most_diversify_portfolio_assets_log_returns_df)
Excess_return_varcovar = excess_return_varcovar(X_df)
display(Excess_return_varcovar)
efficient_frontiere_plot(uncorelated_weighted_portfolio_trails_simulation_df)
```

```
          AGI       ATS       BMO        BN       BTO       CIX       CNQ  \
AGI  0.000922  0.000066  0.000054  0.000106  0.000042  0.000170  0.000091   
ATS  0.000066  0.000568  0.000157  0.000192  0.000203  0.000073  0.000211   
BMO  0.000054  0.000157  0.000350  0.000318  0.000383  0.000156  0.000384   
BN   0.000106  0.000192  0.000318  0.000508  0.000412  0.000187  0.000377   
BTO  0.000042  0.000203  0.000383  0.000412  0.000859  0.000255  0.000451   
CIX  0.000170  0.000073  0.000156  0.000187  0.000255  0.001355  0.000207   
CNQ  0.000091  0.000211  0.000384  0.000377  0.000451  0.000207  0.000955   
CWB  0.000062  0.000097  0.000129  0.000165  0.000173  0.000097  0.000169   
DOL  0.000095  0.000111  0.000177  0.000208  0.000216  0.000122  0.000236   
DOO  0.000094  0.000108  0.000172  0.000203  0.000210  0.000120  0.000227   
ENB  0.000083  0.000133  0.000247  0.000263  0.000300  0.000137  0.000387   
IGM  0.000099  0.000130  0.000181  0.000260  0.000212  0.000125  0.000214   
PEY  0.000085  0.000101  0.000207  0.000243  0.000301  0.000148  0.000271   
SIL  0.000645  0.000120  0.000120  0.000158  0.000124  0.000182  0.000186   
SLF  0.000079  0.000137  0.000251  0.000275  0.000311  0.000149  0.000324   
TD   0.000057  0.000142  0.000269  0.000277  0.000330  0.000132  0.000332   
WFG  0.000170  0.000192  0.000290  0.000333  0.000377  0.000165  0.000375   

          CWB       DOL       DOO       ENB       IGM       PEY       SIL  \
AGI  0.000062  0.000095  0.000094  0.000083  0.000099  0.000085  0.000645   
ATS  0.000097  0.000111  0.000108  0.000133  0.000130  0.000101  0.000120   
BMO  0.000129  0.000177  0.000172  0.000247  0.000181  0.000207  0.000120   
BN   0.000165  0.000208  0.000203  0.000263  0.000260  0.000243  0.000158   
BTO  0.000173  0.000216  0.000210  0.000300  0.000212  0.000301  0.000124   
CIX  0.000097  0.000122  0.000120  0.000137  0.000125  0.000148  0.000182   
CNQ  0.000169  0.000236  0.000227  0.000387  0.000214  0.000271  0.000186   
CWB  0.000116  0.000098  0.000093  0.000113  0.000156  0.000094  0.000099   
DOL  0.000098  0.000147  0.000141  0.000155  0.000152  0.000137  0.000129   
DOO  0.000093  0.000141  0.000144  0.000149  0.000144  0.000137  0.000126   
ENB  0.000113  0.000155  0.000149  0.000307  0.000150  0.000179  0.000129   
IGM  0.000156  0.000152  0.000144  0.000150  0.000304  0.000138  0.000137   
PEY  0.000094  0.000137  0.000137  0.000179  0.000138  0.000216  0.000107   
SIL  0.000099  0.000129  0.000126  0.000129  0.000137  0.000107  0.000681   
SLF  0.000117  0.000162  0.000158  0.000208  0.000169  0.000186  0.000123   
TD   0.000111  0.000159  0.000154  0.000219  0.000157  0.000191  0.000106   
WFG  0.000152  0.000182  0.000180  0.000243  0.000202  0.000220  0.000214   

          SLF        TD       WFG  
AGI  0.000079  0.000057  0.000170  
ATS  0.000137  0.000142  0.000192  
BMO  0.000251  0.000269  0.000290  
BN   0.000275  0.000277  0.000333  
BTO  0.000311  0.000330  0.000377  
CIX  0.000149  0.000132  0.000165  
CNQ  0.000324  0.000332  0.000375  
CWB  0.000117  0.000111  0.000152  
DOL  0.000162  0.000159  0.000182  
DOO  0.000158  0.000154  0.000180  
ENB  0.000208  0.000219  0.000243  
IGM  0.000169  0.000157  0.000202  
PEY  0.000186  0.000191  0.000220  
SIL  0.000123  0.000106  0.000214  
SLF  0.000278  0.000223  0.000260  
TD   0.000223  0.000281  0.000258  
WFG  0.000260  0.000258  0.000809  
```
```
            σp      E_rp  sharpes_ratio       AGI       ATS       BMO  \
0     1.463924  0.049216       0.033619  0.061478  0.028326  0.071094   
1     1.456257  0.050297       0.034539  0.095308  0.051838  0.052191   
2     1.519654  0.050381       0.033153  0.043578  0.099466  0.025888   
3     1.447017  0.055641       0.038452  0.114172  0.100494  0.025847   
4     1.590675  0.052442       0.032969  0.026852  0.037283  0.023290   
...        ...       ...            ...       ...       ...       ...   
9995  1.450366  0.041744       0.028781  0.021174  0.046835  0.024910   
9996  1.499805  0.049326       0.032888  0.030985  0.096667  0.077471   
9997  1.466001  0.053268       0.036335  0.077047  0.034728  0.060395   
9998  1.368392  0.046324       0.033853  0.074358  0.023944  0.005396   
9999  1.480896  0.051103       0.034508  0.064211  0.022347  0.059236   

            BN       BTO       CIX       CNQ       CWB       DOL       DOO  \
0     0.031965  0.040477  0.001287  0.097337  0.106726  0.082775  0.069108   
1     0.090941  0.042232  0.079520  0.032087  0.006965  0.065352  0.064341   
2     0.102048  0.060712  0.003116  0.070548  0.019659  0.064289  0.085615   
3     0.012265  0.012440  0.125598  0.011601  0.027697  0.046409  0.047930   
4     0.119195  0.077515  0.118204  0.071668  0.047571  0.004109  0.101898   
...        ...       ...       ...       ...       ...       ...       ...   
9995  0.028640  0.113835  0.070572  0.031758  0.000404  0.054368  0.109426   
9996  0.043811  0.086070  0.098646  0.093373  0.023148  0.083306  0.106571   
9997  0.092088  0.034346  0.014701  0.080867  0.105010  0.019764  0.073559   
9998  0.048464  0.042889  0.100645  0.024404  0.077556  0.044457  0.085507   
9999  0.106488  0.073483  0.030267  0.028149  0.072220  0.000783  0.036406   

           ENB       IGM       PEY       SIL       SLF        TD       WFG  
0     0.086733  0.056751  0.010179  0.098261  0.041318  0.016895  0.099291  
1     0.103519  0.051785  0.037692  0.024174  0.028138  0.104734  0.069183  
2     0.062783  0.048497  0.033337  0.001310  0.105518  0.072775  0.100860  
3     0.027334  0.119180  0.001083  0.066607  0.100469  0.045745  0.115127  
4     0.096280  0.009467  0.016165  0.006370  0.022246  0.111510  0.110377  
...        ...       ...       ...       ...       ...       ...       ...  
9995  0.076364  0.057101  0.113578  0.065853  0.097676  0.070423  0.017085  
9996  0.064479  0.006410  0.063945  0.015442  0.068136  0.016169  0.025370  
9997  0.046928  0.090525  0.029031  0.035272  0.059986  0.047293  0.098459  
9998  0.087827  0.101470  0.099017  0.075380  0.019033  0.081825  0.007829  
9999  0.042473  0.135150  0.101605  0.069185  0.049160  0.012325  0.096511  

[10000 rows x 20 columns]
```
```
<Figure size 1000x600 with 1 Axes>
```
```python
#--------------------------------------------------Efficient Frontiere Optimal Points-----------------------------------------------
# get data frame top1 portfolio
# selecting the optimal portfolios:portfolios with expected return higher or equal the minimun risky portfolio
#sort the optimal portfolio data frame by selected value:ascending=True
#return the data frame
#---------------------------------------------------------------------------------------------------------------------------------
def efficient_frontiere_selected_sharpe_ratio_portfolio_df(uncorrelated_weighted_portfolio_trails_simulation_df,selected_col):

    uncorrelated_weighted_portfolio_trails_simulation_sorted_df = uncorrelated_weited_portfolio_trails_simulation_df.sort_values(by='sharpes_ratio', ascending=False)
    uncorrelated_weighted_portfolio_trails_simulation_sorted_df = uncorrelated_weited_portfolio_trails_simulation_sorted_df.reset_index(drop=True)

    top1_sharpe_ratio_value = uncorrelated_weighted_portfolio_trails_simulation_sorted_df['sharpes_ratio'].values[0]
    top1_E_rp_value= uncorrelated_weighted_portfolio_trails_simulation_sorted_df['E_rp'].values[0]
    top1_σp_value = uncorrelated_weighted_portfolio_trails_simulation_sorted_df['σp'].values[0]

    # selecting the optimal portfolios:portfolios with expected return higher or equal the minimun risky portfolio
    if selected_col == 'sharpes_ratio':
        uncorrelated_weighted_portfolio_trails_simulation_selected_sharpes_ratio_optimal_portfolios_df = \
        uncorrelated_weighted_portfolio_trails_simulation_sorted_df[uncorrelated_weighted_portfolio_trails_simulation_sorted_df[selected_col] >= top1_sharpe_ratio_value]
    elif selected_col == 'E_rp':
        uncorrelated_weighted_portfolio_trails_simulation_selected_sharpes_ratio_optimal_portfolios_df = uncorrelated_weighted_portfolio_trails_simulation_sorted_df

    # sort the optimal portfolio data frame
    uncorrelated_weighted_portfolio_trails_simulation_selected_sharpes_ratio_optimal_portfolios_df = \
    uncorrelated_weighted_portfolio_trails_simulation_selected_sharpes_ratio_optimal_portfolios_df.sort_values(by='σp', ascending=True)
    uncorrelated_weighted_portfolio_trails_simulation_selected_sharpes_ratio_optimal_portfolios_df = \
    uncorrelated_weighted_portfolio_trails_simulation_selected_sharpes_ratio_optimal_portfolios_df.reset_index(drop=True)
    return uncorelated_portfolio_trails_simulation_selected_sharpes_ratio_optimal_portfolios_df

#----------------------------------------------------------------------------------------
def efficient_frontiere_optimal_sharpe_ratio_portfolios_model_points(uncorrelated_weighted_portfolio_trails_simulation_df,number_of_top_points = 35):
    #sort from maximum sharpe ratio and get top sharpe ratio portfolios
    portfolio_trails_simulation_sharpes_ratio_top_df = uncorrelated_weighted_portfolio_trails_simulation_df.sort_values(by='sharpes_ratio',
                                                                                                                        ascending=False)
    portfolio_trails_simulation_sharpes_ratio_top_df = portfolio_trails_simulation_sharpes_ratio_top_df.reset_index(drop=True)
    uncorelated_portfolio_trails_simulation_sharpes_ratio_top_df =portfolio_trails_simulation_sharpes_ratio_top_df.head(number_of_top_points)
    xpoints_list = []
    ypoints_list = []
    top_sharpe_ratio_value_points_list = []

    for portfolio_number in range(number_of_top_points):
        #top shape ratio
        top_sharpe_ratio_value_points_list.append(portfolio_trails_simulation_sharpes_ratio_top_df['sharpes_ratio'].values[portfolio_number])
        xpoints_list.append(portfolio_trails_simulation_sharpes_ratio_top_df['σp'].values[portfolio_number])
        ypoints_list.append(portfolio_trails_simulation_sharpes_ratio_top_df['E_rp'].values[portfolio_number])

    xpoints = np.array(xpoints_list)
    ypoints = np.array(ypoints_list)
    top_sharpe_ratio_value_points = np.array(top_sharpe_ratio_value_points_list)

    return xpoints, ypoints, top_sharpe_ratio_value_points

def get_maximun_return_portfolio(uncorrelated_weighted_portfolio_trails_simulation_df):

    portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df = uncorrelated_weighted_portfolio_trails_simulation_df.sort_values(by='E_rp',
                                                                                                                                ascending=False)
    portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df = portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df.reset_index(drop=True)

    max_E_rp_sharpe_ratio = portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df['sharpes_ratio'].values[0]
    max_E_rp = portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df['E_rp'].values[0]
    max_E_rp_σp = portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df['σp'].values[0]

    return max_E_rp_sharpe_ratio, max_E_rp, max_E_rp_σp

def get_maximun_risk_portfolio(uncorrelated_weighted_portfolio_trails_simulation_df):
    # here  the portfolios are sotrted from maximum risk
    portfolio_trals_simulation_sharpes_ratio_max_σp_E_rp_selecte_df = \
                                uncorrelated_weighted_portfolio_trails_simulation_df.sort_values(by='σp', ascending=False)
    portfolio_trals_simulation_sharpes_ratio_max_σp_E_rp_selecte_df = \
                                        portfolio_trals_simulation_sharpes_ratio_max_σp_E_rp_selecte_df.reset_index(drop=True)

    max_σp_E_rp_sharpe_ratio = portfolio_trals_simulation_sharpes_ratio_max_σp_E_rp_selecte_df['sharpes_ratio'].values[0]
    max_σp_E_rp = portfolio_trals_simulation_sharpes_ratio_max_σp_E_rp_selecte_df['E_rp'].values[0]
    max_σp = portfolio_trals_simulation_sharpes_ratio_max_σp_E_rp_selecte_df['σp'].values[0]

    return max_σp_E_rp_sharpe_ratio, max_σp_E_rp, max_σp

def get_minimum_risk_portfolio(uncorrelated_weighted_portfolio_trails_simulation_df):
    # here  the portfolios are sotrted from minimum risk
    portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df = \
                                            uncorrelated_weighted_portfolio_trails_simulation_df.sort_values(by='σp', ascending=True)
    portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df = portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df.reset_index(drop=True)
    minimun_σp_E_rp_sharpe_ratio = portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df['sharpes_ratio'].values[0]
    minimun_σp_E_rp = portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df['E_rp'].values[0]
    minimun_σp = portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df['σp'].values[0]
    return minimun_σp_E_rp_sharpe_ratio, minimun_σp_E_rp, minimun_σp

def get_maximum_sharpe_ratio(uncorrelated_weighted_portfolio_trails_simulation_df):
    #sort from maximum sharpe ratio and get top sharpe ratio portfolios
    portfolio_trails_simulation_sharpes_ratio_top_df = \
                                        uncorrelated_weighted_portfolio_trails_simulation_df.sort_values(by='sharpes_ratio', ascending=False)
    portfolio_trails_simulation_sharpes_ratio_top_df = portfolio_trails_simulation_sharpes_ratio_top_df.reset_index(drop=True)
    maximum_sharpe_ratio = portfolio_trails_simulation_sharpes_ratio_top_df['sharpes_ratio'].values[0]
    maximum_sharpe_ratio_σp_E_rp = portfolio_trails_simulation_sharpes_ratio_top_df['E_rp'].values[0]
    maximum_sharpe_ratio_σp = portfolio_trails_simulation_sharpes_ratio_top_df['σp'].values[0]
    return maximum_sharpe_ratio, maximum_sharpe_ratio_σp_E_rp, maximum_sharpe_ratio_σp


#-----------------------------------------------------------------------------------------------------------------------------------
def efficient_frontiere_optimal_portfolios_model_points(uncorrelated_weighted_portfolio_trails_simulation_df,number_of_top_points = 35):

    #number_of_top_points = 35
    #sort from maximum sharpe ratio and get top sharpe ratio portfolios
    portfolio_trails_simulation_sharpes_ratio_top_df = uncorrelated_weighted_portfolio_trails_simulation_df.sort_values(by='sharpes_ratio', ascending=False)
    portfolio_trails_simulation_sharpes_ratio_top_df = portfolio_trails_simulation_sharpes_ratio_top_df.reset_index(drop=True)
    uncorelated_portfolio_trails_simulation_sharpes_ratio_top_df =portfolio_trails_simulation_sharpes_ratio_top_df.head(number_of_top_points)

    # minimum risk portfolio: here  the portfolios are sotrted from minimum risk
    portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df = uncorrelated_weighted_portfolio_trails_simulation_df.sort_values(by='σp', ascending=True)
    portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df = portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df.reset_index(drop=True)
    portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df = portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df.head(number_of_top_points)

    minimun_σp_E_rp_sharpe_ratio = portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df['sharpes_ratio'].values[0]
    minimun_σp_E_rp = portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df['E_rp'].values[0]
    minimun_σp = portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df['σp'].values[0]

    # maximun return portfolio
    portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df = uncorrelated_weighted_portfolio_trails_simulation_df.sort_values(by='E_rp', ascending=False)
    portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df = portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df.reset_index(drop=True)
    portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df = portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df.head(number_of_top_points)

    max_E_rp_sharpe_ratio = portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df['sharpes_ratio'].values[0]
    max_E_rp = portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df['E_rp'].values[0]
    max_E_rp_σp = portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df['σp'].values[0]

    # maximun risk portfolio: here  the portfolios are sotrted from maximum risk
    portfolio_trals_simulation_sharpes_ratio_max_σp_E_rp_selecte_df = uncorrelated_weighted_portfolio_trails_simulation_df.sort_values(by='σp', ascending=False)
    portfolio_trals_simulation_sharpes_ratio_max_σp_E_rp_selecte_df = portfolio_trals_simulation_sharpes_ratio_max_σp_E_rp_selecte_df.reset_index(drop=True)
    portfolio_trals_simulation_sharpes_ratio_max_σp_E_rp_selecte_df = portfolio_trals_simulation_sharpes_ratio_max_σp_E_rp_selecte_df.head(number_of_top_points)



    xpoints_list = []
    ypoints_list = []
    top_sharpe_ratio_value_points_list = []

    for portfolio_number in range(number_of_top_points):

        #top shape ratio
        top_sharpe_ratio_value_points_list.append(portfolio_trails_simulation_sharpes_ratio_top_df['sharpes_ratio'].values[portfolio_number])
        xpoints_list.append(portfolio_trails_simulation_sharpes_ratio_top_df['σp'].values[portfolio_number])
        ypoints_list.append(portfolio_trails_simulation_sharpes_ratio_top_df['E_rp'].values[portfolio_number])

        # minimum risk portfolio:
        top_sharpe_ratio_value_points_list.append(portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df['sharpes_ratio'].values[portfolio_number])
        xpoints_list.append(portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df['σp'].values[portfolio_number])
        ypoints_list.append(portfolio_trails_simulation_sharpes_ratio_minun_σp_E_rp_df['E_rp'].values[portfolio_number])

        # maximun return portfolio
        top_sharpe_ratio_value_points_list.append(portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df['sharpes_ratio'].values[portfolio_number])
        xpoints_list.append(portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df['σp'].values[portfolio_number])
        ypoints_list.append(portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df['E_rp'].values[portfolio_number])


    xpoints = np.array(xpoints_list)
    ypoints = np.array(ypoints_list)
    top_sharpe_ratio_value_points = np.array(top_sharpe_ratio_value_points_list)


    return xpoints.clip(minimun_σp,max_E_rp_σp),ypoints.clip(minimun_σp_E_rp,max_E_rp), \
                    top_sharpe_ratio_value_points.clip(minimun_σp_E_rp_sharpe_ratio,max_E_rp_sharpe_ratio)

def get_maximun_minimum_points(df):

    # maximun return portfolio
    max_df = df.sort_values(by='E_rp', ascending=False)
    max_df = max_df.reset_index(drop=True)
    max_df = max_df.head(1)
    max_E_rp_σp = max_df['σp']
    max_E_rp    = max_df['E_rp']
    max_E_rp_sharpe_ratio = max_df['sharpes_ratio']

    # minimum return portfolio
    min_df = df.sort_values(by='E_rp', ascending=True)
    min_df = min_df.reset_index(drop=True)
    min_df = min_df.head(1)
    minimun_σp = min_df['σp']
    minimun_σp_E_rp    = min_df['E_rp']
    minimun_σp_E_rp_sharpe_ratio = min_df['sharpes_ratio']

    return max_E_rp_σp, max_E_rp, max_E_rp_sharpe_ratio, minimun_σp, minimun_σp_E_rp, minimun_σp_E_rp_sharpe_ratio

#-----------------------------------------Efficient Frontiere Model Plotting-----------------------------------------------------------------------------------
#call efficient_frontiere_optimal_portfolios_df to include sharpe ration dataframe to the trails protfolios dataframe
#and select the optimal portfolios
#prepare data for plotting and create the scatter plot
#include sharpe ration dataframe to the trails protfolios dataframe
#select the optimal portfolios(portfolios with expected return higher or equal to the minimumal risk portfolio)
#sorted by sharpe ration efficient_frontiere_selected_sharpe_ratio_portfolio_df
#------------------------------------------------------------------------------------------------------------------------------
def plot_fitted_curve(uncorrelated_weighted_portfolio_trails_simulation_df,fig, ax, label, marker, color ):
    #points plotting

    xpoints,ypoints,top_sharpe_ratio_value_points = \
                efficient_frontiere_optimal_portfolios_model_points(uncorrelated_weighted_portfolio_trails_simulation_df,7)

    row, col = uncorelated_weighted_portfolio_trails_simulation_df.shape
    #--model definition---
    mymodel = np.poly1d(np.polyfit(xpoints, ypoints,2))
    popt = np.polyfit(xpoints, ypoints,2)
    a, b, c = popt
    poly_d2_form = str('y =%.5f * x^2 + %.5f * x + %.5f' % (a, b, c))
    display(np.polyfit(xpoints, ypoints,2))
    myline = np.linspace(xpoints.min(), xpoints.max(), row)

    # optimal portfolios plotting
    ypred = mymodel(myline)
    ax.plot(xpoints,ypoints,'*',color='red',label='Optimal portfolios')
    ax.plot(myline, mymodel(myline),'.',color="blue",label=label + ':\n'+poly_d2_form)
    print(r2_score(ypoints, mymodel(xpoints)))

def plot_random_portfolios(uncorrelated_weighted_portfolio_trails_simulation_df, fig, ax, colorbar = 'yes'):

    #random portfolio plotting
    optimal_portfolios_df = uncorrelated_weighted_portfolio_trails_simulation_df
    sharpes_ratio_optimal_portfolios_σp_col = optimal_portfolios_df['σp']
    sharpes_ratio_optimal_portfolios_E_rp_col = optimal_portfolios_df['E_rp']
    optimal_portfolios_sharpes_ratio_col = optimal_portfolios_df['sharpes_ratio']

    scplt = ax.scatter(sharpes_ratio_optimal_portfolios_σp_col, sharpes_ratio_optimal_portfolios_E_rp_col, marker="o",
                       c=optimal_portfolios_sharpes_ratio_col, cmap="viridis",label='Random Portfolios')
    if colorbar == 'yes':
        cb = fig.colorbar(scplt, ax=ax, label='Sharpe Ratio')
    ax.set_title("Towards an Efficient Frontier Model - Random portfolios Efficient Frontier")


def plot_fitted_curve_and_random_portfolios(uncorrelated_weighted_portfolio_trails_simulation_df):
    fig, ax =plt.subplots(figsize=(12, 5))
    plot_fitted_curve(uncorrelated_weighted_portfolio_trails_simulation_df,fig, ax, label='Model to Approximate', marker= '*', color='red')
    plot_random_portfolios(uncorrelated_weighted_portfolio_trails_simulation_df, fig, ax)
    ax.legend(prop = { "size": 8 })

#plot_fitted_curve_and_random_portfolios(uncorelated_weighted_portfolio_trails_simulation_df)
```

```python
#----------------------------------------------------------------------------------
# minimum risk portfolio: here  the portfolios are sotrted from minimum risk
#-----------------------------------------------------------------------------------
def portfolio_strategy_minimum_risk(uncorelated_weighted_portfolio_trails_simulation_df,number_of_top_points):

    portfolio_trails_simulation_minimum_risk_σp_E_rp_df = uncorelated_weighted_portfolio_trails_simulation_df.sort_values(
                                                        by='σp', ascending=True)
    portfolio_trails_simulation_minimum_risk_σp_E_rp_df = portfolio_trails_simulation_minimum_risk_σp_E_rp_df.reset_index(drop=True)
    portfolio_trails_simulation_minimum_risk_σp_E_rp_df = portfolio_trails_simulation_minimum_risk_σp_E_rp_df.head(number_of_top_points)

    portfolio_weight_df = portfolio_trails_simulation_minimum_risk_σp_E_rp_df[most_diversify_portfolio_assets_list]

    portfolio_weight_df = portfolio_weight_df*100

    portfolio_weight_df1 = portfolio_weight_df.head(1)
    portfolio_weight =portfolio_weight_df1.columns.values.tolist()
    asset_stickers = portfolio_weight_df1.iloc[0].tolist()
    portfolio_investment_strategy_df = pd.DataFrame({'Portfolio Weight':portfolio_weight,'Asset Stickers':asset_stickers})
    portfolio_investment_strategy_df = portfolio_investment_strategy_df.sort_values(by='Asset Stickers',ascending=True)

    portfolio_investment_strategy_Trans_df = portfolio_investment_strategy_df.transpose()
    strategy_Weight = portfolio_investment_strategy_df['Portfolio Weight']
    strategy_Stickers = portfolio_investment_strategy_df['Asset Stickers']

    return strategy_Weight, strategy_Stickers, portfolio_trails_simulation_minimum_risk_σp_E_rp_df

#----------------------------------------------------------------------------
#maximun risk portfolio: here  the portfolios are sotrted from minimum risk
#----------------------------------------------------------------------------
def portfolio_strategy_maximun_risk(uncorelated_weighted_portfolio_trails_simulation_df,number_of_top_points):
    #log_returns,threshold
    portfolio_trails_simulation_max_risk_σp_E_rp_df = uncorelated_weighted_portfolio_trails_simulation_df.sort_values(
                                                        by='σp', ascending=False)
    portfolio_trails_simulation_max_risk_σp_E_rp_df = portfolio_trails_simulation_max_risk_σp_E_rp_df.reset_index(drop=True)
    portfolio_trails_simulation_max_risk_σp_E_rp_df = portfolio_trails_simulation_max_risk_σp_E_rp_df.head(number_of_top_points)
    portfolio_weight_df = portfolio_trails_simulation_max_risk_σp_E_rp_df[most_diversify_portfolio_assets_list]
    portfolio_weight_df = portfolio_weight_df*100

    portfolio_weight_df1 = portfolio_weight_df.head(1)
    portfolio_weight =portfolio_weight_df1.columns.values.tolist()
    asset_stickers = portfolio_weight_df1.iloc[0].tolist()
    portfolio_investment_strategy_df = pd.DataFrame({'Portfolio Weight':portfolio_weight,'Asset Stickers':asset_stickers})
    portfolio_investment_strategy_df = portfolio_investment_strategy_df.sort_values(by='Asset Stickers',ascending=True)

    portfolio_investment_strategy_Trans_df = portfolio_investment_strategy_df.transpose()
    #display(portfolio_investment_strategy_Trans_df)
    strategy_Weight = portfolio_investment_strategy_df['Portfolio Weight']
    strategy_Stickers = portfolio_investment_strategy_df['Asset Stickers']

    return strategy_Weight, strategy_Stickers, portfolio_trails_simulation_max_risk_σp_E_rp_df

#---------------------------------
# maximun return portfolio
#--------------------------------
def portfolio_strategy_maximun_return(uncorelated_weighted_portfolio_trails_simulation_df,number_of_top_points):

    portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df = uncorelated_weighted_portfolio_trails_simulation_df.sort_values(
                                                        by='E_rp', ascending=False)
    portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df = portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df.reset_index(drop=True)
    portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df = portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df.head(number_of_top_points)

    portfolio_weight_df = portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df[most_diversify_portfolio_assets_list]

    portfolio_weight_df = portfolio_weight_df*100

    portfolio_weight_df1 = portfolio_weight_df.head(1)
    portfolio_weight =portfolio_weight_df1.columns.values.tolist()
    asset_stickers = portfolio_weight_df1.iloc[0].tolist()
    portfolio_investment_strategy_df = pd.DataFrame({'Portfolio Weight':portfolio_weight,'Asset Stickers':asset_stickers})
    portfolio_investment_strategy_df = portfolio_investment_strategy_df.sort_values(by='Asset Stickers',ascending=True)

    portfolio_investment_strategy_Trans_df = portfolio_investment_strategy_df.transpose()
    #display(portfolio_investment_strategy_Trans_df)
    strategy_Weight = portfolio_investment_strategy_df['Portfolio Weight']
    strategy_Stickers = portfolio_investment_strategy_df['Asset Stickers']

    return strategy_Weight, strategy_Stickers, portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df

#-------------------------------------------------------------------------
#sort from maximum sharpe ration and get top sharpe ratio portfolios
#-------------------------------------------------------------------------
def portfolio_strategy_top_sharpe_ratio(uncorelated_weighted_portfolio_trails_simulation_df,number_of_top_points):

    portfolio_trails_simulation_sharpes_ratio_top_df = uncorelated_weighted_portfolio_trails_simulation_df.sort_values(
                                                        by='sharpes_ratio', ascending=False)
    portfolio_trails_simulation_sharpes_ratio_top_df = portfolio_trails_simulation_sharpes_ratio_top_df.reset_index(drop=True)
    uncorrelated_portfolio_trails_simulation_sharpes_ratio_top_df = portfolio_trails_simulation_sharpes_ratio_top_df.head(number_of_top_points)

    portfolio_weight_df = uncorrelated_portfolio_trails_simulation_sharpes_ratio_top_df[most_diversify_portfolio_assets_list]
    portfolio_weight_df = portfolio_weight_df*100

    portfolio_weight_df1 = portfolio_weight_df.head(1)
    portfolio_weight =portfolio_weight_df1.columns.values.tolist()
    asset_stickers = portfolio_weight_df1.iloc[0].tolist()
    portfolio_investment_strategy_df = pd.DataFrame({'Portfolio Weight':portfolio_weight,'Asset Stickers':asset_stickers})
    portfolio_investment_strategy_df = portfolio_investment_strategy_df.sort_values(by='Asset Stickers',ascending=True)

    portfolio_investment_strategy_Trans_df = portfolio_investment_strategy_df.transpose()
    #display(portfolio_investment_strategy_Trans_df)
    strategy_Weight = portfolio_investment_strategy_df['Portfolio Weight']
    strategy_Stickers = portfolio_investment_strategy_df['Asset Stickers']

    return strategy_Weight, strategy_Stickers, uncorrelated_portfolio_trails_simulation_sharpes_ratio_top_df


#------------------------------------------------------------------------
#sort from maximum sharpe ratio and get top sharpe ratio portfolios
#------------------------------------------------------------------------
def portfolio_strategy_plotting(uncorelated_weighted_portfolio_trails_simulation_df,number_of_top_points):
    fig, ax =plt.subplots(2,2, figsize=(14, 10))

    strategy_Weight, strategy_Stickers,uncorrelated_portfolio_trails_simulation_sharpes_ratio_top_df = \
        portfolio_strategy_top_sharpe_ratio(uncorelated_weighted_portfolio_trails_simulation_df, number_of_top_points)

    bar_container= ax[0,0].barh(strategy_Weight,strategy_Stickers)
    # setting label of y-axis
    ax[0,0].set_ylabel("Asset Stickers")
    # setting label of x-axis
    #ax[0,0].set_xlabel("Portfolio Weight")
    ax[0,0].set_title("Maximum Sharpe Ratio Portfolio Assets Allocation")
    ax[0,0].bar_label(bar_container, fmt='{:,.0f}%')


    # maximun return portfolio
    strategy_Weight, strategy_Stickers,portfolio_trails_simulation_sharpes_ratio_max_σp_E_rp_df = \
        portfolio_strategy_maximun_return(uncorelated_weighted_portfolio_trails_simulation_df, number_of_top_points)
    bar_container= ax[0,1].barh(strategy_Weight,strategy_Stickers)

    # setting label of y-axis
    ax[0,1].set_ylabel("Asset Stickers")
    # setting label of x-axis
    #ax[0,1].set_xlabel("Portfolio Weight")
    ax[0,1].set_title("Maximun Return Portfolio Assets Allocation")
    ax[0,1].bar_label(bar_container, fmt='{:,.0f}%')

     # maximun risk portfolio: here  the portfolios are sotrted from minimum risk
    strategy_Weight, strategy_Stickers,portfolio_trails_simulation_max_risk_σp_E_rp_df = \
        portfolio_strategy_maximun_risk(uncorelated_weighted_portfolio_trails_simulation_df, number_of_top_points)

    bar_container= ax[1,0].barh(strategy_Weight,strategy_Stickers)
    # setting label of y-axis
    ax[1,0].set_ylabel("Asset Stickers")
    # setting label of x-axis
    #ax[1,0].set_xlabel("Portfolio Weight")
    ax[1,0].set_title("Maximun risk Portfolio Assets Allocation")
    ax[1,0].bar_label(bar_container, fmt='{:,.0f}%')

    # minimum risk portfolio: here  the portfolios are sotrted from maximun risk
    strategy_Weight, strategy_Stickers,portfolio_trails_simulation_minimum_risk_σp_E_rp_df = \
        portfolio_strategy_minimum_risk(uncorelated_weighted_portfolio_trails_simulation_df, number_of_top_points)

    bar_container= ax[1,1].barh(strategy_Weight,strategy_Stickers)
    # setting label of y-axis
    ax[1,1].set_ylabel("Asset Stickers")
    # setting label of x-axis
    #ax[1,1].set_xlabel("Portfolio Weight")
    ax[1,1].set_title("Minimum risk Portfolio Assets Allocation")
    ax[1,1].bar_label(bar_container, fmt='{:,.0f}%')

    plt.show()

plot_fitted_curve_and_random_portfolios(uncorelated_weighted_portfolio_trails_simulation_df)
portfolio_strategy_plotting(uncorelated_weighted_portfolio_trails_simulation_df,10)
```

```
array([-0.28822396,  0.89777456, -0.63458208])
```
```
0.9341898618645373

```
```
<Figure size 1200x500 with 2 Axes>
```
```
<Figure size 1400x1000 with 4 Axes>
```
  


## Efficient Frontier modelling using Machine Learning technics


#### Data Splitting / Model Selection


```python
# Data Splitting / Model Selection
def polynomial_degree2_model(uncorelated_weighted_portfolio_trails_simulation_df):
     # Load the data : original random portfolios data points
    xpoints, ypoints, original_random_sharpe_ratio = \
    efficient_frontiere_optimal_portfolios_model_points( uncorelated_weighted_portfolio_trails_simulation_df)

    #original_random_portfolios_df = pd.DataFrame({'σp':xpoints,'E_rp':ypoints,'sharpes_ratio':original_random_sharpe_ratio})

    # Build the model
    def model_poly_d2(x, a, b, c):
        return b * x**2 + a * x + c

    #Split tranning, validation and testing data
    x_train_poly_d2, x_test_poly_d2, y_train_poly_d2, y_test_poly_d2 = \
    train_test_split(xpoints, ypoints, test_size=0.3, random_state=42)
    x_model_validation_poly_d2, x_model_testing_poly_d2 = train_test_split(np.linspace(min(xpoints), max(xpoints),
                                                                                       len(xpoints)), test_size=0.3, random_state=42)

    # model traning to get paarameters
    popt_poly_d2, pcov_poly_d2 = curve_fit(model_poly_d2, x_train_poly_d2,y_train_poly_d2)
    a, b, c = popt_poly_d2

    poly_d2_form  = str('y =%.5f * x^2 + %.5f * x + %.5f' % (a, b, c))

    return x_train_poly_d2, x_test_poly_d2, y_train_poly_d2, y_test_poly_d2,popt_poly_d2, \
                pcov_poly_d2, x_model_validation_poly_d2, x_model_testing_poly_d2, model_poly_d2, poly_d2_form

#x_train_poly_d2, x_test_poly_d2, y_train_poly_d2, y_test_poly_d2,popt_poly_d2, \
#                pcov_poly_d2, x_model_validation_poly_d2, x_model_testing_poly_d2, model_poly_d2, poly_d2_form = \
#                                            polynomial_degree2_model(uncorelated_weighted_portfolio_trails_simulation_df)

```

```python
def polynomial_degree3_log_model(uncorelated_weighted_portfolio_trails_simulation_df):
    # Load the data : original random portfolios data points
    xpoints,ypoints,original_random_sharpe_ratio = efficient_frontiere_optimal_portfolios_model_points(uncorelated_weighted_portfolio_trails_simulation_df)

    #original_random_portfolios_df = pd.DataFrame({'σp':xpoints,'E_rp':ypoints,'sharpes_ratio':original_random_sharpe_ratio})

    # Build the model
    def model_poly_d3_log(x, a, b, c, d, e):
        return a * np.log(abs(b )* x) + c*x**3 +d*x**2 + e

    #Split tranning and testing data
    x_train_poly_d3_log, x_test_poly_d3_log, y_train_poly_d3_log, y_test_poly_d3_log = \
                                                    train_test_split(xpoints, ypoints, test_size=0.3, random_state=42)
    x_model_validation_poly_d3_log, x_model_testing_poly_d3_log = \
                                train_test_split(np.linspace(min(xpoints), max(xpoints), len(xpoints)), test_size=0.3, random_state=42)

    # model validation data
    #x_model_validation = np.linspace(min(x_train), max(x_train), number_of_top_points*3)

    # model traning to get poarameters
    popt_poly_d3_log, pcov_poly_d3_log = curve_fit(model_poly_d3_log, x_train_poly_d3_log,y_train_poly_d3_log)
    a, b, c, d, e = popt_poly_d3_log

    poly_d3_log_form = str('y =%.5f * np.log( %.5f*x) + %.5f * x**3 + %.5f * x + %.5f' % (a, b, c, d, e))


    return x_train_poly_d3_log, x_test_poly_d3_log, y_train_poly_d3_log, y_test_poly_d3_log,popt_poly_d3_log, pcov_poly_d3_log, \
                                x_model_validation_poly_d3_log, x_model_testing_poly_d3_log, model_poly_d3_log, poly_d3_log_form

#x_train_poly_d3_log, x_test_poly_d3_log, y_train_poly_d3_log, y_test_poly_d3_log,popt_poly_d3_log, pcov_poly_d3_log, \
#                                x_model_validation_poly_d3_log, x_model_testing_poly_d3_log, model_poly_d3_log, poly_d3_log_form = \
#                                                            polynomial_degree3_log_model(uncorelated_weighted_portfolio_trails_simulation_df)

```

```python
def polynomial_degree5_log_model(uncorelated_weighted_portfolio_trails_simulation_df):
     # Load the data : original random portfolios data points
    xpoints,ypoints,original_random_sharpe_ratio = efficient_frontiere_optimal_portfolios_model_points(uncorelated_weighted_portfolio_trails_simulation_df)

    # Build the model
    def model_poly_d5_log(x, a, b, c):
        return a*np.log(abs(b)*x) + c*x**5

    #Split tranning and testing data
    x_train_poly_d5_log, x_test_poly_d5_log, y_train_poly_d5_log, y_test_poly_d5_log = \
                                        train_test_split(xpoints, ypoints, test_size=0.3, random_state=42)
    x_model_validation_poly_d5_log, x_model_testing_poly_d5_log = \
                                train_test_split(np.linspace(min(xpoints), max(xpoints), len(xpoints)), test_size=0.3, random_state=42)

    popt_poly_d5_log, pcov_poly_d5_log = curve_fit(model_poly_d5_log, x_train_poly_d5_log,y_train_poly_d5_log)
    a, b, c = popt_poly_d5_log

    poly_d5_log_form = str('y =%.5f * np.log( %.5f*x) + %.5f * x**5' % (a, b, c))

    return x_train_poly_d5_log, x_test_poly_d5_log, y_train_poly_d5_log, y_test_poly_d5_log, popt_poly_d5_log, pcov_poly_d5_log, \
                                x_model_validation_poly_d5_log, x_model_testing_poly_d5_log, model_poly_d5_log, poly_d5_log_form

#x_train_poly_d5_log, x_test_poly_d5_log, y_train_poly_d5_log, y_test_poly_d5_log, popt_poly_d5_log, pcov_poly_d5_log, \
#                                x_model_validation_poly_d5_log, x_model_testing_poly_d5_log, model_poly_d5_log, poly_d5_log_form = \
#                polynomial_degree5_log_model(uncorelated_weighted_portfolio_trails_simulation_df)


```

```python
def models_plotting(x, y, uncorrelated_weighted_portfolio_trails_simulation_df, fig, ax, model_form):
    #------Random portfolio data plotting
    plot_random_portfolios(uncorrelated_weighted_portfolio_trails_simulation_df, fig, ax,'no')
    cspl = ax.scatter(x=x, y=y, c=y/x, cmap="viridis",label='Efficient Frontier:\n'+model_form)
    #-----------model to approximate
    plot_fitted_curve(uncorrelated_weighted_portfolio_trails_simulation_df,fig, ax, label='Fitted Curve', marker= '*', color='red')

    ax.legend(bbox_to_anchor=(0.72, 1.38), ncol=1, prop = { "size": 8})
    return cspl
```

```python
def dataframe_clipping(x_σp, y_E_rp, y_E_rp_pred ):

    clipped_df = pd.DataFrame({'σp':x_σp,'E_rp':y_E_rp,'y_E_rp_pred':y_E_rp_pred,'error':y_E_rp_pred - y_E_rp})
    clipped_df = clipped_df.sort_values(by='error',ascending=False)
    clipped_df['y_optimal_E_rp'] = np.where(clipped_df['E_rp'] <= clipped_df['y_E_rp_pred'], clipped_df['E_rp'],clipped_df['y_E_rp_pred'] )
    clipped_df['sharpes_ratio'] = clipped_df['y_optimal_E_rp']/clipped_df['σp']
    return clipped_df[clipped_df['error'] >= 0]

def model_uperBound_efficient_frontier( uncorrelated_weighted_portfolio_trails_simulation_df, model, model_popt,
                                       ax , mode_form, random_points = 0):

    optimal_portfolios_df = uncorrelated_weighted_portfolio_trails_simulation_df
    x_σp = uncorrelated_weighted_portfolio_trails_simulation_df['σp']
    y_E_rp = uncorelated_weighted_portfolio_trails_simulation_df['E_rp']
    row, col = uncorelated_weighted_portfolio_trails_simulation_df.shape

    #here the orriginal data frame is clipped to eliminate the upper bound Outlier
    y_E_rp_pred = model(x_σp, *model_popt)
    clipped_df = dataframe_clipping(x_σp, y_E_rp, y_E_rp_pred )

    xpoints,ypoints,top_sharpe_ratio_value_points = efficient_frontiere_optimal_portfolios_model_points(clipped_df,7)

    #------Random portfolio data plotting
    if random_points == 0:

        scplt = ax.scatter(clipped_df['σp'], clipped_df['E_rp'], marker="o", c=clipped_df['E_rp']/clipped_df['σp'],
                       cmap="viridis",label='Random Portfolios')

    else:
        xrandom_points,yrandom_points,random_sharpe_ratio_value_points = \
                        efficient_frontiere_optimal_sharpe_ratio_portfolios_model_points(clipped_df,random_points)
        scplt = ax.scatter(x=xrandom_points, y=yrandom_points, marker="o", c= random_sharpe_ratio_value_points,
                       cmap="viridis",label='Random Portfolios')


    #efficient frontier plotting
    x_model_σp = np.linspace(xpoints.min(), xpoints.max(), row)
    y_model_E_rp_pred = model(x_model_σp, *model_popt)
    cspl = ax.scatter(x=x_model_σp, y=y_model_E_rp_pred, marker="*", c= y_E_rp_pred/x_model_σp,
                      cmap="viridis",label='Efficient Frontier:\n'+mode_form)

    ax.set_title("Boundary Random portfolios Efficient Frontier")
    ax.legend(bbox_to_anchor=(0.72, 1.38), ncol=1, prop = { "size": 8})

    return scplt
```

#### Model Evaluation

##### Model validation

we validate the model(model parameters) on the training and the validation data.


```python

def evalute_model_parameters(uncorelated_weighted_portfolio_trails_simulation_df):

    #polynoial degree 2 model  b * x**2 + a * x + c
    x_train_poly_d2, x_test_poly_d2, y_train_poly_d2, y_test_poly_d2,popt_poly_d2, \
                pcov_poly_d2, x_model_validation_poly_d2, x_model_testing_poly_d2, model_poly_d2, poly_d2_form = \
                                                            polynomial_degree2_model(uncorelated_weighted_portfolio_trails_simulation_df)
    # model parameters
    a, b, c = popt_poly_d2
    #model prediction
    y_model_validation_pred_poly_d2 = model_poly_d2(x_model_validation_poly_d2, a, b, c)

    #polynomial degree 3 log model: a * np.log(b * x) + c*x**3 +d*x**2 + e
    x_train_poly_d3_log, x_test_poly_d3_log, y_train_poly_d3_log, y_test_poly_d3_log,popt_poly_d3_log, pcov_poly_d3_log, \
                                x_model_validation_poly_d3_log, x_model_testing_poly_d3_log, model_poly_d3_log, poly_d3_log_form = \
                                                            polynomial_degree3_log_model(uncorelated_weighted_portfolio_trails_simulation_df)
    # model parameters
    a, b, c, d, e = popt_poly_d3_log
    y_model_validation_pred_poly_d3_log = model_poly_d3_log(x_model_validation_poly_d3_log, a, abs(b), c, d, e)

    #polynomial degree 5 log model: a*np.log(b*x) + c*x**5
    x_train_poly_d5_log, x_test_poly_d5_log, y_train_poly_d5_log, y_test_poly_d5_log, popt_poly_d5_log, pcov_poly_d5_log, \
                                x_model_validation_poly_d5_log, x_model_testing_poly_d5_log, model_poly_d5_log, poly_d5_log_form = \
                polynomial_degree5_log_model(uncorelated_weighted_portfolio_trails_simulation_df)
    # model parameters
    a, b, c = popt_poly_d5_log
    y_model_validation_pred_poly_d5_log = model_poly_d5_log(x_model_validation_poly_d5_log, a,abs(b), c)


    return popt_poly_d2, popt_poly_d3_log, popt_poly_d5_log, x_model_validation_poly_d2,x_model_validation_poly_d3_log, \
           x_model_validation_poly_d5_log, y_train_poly_d2, y_model_validation_pred_poly_d2, y_train_poly_d3_log, \
           y_model_validation_pred_poly_d3_log, y_train_poly_d5_log, y_model_validation_pred_poly_d5_log, model_poly_d2, \
           model_poly_d3_log, model_poly_d5_log, poly_d2_form, poly_d3_log_form, poly_d5_log_form

#popt_poly_d2, popt_poly_d3_log, popt_poly_d5_log, x_model_validation_poly_d2,x_model_validation_poly_d3_log, \
#x_model_validation_poly_d5_log, y_train_poly_d2, y_model_validation_pred_poly_d2, y_train_poly_d3_log, \
#y_model_validation_pred_poly_d3_log, y_train_poly_d5_log, y_model_validation_pred_poly_d5_log, model_poly_d2, \
#model_poly_d3_log, model_poly_d5_log, poly_d2_form, poly_d3_log_form, poly_d5_log_form = \
#                                        evalute_model_parameters(uncorelated_weighted_portfolio_trails_simulation_df)
```

```python
def model_validation_plotting(uncorrelated_weighted_portfolio_trails_simulation_df):

    fig, ax =plt.subplots(2,2,figsize=(13, 13), constrained_layout=True)

    popt_poly_d2, popt_poly_d3_log, popt_poly_d5_log, x_model_validation_poly_d2,x_model_validation_poly_d3_log, \
    x_model_validation_poly_d5_log, y_train_poly_d2, y_model_validation_pred_poly_d2, y_train_poly_d3_log, \
    y_model_validation_pred_poly_d3_log, y_train_poly_d5_log, y_model_validation_pred_poly_d5_log, model_poly_d2, \
    model_poly_d3_log, model_poly_d5_log, poly_d2_form, poly_d3_log_form, poly_d5_log_form = \
                                        evalute_model_parameters(uncorelated_weighted_portfolio_trails_simulation_df)

    cspl1 = models_plotting(x_model_validation_poly_d2, y_model_validation_pred_poly_d2,
                           uncorrelated_weighted_portfolio_trails_simulation_df, fig, ax[0,0], poly_d2_form)
    cspl2 = models_plotting(x_model_validation_poly_d3_log, y_model_validation_pred_poly_d3_log,
                               uncorrelated_weighted_portfolio_trails_simulation_df, fig, ax[0,1],poly_d3_log_form)
    cspl = models_plotting(x_model_validation_poly_d5_log, y_model_validation_pred_poly_d5_log,
                               uncorrelated_weighted_portfolio_trails_simulation_df, fig, ax[1,0], poly_d5_log_form)
    cplt4 =  model_uperBound_efficient_frontier(uncorrelated_weighted_portfolio_trails_simulation_df, model_poly_d2,popt_poly_d2,
                                       ax[1,1], poly_d2_form)

    cb = fig.colorbar(cspl, ax=ax, label='Sharpe Ratio',orientation='horizontal',shrink=0.6)

model_validation_plotting(uncorelated_weighted_portfolio_trails_simulation_df)
```

```
array([-0.28822396,  0.89777456, -0.63458208])
```
```
0.9341898618645373

```
```
array([-0.28822396,  0.89777456, -0.63458208])
```
```
0.9341898618645373

```
```
array([-0.28822396,  0.89777456, -0.63458208])
```
```
0.9341898618645373

```
```
<Figure size 1300x1300 with 5 Axes>
```
#### Model fine-tuning


```python
def fine_tune_hyperparmeters(uncorelated_weighted_portfolio_trails_simulation_df):

     #polynoial degree 2 model  b * x**2 + a * x + c
    x_train_poly_d2, x_test_poly_d2, y_train_poly_d2, y_test_poly_d2,popt_poly_d2, \
                pcov_poly_d2, x_model_validation_poly_d2, x_model_testing_poly_d2, model_poly_d2, poly_d2_form = \
                                             polynomial_degree2_model(uncorelated_weighted_portfolio_trails_simulation_df)

    y_model_turning_pred_poly_d2 = model_poly_d2(x_model_validation_poly_d2, 0.075, -0.019, -0.007)
    poly_d2_form  = str('y =%.5f * x^2 + %.5f * x + %.5f' % (0.07, -0.016, -0.009))

    #polynomial degree 3 log model: a * np.log(b * x) + c*x**3 +d*x**2 + e
    x_train_poly_d3_log, x_test_poly_d3_log, y_train_poly_d3_log, y_test_poly_d3_log,popt_poly_d3_log, pcov_poly_d3_log, \
                                x_model_validation_poly_d3_log, x_model_testing_poly_d3_log, model_poly_d3_log, poly_d3_log_form = \
                                         polynomial_degree3_log_model(uncorelated_weighted_portfolio_trails_simulation_df)

    y_model_tuning_pred_poly_d3_log = model_poly_d3_log(x_model_validation_poly_d3_log,  0.256, 0.348, 0.00793, -0.060, 0.343)
    poly_d3_log_form = str('y =%.5f * np.log( %.5f*x) + %.5f * x**3 + %.5f * x + %.5f' % (0.256, 0.348, 0.00793, -0.060, 0.343))

    #polynomial degree 5 log model: a*np.log(b*x) + c*x**5
    x_train_poly_d5_log, x_test_poly_d5_log, y_train_poly_d5_log, y_test_poly_d5_log, popt_poly_d5_log, pcov_poly_d5_log, \
                                x_model_validation_poly_d5_log, x_model_testing_poly_d5_log, model_poly_d5_log, poly_d5_log_form = \
                polynomial_degree5_log_model(uncorelated_weighted_portfolio_trails_simulation_df)

    y_model_turning_pred_poly_d5_log = model_poly_d5_log(x_model_validation_poly_d5_log,  0.085, 1.44, -0.00058)
    poly_d5_log_form = str('y =%.5f * np.log( %.5f*x) + %.5f * x**5' % (0.085, 1.44, -0.00058))

    return model_poly_d2, model_poly_d3_log, model_poly_d5_log, popt_poly_d2, popt_poly_d3_log, popt_poly_d5_log, x_model_validation_poly_d2, \
           x_model_validation_poly_d3_log, x_model_validation_poly_d5_log, y_train_poly_d2, y_model_turning_pred_poly_d2, y_train_poly_d3_log, \
           y_model_tuning_pred_poly_d3_log, y_train_poly_d5_log, y_model_turning_pred_poly_d5_log, poly_d2_form, \
           poly_d3_log_form, poly_d5_log_form

#model_poly_d2, model_poly_d3_log, model_poly_d5_log, popt_poly_d2, popt_poly_d3_log, popt_poly_d5_log, x_model_validation_poly_d2, x_model_validation_poly_d3_log, \
#x_model_validation_poly_d5_log, y_train_poly_d2, y_model_tuning_pred_poly_d2, y_train_poly_d3_log, \
#y_model_tuning_pred_poly_d3_log, y_train_poly_d5_log, y_model_tuning_pred_poly_d5_log, poly_d2_form, \
#poly_d3_log_form, poly_d5_log_form= fine_tune_hyperparmeters(uncorelated_weighted_portfolio_trails_simulation_df)

```

```python
def model_tuning_plotting(uncorrelated_weighted_portfolio_trails_simulation_df):

    fig, ax =plt.subplots(2,2,figsize=(13, 13), constrained_layout=True)

    print(" Models Fine-tuning ")

    model_poly_d2, model_poly_d3_log, model_poly_d5_log, popt_poly_d2, popt_poly_d3_log, popt_poly_d5_log, x_model_validation_poly_d2, x_model_validation_poly_d3_log, \
    x_model_validation_poly_d5_log, y_train_poly_d2, y_model_tuning_pred_poly_d2, y_train_poly_d3_log, \
    y_model_tuning_pred_poly_d3_log, y_train_poly_d5_log, y_model_tuning_pred_poly_d5_log, poly_d2_form, \
    poly_d3_log_form, poly_d5_log_form= fine_tune_hyperparmeters(uncorelated_weighted_portfolio_trails_simulation_df)

    cspl1 = models_plotting(x_model_validation_poly_d2, y_model_tuning_pred_poly_d2,
                           uncorrelated_weighted_portfolio_trails_simulation_df, fig, ax[0,0], poly_d2_form)
    cspl2 = models_plotting(x_model_validation_poly_d3_log, y_model_tuning_pred_poly_d3_log,
                               uncorrelated_weighted_portfolio_trails_simulation_df, fig, ax[0,1], poly_d3_log_form)
    cspl = models_plotting(x_model_validation_poly_d5_log, y_model_tuning_pred_poly_d5_log,
                               uncorrelated_weighted_portfolio_trails_simulation_df, fig, ax[1,0], poly_d5_log_form)
    cplt4 = model_uperBound_efficient_frontier(uncorrelated_weighted_portfolio_trails_simulation_df,
                                       model_poly_d2,popt_poly_d2, ax[1,1], poly_d2_form,7000)

    cb = fig.colorbar(cspl, ax=ax, label='Sharpe Ratio',orientation='horizontal',shrink=0.6)

model_tuning_plotting(uncorelated_weighted_portfolio_trails_simulation_df)
```

```
 Models Fine-tuning 

```
```
array([-0.28822396,  0.89777456, -0.63458208])
```
```
0.9341898618645373

```
```
array([-0.28822396,  0.89777456, -0.63458208])
```
```
0.9341898618645373

```
```
array([-0.28822396,  0.89777456, -0.63458208])
```
```
0.9341898618645373

```
```
<Figure size 1300x1300 with 5 Axes>
```
#### Model Testing


```python
def test_the_model(uncorelated_weighted_portfolio_trails_simulation_df):
     #polynoial degree 2 model  b * x**2 + a * x + c
    x_train_poly_d2, x_test_poly_d2, y_train_poly_d2, y_test_poly_d2,popt_poly_d2, \
                pcov_poly_d2, x_model_validation_poly_d2, x_model_testing_poly_d2, model_poly_d2, poly_d2_form = \
                                             polynomial_degree2_model(uncorelated_weighted_portfolio_trails_simulation_df)

    y_model_test_pred_poly_d2 = model_poly_d2(x_test_poly_d2, 0.07, -0.016, -0.009)
    poly_d2_form  = str('y =%.5f * x^2 + %.5f * x + %.5f' % (0.07, -0.016, -0.009))

    #polynomial degree 3 log model: a * np.log(b * x) + c*x**3 +d*x**2 + e
    x_train_poly_d3_log, x_test_poly_d3_log, y_train_poly_d3_log, y_test_poly_d3_log,popt_poly_d3_log, pcov_poly_d3_log, \
                                x_model_validation_poly_d3_log, x_model_testing_poly_d3_log, model_poly_d3_log, poly_d3_log_form = \
                                         polynomial_degree3_log_model(uncorelated_weighted_portfolio_trails_simulation_df)

    y_model_test_pred_poly_d3_log = model_poly_d3_log(x_test_poly_d3_log,  0.256, 0.348, 0.00793, -0.060, 0.343)
    poly_d3_log_form = str('y =%.5f * np.log( %.5f*x) + %.5f * x**3 + %.5f * x + %.5f' % (0.256, 0.348, 0.00793, -0.060, 0.343))

    #polynomial degree 5 log model: a*np.log(b*x) + c*x**5
    x_train_poly_d5_log, x_test_poly_d5_log, y_train_poly_d5_log, y_test_poly_d5_log, popt_poly_d5_log, pcov_poly_d5_log, \
                                x_model_validation_poly_d5_log, x_model_testing_poly_d5_log, model_poly_d5_log, poly_d5_log_form = \
                polynomial_degree5_log_model(uncorelated_weighted_portfolio_trails_simulation_df)

    y_model_test_pred_poly_d5_log = model_poly_d5_log(x_test_poly_d5_log,  0.085, 1.44, -0.00058)
    poly_d5_log_form = str('y =%.5f * np.log( %.5f*x) + %.5f * x**5' % (0.085, 1.44, -0.00058))

    return model_poly_d2, model_poly_d3_log, model_poly_d5_log, popt_poly_d2, popt_poly_d3_log, popt_poly_d5_log, x_test_poly_d2, x_test_poly_d3_log, x_test_poly_d5_log, \
           y_test_poly_d2, y_model_test_pred_poly_d2,y_test_poly_d3_log, y_model_test_pred_poly_d3_log, y_test_poly_d5_log, \
           y_model_test_pred_poly_d5_log, poly_d2_form, poly_d3_log_form, poly_d5_log_form

model_poly_d2, model_poly_d3_log, model_poly_d5_log, popt_poly_d2, popt_poly_d3_log, popt_poly_d5_log, x_test_poly_d2, x_test_poly_d3_log, x_test_poly_d5_log, \
y_test_poly_d2, y_model_test_pred_poly_d2,y_test_poly_d3_log, y_model_test_pred_poly_d3_log, y_test_poly_d5_log, \
y_model_test_pred_poly_d5_log, poly_d2_form, poly_d3_log_form, poly_d5_log_form= \
                test_the_model(uncorelated_weighted_portfolio_trails_simulation_df)
```

```python
def model_testing_plotting(uncorrelated_weighted_portfolio_trails_simulation_df):

    fig, ax =plt.subplots(2,2,figsize=(13, 13), constrained_layout=True)

    print(" Model Testing ")
    model_poly_d2, model_poly_d3_log, model_poly_d5_log, popt_poly_d2, popt_poly_d3_log, popt_poly_d5_log, x_test_poly_d2, x_test_poly_d3_log, x_test_poly_d5_log, \
    y_test_poly_d2, y_model_test_pred_poly_d2,y_test_poly_d3_log, y_model_test_pred_poly_d3_log, y_test_poly_d5_log, \
    y_model_test_pred_poly_d5_log, poly_d2_form, poly_d3_log_form, poly_d5_log_form= \
                test_the_model(uncorelated_weighted_portfolio_trails_simulation_df)

    cspl1 = models_plotting(x_test_poly_d2, y_model_test_pred_poly_d2,
                           uncorrelated_weighted_portfolio_trails_simulation_df, fig, ax[0,0], poly_d2_form)
    cspl2 = models_plotting(x_test_poly_d3_log, y_model_test_pred_poly_d3_log,
                               uncorrelated_weighted_portfolio_trails_simulation_df, fig, ax[0,1], poly_d3_log_form)
    cspl = models_plotting(x_test_poly_d5_log, y_model_test_pred_poly_d5_log,
                               uncorrelated_weighted_portfolio_trails_simulation_df, fig, ax[1,0], poly_d5_log_form)
    cplt4 = model_uperBound_efficient_frontier(uncorrelated_weighted_portfolio_trails_simulation_df, \
                                        model_poly_d2,popt_poly_d2, ax[1,1], poly_d2_form)

    cb = fig.colorbar(cspl, ax=ax, label='Sharpe Ratio',orientation='horizontal',shrink=0.6)

model_testing_plotting(uncorelated_weighted_portfolio_trails_simulation_df)
```

```
 Model Testing 

```
```
array([-0.28822396,  0.89777456, -0.63458208])
```
```
0.9341898618645373

```
```
array([-0.28822396,  0.89777456, -0.63458208])
```
```
0.9341898618645373

```
```
array([-0.28822396,  0.89777456, -0.63458208])
```
```
0.9341898618645373

```
```
<Figure size 1300x1300 with 5 Axes>
```
#### Goodness of Fit Statistics


```python

def error_metrics_statistics(y_true_0, y_pred_0,y_true_1, y_pred_1,y_true_2, y_pred_2,  poly_d2_form, poly_d3_log_form, poly_d5_log_form ):

    display('Poly_d2 : '+poly_d2_form)
    display('Poly_d3_log: '+poly_d3_log_form)
    display('Poly_d5_log: '+poly_d5_log_form)

    error_metrics_table = [['Type Error', 'Poly_d2 Error', 'Poly_d3_log Error','Poly_d5_log Error'],
         ['Mean Absolute Error(MAE)', mean_absolute_error(y_true_0, y_pred_0),mean_absolute_error(y_true_1, y_pred_1),mean_absolute_error(y_true_2, y_pred_2)],
         ['Mean Absolute Percentage Error(MAPE)', mean_absolute_percentage_error(y_true_0, y_pred_0),mean_absolute_percentage_error(y_true_1, y_pred_1),mean_absolute_percentage_error(y_true_2, y_pred_2)],
         ['Neg.Mean Squared Error(RMSE)', -mean_squared_error(y_true_0, y_pred_0),-mean_squared_error(y_true_1, y_pred_1),-mean_squared_error(y_true_2, y_pred_2)],
         ['R-squared score', r2_score(y_true_0, y_pred_0),r2_score(y_true_1, y_pred_1),r2_score(y_true_2, y_pred_2)],
         ['Mean Squared Error(MSE)',mean_squared_error(y_true_0, y_pred_0),mean_squared_error(y_true_1, y_pred_1),mean_squared_error(y_true_2, y_pred_2)],
         ['Mean Squared Log Error(MSLE)', mean_squared_log_error(y_true_0, y_pred_0),mean_squared_log_error(y_true_1, y_pred_1),mean_squared_log_error(y_true_2, y_pred_2)]]

    return error_metrics_table
```

```python
def model_residual_metrics(y_train, y_model_validation_pred, y_model_tuning_pred, y_test, y_model_test_pred):

    validation_residual = y_train - y_model_validation_pred
    residuals_tuning_train = y_train - y_model_tuning_pred
    residuals_test = y_test - y_model_test_pred
    return validation_residual, residuals_tuning_train, residuals_test

def model_residual_plotting(y_train, y_model_validation_pred, y_model_tuning_pred, y_test, y_model_test_pred, ax, title):

    validation_residual, residuals_tuning_train, residuals_test = \
                                    model_residual_metrics(y_train, y_model_validation_pred, y_model_tuning_pred, y_test, y_model_test_pred)


    sns.scatterplot(ax=ax,x=y_model_validation_pred, y=validation_residual, label='Validation')
    sns.scatterplot(ax=ax,x=y_model_tuning_pred, y=residuals_tuning_train, label='Tuning')
    sns.scatterplot(ax=ax,x=y_model_test_pred, y=residuals_test, label='Test')

    ax.hlines(0, min(y_model_validation_pred), max(y_model_validation_pred), colors='r', linestyles='dashed')
    ax.hlines(0, min(y_model_tuning_pred), max(y_model_tuning_pred), colors='r', linestyles='dashed')
    ax.hlines(0, min(y_model_test_pred), max(y_model_test_pred), colors='r', linestyles='dashed')

    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Residuals')
    ax.set_title(title)

```

```python
def error_distribution(y_train, y_model_validation_pred, y_model_tuning_pred, y_test, y_model_test_pred, ax, title):
    #residual calculation
    validation_residual, residuals_tuning_train, residuals_test = \
                                    model_residual_metrics(y_train, y_model_validation_pred, y_model_tuning_pred, y_test, y_model_test_pred)

    # Calculate errors
    error_validation = -1*validation_residual
    tuning_error_train = -1*residuals_tuning_train
    error_test = -1*residuals_test

    # Plot error distribution

    sns.histplot(ax=ax, x=error_validation, kde=True, label='Validation errors', color='blue')
    sns.histplot(ax=ax, x=tuning_error_train, kde=True, label='Tuning errors', color='orange')
    sns.histplot(ax=ax, x=error_test, kde=True, label='Test errors', color='green')

    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    ax.set_title('poly_d2 Error distribution')
    ax.legend()


```

```python

def residual_and_error_plotting(uncorrelated_weighted_portfolio_trails_simulation_df):

    fig, ax =plt.subplots(2,3,figsize=(17, 17))
    #model validation
    popt_poly_d2, popt_poly_d3_log, popt_poly_d5_log, x_model_validation_poly_d2,x_model_validation_poly_d3_log, \
    x_model_validation_poly_d5_log, y_train_poly_d2, y_model_validation_pred_poly_d2, y_train_poly_d3_log, \
    y_model_validation_pred_poly_d3_log, y_train_poly_d5_log, y_model_validation_pred_poly_d5_log, model_poly_d2, \
    model_poly_d3_log, model_poly_d5_log, poly_d2_form, poly_d3_log_form, poly_d5_log_form = \
                                        evalute_model_parameters(uncorelated_weighted_portfolio_trails_simulation_df)
    # Model Fine-tuning
    model_poly_d2, model_poly_d3_log, model_poly_d5_log, popt_poly_d2, popt_poly_d3_log, popt_poly_d5_log, x_model_validation_poly_d2, \
    x_model_validation_poly_d3_log, x_model_validation_poly_d5_log, y_train_poly_d2, y_model_tuning_pred_poly_d2, y_train_poly_d3_log, \
    y_model_tuning_pred_poly_d3_log, y_train_poly_d5_log, y_model_tuning_pred_poly_d5_log, poly_d2_form, \
    poly_d3_log_form, poly_d5_log_form= fine_tune_hyperparmeters(uncorelated_weighted_portfolio_trails_simulation_df)


    # Model Testing
    model_poly_d2, model_poly_d3_log, model_poly_d5_log, popt_poly_d2, popt_poly_d3_log, popt_poly_d5_log, x_test_poly_d2, x_test_poly_d3_log, x_test_poly_d5_log, \
    y_test_poly_d2, y_model_test_pred_poly_d2,y_test_poly_d3_log, y_model_test_pred_poly_d3_log, y_test_poly_d5_log, \
    y_model_test_pred_poly_d5_log, poly_d2_form, poly_d3_log_form, poly_d5_log_form= \
                                    test_the_model(uncorelated_weighted_portfolio_trails_simulation_df)

    #-------------------------------------residual plotting---------------------------------------------------------------
    #model validation

    # poly_d2_residual
    model_residual_plotting(y_train_poly_d2, y_model_validation_pred_poly_d2, y_model_tuning_pred_poly_d2,
                        y_test_poly_d2, y_model_test_pred_poly_d2, ax[0,0],'Poly_d2')
    #poly_d3_log
    model_residual_plotting(y_train_poly_d3_log, y_model_validation_pred_poly_d3_log, y_model_tuning_pred_poly_d3_log,
                        y_test_poly_d3_log, y_model_test_pred_poly_d3_log, ax[0,1],'poly_d3_log')
    #poly_d5_log
    model_residual_plotting(y_train_poly_d5_log, y_model_validation_pred_poly_d5_log, y_model_tuning_pred_poly_d5_log,
                        y_test_poly_d5_log, y_model_test_pred_poly_d5_log, ax[0,2],'poly_d5_log')

    #-----------error plotting---------------------------------------------------------------------------------------------
    # poly_d2_residual
    error_distribution(y_train_poly_d2, y_model_validation_pred_poly_d2, y_model_tuning_pred_poly_d2,
                        y_test_poly_d2, y_model_test_pred_poly_d2, ax[1,0],'Poly_d2')
    #poly_d3_log
    error_distribution(y_train_poly_d3_log, y_model_validation_pred_poly_d3_log, y_model_tuning_pred_poly_d3_log,
                        y_test_poly_d3_log, y_model_test_pred_poly_d3_log, ax[1,1],'poly_d3_log')
    #poly_d5_log
    error_distribution(y_train_poly_d5_log, y_model_validation_pred_poly_d5_log, y_model_tuning_pred_poly_d5_log,
                        y_test_poly_d5_log, y_model_test_pred_poly_d5_log, ax[1,2],'poly_d5_log')


```

```python
def model_evalution_report(uncorrelated_weighted_portfolio_trails_simulation_df):

    print(" Model Validation ")
    popt_poly_d2, popt_poly_d3_log, popt_poly_d5_log, x_model_validation_poly_d2,x_model_validation_poly_d3_log, \
    x_model_validation_poly_d5_log, y_train_poly_d2, y_model_validation_pred_poly_d2, y_train_poly_d3_log, \
    y_model_validation_pred_poly_d3_log, y_train_poly_d5_log, y_model_validation_pred_poly_d5_log, model_poly_d2, \
    model_poly_d3_log, model_poly_d5_log, poly_d2_form, poly_d3_log_form, poly_d5_log_form = \
                                        evalute_model_parameters(uncorelated_weighted_portfolio_trails_simulation_df)

    print(tabulate(error_metrics_statistics(y_train_poly_d2, y_model_validation_pred_poly_d2, y_train_poly_d3_log, y_model_validation_pred_poly_d3_log,
                y_train_poly_d5_log, y_model_validation_pred_poly_d5_log, poly_d2_form, poly_d3_log_form, poly_d5_log_form), headers='firstrow',
                   tablefmt='fancy_grid', maxcolwidths=[None, 8]))

    print(" Model Fine-tuning ")
    model_poly_d2, model_poly_d3_log, model_poly_d5_log, popt_poly_d2, popt_poly_d3_log, popt_poly_d5_log, x_model_validation_poly_d2, \
    x_model_validation_poly_d3_log, x_model_validation_poly_d5_log, y_train_poly_d2, y_model_tuning_pred_poly_d2, y_train_poly_d3_log, \
    y_model_tuning_pred_poly_d3_log, y_train_poly_d5_log, y_model_tuning_pred_poly_d5_log, poly_d2_form, \
    poly_d3_log_form, poly_d5_log_form= fine_tune_hyperparmeters(uncorelated_weighted_portfolio_trails_simulation_df)

    print(tabulate(error_metrics_statistics(y_train_poly_d2, y_model_tuning_pred_poly_d2, y_train_poly_d3_log, y_model_tuning_pred_poly_d3_log, y_train_poly_d5_log,
                        y_model_tuning_pred_poly_d5_log, poly_d2_form, poly_d3_log_form, poly_d5_log_form), headers='firstrow',
                   tablefmt='fancy_grid', maxcolwidths=[None, 8]))

    print(" Model Testing ")
    model_poly_d2, model_poly_d3_log, model_poly_d5_log, popt_poly_d2, popt_poly_d3_log, popt_poly_d5_log, x_test_poly_d2, x_test_poly_d3_log, x_test_poly_d5_log, \
    y_test_poly_d2, y_model_test_pred_poly_d2,y_test_poly_d3_log, y_model_test_pred_poly_d3_log, y_test_poly_d5_log, \
    y_model_test_pred_poly_d5_log, poly_d2_form, poly_d3_log_form, poly_d5_log_form= \
                test_the_model(uncorelated_weighted_portfolio_trails_simulation_df)


    print(tabulate(error_metrics_statistics(y_test_poly_d2, y_model_test_pred_poly_d2,y_test_poly_d3_log, y_model_test_pred_poly_d3_log, y_test_poly_d5_log,
                        y_model_test_pred_poly_d5_log,  poly_d2_form, poly_d3_log_form, poly_d5_log_form  ), headers='firstrow',
                   tablefmt='fancy_grid', maxcolwidths=[None, 8]))

    residual_and_error_plotting(uncorelated_weighted_portfolio_trails_simulation_df)

```

####  Model Evalution Report


```python
 model_evalution_report(uncorelated_weighted_portfolio_trails_simulation_df)
```

```
 Model Validation 

```
```
'Poly_d2 : y =1.07801 * x^2 + -0.35063 * x + -0.76641'
```
```
'Poly_d3_log: y =2.11695 * np.log( 0.87254*x) + 0.30024 * x**3 + -1.13850 * x + 1.03991'
```
```
'Poly_d5_log: y =0.36785 * np.log( 0.94175*x) + -0.00860 * x**5'
```
```
╒══════════════════════════════════════╤═════════════════╤═════════════════════╤═════════════════════╕
│ Type Error                           │   Poly_d2 Error │   Poly_d3_log Error │   Poly_d5_log Error │
╞══════════════════════════════════════╪═════════════════╪═════════════════════╪═════════════════════╡
│ Mean Absolute Error(MAE)             │      0.00888175 │         0.00889831  │         0.00887809  │
├──────────────────────────────────────┼─────────────────┼─────────────────────┼─────────────────────┤
│ Mean Absolute Percentage Error(MAPE) │      0.172354   │         0.172944    │         0.172191    │
├──────────────────────────────────────┼─────────────────┼─────────────────────┼─────────────────────┤
│ Neg.Mean Squared Error(RMSE)         │     -0.0001423  │        -0.000145037 │        -0.000141573 │
├──────────────────────────────────────┼─────────────────┼─────────────────────┼─────────────────────┤
│ R-squared score                      │     -1.03709    │        -1.07626     │        -1.02668     │
├──────────────────────────────────────┼─────────────────┼─────────────────────┼─────────────────────┤
│ Mean Squared Error(MSE)              │      0.0001423  │         0.000145037 │         0.000141573 │
├──────────────────────────────────────┼─────────────────┼─────────────────────┼─────────────────────┤
│ Mean Squared Log Error(MSLE)         │      0.00012861 │         0.000131169 │         0.000127932 │
╘══════════════════════════════════════╧═════════════════╧═════════════════════╧═════════════════════╛
 Model Fine-tuning 

```
```
'Poly_d2 : y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'Poly_d3_log: y =0.25600 * np.log( 0.34800*x) + 0.00793 * x**3 + -0.06000 * x + 0.34300'
```
```
'Poly_d5_log: y =0.08500 * np.log( 1.44000*x) + -0.00058 * x**5'
```
```
╒══════════════════════════════════════╤═════════════════╤═════════════════════╤═════════════════════╕
│ Type Error                           │   Poly_d2 Error │   Poly_d3_log Error │   Poly_d5_log Error │
╞══════════════════════════════════════╪═════════════════╪═════════════════════╪═════════════════════╡
│ Mean Absolute Error(MAE)             │     0.0081338   │         0.0115017   │         0.00804915  │
├──────────────────────────────────────┼─────────────────┼─────────────────────┼─────────────────────┤
│ Mean Absolute Percentage Error(MAPE) │     0.175796    │         0.237246    │         0.16459     │
├──────────────────────────────────────┼─────────────────┼─────────────────────┼─────────────────────┤
│ Neg.Mean Squared Error(RMSE)         │    -0.000119847 │        -0.000192583 │        -0.000102487 │
├──────────────────────────────────────┼─────────────────┼─────────────────────┼─────────────────────┤
│ R-squared score                      │    -0.715665    │        -1.75691     │        -0.467142    │
├──────────────────────────────────────┼─────────────────┼─────────────────────┼─────────────────────┤
│ Mean Squared Error(MSE)              │     0.000119847 │         0.000192583 │         0.000102487 │
├──────────────────────────────────────┼─────────────────┼─────────────────────┼─────────────────────┤
│ Mean Squared Log Error(MSLE)         │     0.000108135 │         0.000172594 │         9.24266e-05 │
╘══════════════════════════════════════╧═════════════════╧═════════════════════╧═════════════════════╛
 Model Testing 

```
```
'Poly_d2 : y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'Poly_d3_log: y =0.25600 * np.log( 0.34800*x) + 0.00793 * x**3 + -0.06000 * x + 0.34300'
```
```
'Poly_d5_log: y =0.08500 * np.log( 1.44000*x) + -0.00058 * x**5'
```
```
╒══════════════════════════════════════╤═════════════════╤═════════════════════╤═════════════════════╕
│ Type Error                           │   Poly_d2 Error │   Poly_d3_log Error │   Poly_d5_log Error │
╞══════════════════════════════════════╪═════════════════╪═════════════════════╪═════════════════════╡
│ Mean Absolute Error(MAE)             │     0.00436791  │         0.00903178  │         0.00325997  │
├──────────────────────────────────────┼─────────────────┼─────────────────────┼─────────────────────┤
│ Mean Absolute Percentage Error(MAPE) │     0.0967441   │         0.177512    │         0.0716103   │
├──────────────────────────────────────┼─────────────────┼─────────────────────┼─────────────────────┤
│ Neg.Mean Squared Error(RMSE)         │    -4.6615e-05  │        -9.55576e-05 │        -2.57461e-05 │
├──────────────────────────────────────┼─────────────────┼─────────────────────┼─────────────────────┤
│ R-squared score                      │     0.310725    │        -0.412967    │         0.619305    │
├──────────────────────────────────────┼─────────────────┼─────────────────────┼─────────────────────┤
│ Mean Squared Error(MSE)              │     4.6615e-05  │         9.55576e-05 │         2.57461e-05 │
├──────────────────────────────────────┼─────────────────┼─────────────────────┼─────────────────────┤
│ Mean Squared Log Error(MSLE)         │     4.23611e-05 │         8.57629e-05 │         2.346e-05   │
╘══════════════════════════════════════╧═════════════════╧═════════════════════╧═════════════════════╛

```
```
<Figure size 1700x1700 with 6 Axes>
```
#### Winning Model


```python
def get_wining_model(uncorrelated_weighted_portfolio_trails_simulation_df):
    model_poly_d2, model_poly_d3_log, model_poly_d5_log, popt_poly_d2, popt_poly_d3_log, popt_poly_d5_log, \
    x_test_poly_d2, x_test_poly_d3_log, x_test_poly_d5_log, y_test_poly_d2, y_model_test_pred_poly_d2, \
    y_test_poly_d3_log, y_model_test_pred_poly_d3_log, y_test_poly_d5_log, \
    y_model_test_pred_poly_d5_log, poly_d2_form, poly_d3_log_form, poly_d5_log_form= test_the_model(uncorelated_weighted_portfolio_trails_simulation_df)

    return model_poly_d2, popt_poly_d2, poly_d2_form
```

### Prediction when the investor's risk level metric (portfolio standard deviation) is known

Here we will use the wining efficient frontier model to predict the portfolio expected return. Then will calculater the portfolio weightsand and investment strategy

The following 2 Strategies will be implemented to manage the volatility:



1. Asset Allocation: Adjusting the proportion of different asset classes in a portfolio to balance risk.

2. Diversification: Spreading investments across various sectors.


```python
def predict_portfolio_expectded_return(uncorrelated_weighted_portfolio_trails_simulation_df, risk):
    model_poly_d2, popt_poly_d2, poly_d2_form = get_wining_model(uncorrelated_weighted_portfolio_trails_simulation_df)
    display(poly_d2_form)
    return model_poly_d2(risk, *popt_poly_d2)


pred_portfolio_expected_return  = predict_portfolio_expectded_return(uncorelated_weighted_portfolio_trails_simulation_df, 1.3)

```

```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```python

def get_assets_expected_returns_and_stickers(log_returns, most_diversify_portfolio_assets_list):
    #uncorrelated_assets_log_returns = uncorrelated_assets_returns_log_returns_df(log_returns, selecting_uncorrelated_assets(log_returns,threshold))
    uncorrelated_assets_log_returns = uncorrelated_assets_returns_log_returns_df(log_returns, most_diversify_portfolio_assets_list)
    uncorrelated_assets_expected_return = uncorrelated_assets_log_returns.mean()
    #display(uncorrelated_assets_espected_return)
    #type(uncorrelated_assets_espected_return)
    assets_ticker_list = uncorrelated_assets_expected_return.index.tolist()
    #display(assets_ticker_list)
    assets_expected_returns_list = uncorrelated_assets_expected_return.to_list()
    return assets_expected_returns_list, assets_ticker_list

assets_expected_returns_list, assets_ticker_list = get_assets_expected_returns_and_stickers(log_returns,most_diversify_portfolio_assets_list)
```

```python

def get_portfolio_investment_strategy_df( log_returns, uncorrelated_weighted_portfolio_trails_simulation_df,
                                         most_diversify_portfolio_assets_list, portfolio_risk):

    sum_weight_and_portfolio_return_list = []

    #stocks expected return
    assets_expected_returns_list, assets_ticker_list = get_assets_expected_returns_and_stickers(log_returns,most_diversify_portfolio_assets_list)
    assets_expected_returns_list = np.array(assets_expected_returns_list)*100
    assets_expected_returns_list = list(np.round(assets_expected_returns_list, 3))

    #predicted portfolio expected return, given the portfolio volatility(risk)
    portfolio_return_predicted_value= round(predict_portfolio_expectded_return(uncorrelated_weighted_portfolio_trails_simulation_df, portfolio_risk),3)

    #assets expected return absolute deviation from the portfolio expected return
    assets_expected_return_absolute_deviation_list = abs(portfolio_return_predicted_value - assets_expected_returns_list)
    assets_expected_return_absolute_deviation_list = list(np.round(assets_expected_return_absolute_deviation_list, 3))
    sum_expected_return_absolute_deviation = round(sum(assets_expected_return_absolute_deviation_list),3)

    #assets weight coefficients list
    assets_weight_list = assets_expected_return_absolute_deviation_list/sum_expected_return_absolute_deviation
    assets_weight_list = list(np.round(assets_weight_list, 3))

    #include the index content into the portfolio strategy data frame
    portfolio_content_df = index_content_df[index_content_df['Ticker'].isin(assets_ticker_list)]

    #portfolio strategy data frame
    portfolio_investment_strategy_df = pd.DataFrame({'Ticker':assets_ticker_list,'Weight':assets_weight_list,
                                                     'Asset Espected Returns':assets_expected_returns_list})
    portfolio_investment_strategy_df = portfolio_investment_strategy_df.sort_values(by='Weight',ascending=True)

    #merge content data frame and the weght data frame
    portfolio_investment_strategy_df = pd.merge(portfolio_content_df, portfolio_investment_strategy_df, how="inner", on=["Ticker"])

    return portfolio_investment_strategy_df


portfolio_investment_strategy_df = get_portfolio_investment_strategy_df( log_returns,
                                                    uncorelated_weighted_portfolio_trails_simulation_df, most_diversify_portfolio_assets_list, 1.3)
```

```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```python
def portfolio_annotation(x, y, text, ax):
    # Loop for annotation of all points
    for i in range(len(x)):
        ax.annotate(text[i]+'(σp='+str(round(x[i],3))+';E_rp='+ str(round(y[i],3))+')',
                    xy=(x[i], y[i]),xycoords='data', xytext= (x[i], y[i] ))

```

```python
def plotting_selected_efficient_frontier_predicted_portfolio(uncorrelated_weighted_portfolio_trails_simulation_df,risk):
    fig, ax =plt.subplots(figsize=(12, 5))
    text = ["A", "B", "C", "D", "E", "F"]
    model_poly_d2, popt_poly_d2, poly_d2_form = get_wining_model(uncorrelated_weighted_portfolio_trails_simulation_df)
    predicted_return = model_poly_d2(risk, *popt_poly_d2)
    ax.plot(risk, predicted_return,'*',color='red',label='Optimal portfolios')
    scplt = model_uperBound_efficient_frontier(uncorrelated_weighted_portfolio_trails_simulation_df, model_poly_d2,popt_poly_d2, ax, poly_d2_form)
    portfolio_annotation(risk, predicted_return, text, ax)
    cb = fig.colorbar(scplt, ax=ax, label='Sharpe Ratio')
```

```python
def plot_investment_strategy_pie_chart(log_returns, uncorelated_weighted_portfolio_trails_simulation_df,
                                       most_diversify_portfolio_assets_list, portfolio_risk, risk_profile = ''):

    portfolio_investment_strategy_df = get_portfolio_investment_strategy_df( \
                log_returns,uncorelated_weighted_portfolio_trails_simulation_df, most_diversify_portfolio_assets_list, portfolio_risk)
    portfolio_investment_strategy_df = portfolio_investment_strategy_df.sort_values(by='Weight',ascending=True)

    industry_labels = portfolio_investment_strategy_df['Industry'].values
    sector_labels = portfolio_investment_strategy_df['Sector'].values
    weight_values = portfolio_investment_strategy_df['Weight'].values

    # Create subplots: use 'domain' type for Pie subplot
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

    fig.add_trace(go.Pie(labels=industry_labels, values=weight_values, name="Industry",
                        legendgroup="Industry",  # this can be any string, not just "group"
                        legendgrouptitle_text="Industry"), 1, 1)
    fig.add_trace(go.Pie(labels=sector_labels, values=weight_values, name="Sector",
                        legendgroup="Sector",  # this can be any string, not just "group"
                        legendgrouptitle_text="Sector"), 1, 2)



    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.5, hoverinfo="label+percent+name")

    fig.update_layout(
    title_text= risk_profile+" Suggested Investment by Industry & Sector",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Industry', x=0.14, y=0.5, font_size=20, showarrow=False),
                 dict(text='Sector', x=0.84, y=0.5, font_size=20, showarrow=False)],
    height=500,
    width=800,
    autosize=True,
    margin=dict(t=0, b=0, l=50, r=0),
    legend_tracegroupgap = 0,
    legend=dict(
                    orientation="v",
                    yanchor="bottom",
                    y=0,
                    xanchor="right",
                    x=1.5),
     title=dict(
                    y=0.9,
                    x=0.1,
                    xanchor= 'left',
                    yanchor= 'top'))

    fig.show()
```

```python
def plot_asset_return_pie_chart(log_returns, uncorelated_weighted_portfolio_trails_simulation_df,
                                most_diversify_portfolio_assets_list, portfolio_risk, risk_profile = ''):

    portfolio_investment_strategy_df = get_portfolio_investment_strategy_df( \
                log_returns,uncorelated_weighted_portfolio_trails_simulation_df, most_diversify_portfolio_assets_list, portfolio_risk)
    portfolio_investment_strategy_df = portfolio_investment_strategy_df.sort_values(by='Asset Espected Returns',ascending=True)

    industry_labels = portfolio_investment_strategy_df['Industry'].values
    sector_labels = portfolio_investment_strategy_df['Sector'].values
    weight_values = portfolio_investment_strategy_df['Asset Espected Returns'].values

    # Create subplots: use 'domain' type for Pie subplot
    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

    fig.add_trace(go.Pie(labels=industry_labels, values=weight_values, name="Industry",
                        legendgroup="Industry",  # this can be any string, not just "group"
                        legendgrouptitle_text="Industry"), 1, 1)
    fig.add_trace(go.Pie(labels=sector_labels, values=weight_values, name="Sector",
                        legendgroup="Sector",  # this can be any string, not just "group"
                        legendgrouptitle_text="Sector"), 1, 2)

    # Use `hole` to create a donut-like pie chart
    fig.update_traces(hole=.5, hoverinfo="label+percent+name")

    fig.update_layout(
    title_text=risk_profile+" Asset Returns by Industry & Sector",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Industry', x=0.14, y=0.5, font_size=20, showarrow=False),
                 dict(text='Sector', x=0.84, y=0.5, font_size=20, showarrow=False)],
    height=500,
    width=800,
    autosize=True,
    margin=dict(t=0, b=0, l=50, r=0),
    legend_tracegroupgap = 0,
    legend=dict(
                    orientation="v",
                    yanchor="bottom",
                    y=0,
                    xanchor="right",
                    x=1.5),
     title=dict(
                    y=0.9,
                    x=0.1,
                    xanchor= 'left',
                    yanchor= 'top'))

    fig.show()
```

```python
#Finding weights of portfolio when return given
def plot_asset_return( log_returns, uncorelated_weighted_portfolio_trails_simulation_df,
                      most_diversify_portfolio_assets_list, portfolio_risk,risk_profile = ''):
    fig, ax =plt.subplots(figsize=(12, 6))

    #plotting Asset Espected Returns
    portfolio_investment_strategy_df = get_portfolio_investment_strategy_df( \
                log_returns,uncorelated_weighted_portfolio_trails_simulation_df, most_diversify_portfolio_assets_list, portfolio_risk)
    portfolio_investment_strategy_df = portfolio_investment_strategy_df.sort_values(by='Asset Espected Returns',ascending=True)
    column_list = [':      ' for i in range(len(portfolio_investment_strategy_df))]
    column_df = pd.DataFrame({'colum': column_list})

    asset_return = portfolio_investment_strategy_df['Asset Espected Returns']
    strategy_Stickers = portfolio_investment_strategy_df['Sector'] + column_df['colum'] + \
                        portfolio_investment_strategy_df['Industry'] + column_df['colum'] + \
                        portfolio_investment_strategy_df['Company'] + \
                        column_df['colum'] + portfolio_investment_strategy_df['Ticker']

    bar_container= ax.barh(strategy_Stickers, asset_return*100)
    ax.axes.get_xaxis().set_visible(False)
    # setting label of y-axis
    ax.set_ylabel("Asset Tickers")
    # setting label of x-axis
    ax.set_xlabel("Asset Return")
    ax.set_title(risk_profile+" Asset Return",fontsize=22,  horizontalalignment='right',fontweight='roman')
    ax.bar_label(bar_container, fmt='{:,.1f}%')


    plt.show()
    #Asset return pie chart
    plot_asset_return_pie_chart( log_returns, uncorelated_weighted_portfolio_trails_simulation_df,
                                most_diversify_portfolio_assets_list, portfolio_risk, risk_profile)

```

```python
#Finding weights of portfolio when return given
def plot_predicted_portfolio_weight( log_returns, uncorelated_weighted_portfolio_trails_simulation_df,
                                    most_diversify_portfolio_assets_list, portfolio_risk, risk_profile = ''):
    fig, ax =plt.subplots(figsize=(12, 6))
    portfolio_investment_strategy_df = get_portfolio_investment_strategy_df( \
                log_returns,uncorelated_weighted_portfolio_trails_simulation_df, most_diversify_portfolio_assets_list, portfolio_risk)
    portfolio_investment_strategy_df = portfolio_investment_strategy_df.sort_values(by='Weight',ascending=True)
    column_list = [':      ' for i in range(len(portfolio_investment_strategy_df))]
    column_df = pd.DataFrame({'colum': column_list})

    #plotting
    display(portfolio_investment_strategy_df.style.hide(axis='index'))

    strategy_Weight = portfolio_investment_strategy_df['Weight']
    strategy_Stickers = portfolio_investment_strategy_df['Sector'] + column_df['colum'] + \
                        portfolio_investment_strategy_df['Industry'] + column_df['colum'] + \
                        portfolio_investment_strategy_df['Company'] + \
                        column_df['colum'] + portfolio_investment_strategy_df['Ticker']
    bar_container= ax.barh(strategy_Stickers, strategy_Weight*100)

    ax.axes.get_xaxis().set_visible(False)
    #setting label of y-axis
    ax.set_ylabel("Asset Stickers")
    # setting label of x-axis
    ax.set_xlabel("Portfolio Weight")
    ax.set_title(risk_profile+" suggested Portfolio Allocation", fontsize=22, horizontalalignment='right')
    ax.bar_label(bar_container, fmt='{:,.1f}%')

    plt.show()

    #Investement strategy pie chart
    plot_investment_strategy_pie_chart( log_returns, uncorelated_weighted_portfolio_trails_simulation_df,
                                       most_diversify_portfolio_assets_list, portfolio_risk, risk_profile)

```

```python

def get_portolio_risk_input(uncorrelated_weighted_portfolio_trails_simulation_df, portfolio_risk):
    predited_portfolio_return = predict_portfolio_expectded_return(uncorrelated_weighted_portfolio_trails_simulation_df, portfolio_risk)
    prediction_df = pd.DataFrame([{'portfolio_risk':portfolio_risk,'Predited Portfolio Return':predited_portfolio_return}])
    display(prediction_df.style.hide(axis='index'))

```

```python
def risk_tolerence_threshold(uncorrelated_weighted_portfolio_trails_simulation_df, portfolio_risk = 1.3):
    #define threshold to track the investor risk tolerence:High risk tolerance (aggressive investors), Moderate risk tolerance (moderate investors)
    #Low risk tolerance (conservative investors)
    pred_random_portfolio_return = predict_portfolio_expectded_return(uncorrelated_weighted_portfolio_trails_simulation_df, portfolio_risk)

    max_E_rp_sharpe_ratio, max_E_rp, max_E_rp_σp = get_maximun_return_portfolio(uncorrelated_weighted_portfolio_trails_simulation_df)
    pred_maximun_return_portfolio = predict_portfolio_expectded_return(uncorrelated_weighted_portfolio_trails_simulation_df, max_E_rp_σp)

    max_σp_E_rp_sharpe_ratio, max_σp_E_rp, max_σp = get_maximun_risk_portfolio(uncorrelated_weighted_portfolio_trails_simulation_df)
    pred_maximun_risk_portfolio = predict_portfolio_expectded_return(uncorrelated_weighted_portfolio_trails_simulation_df, max_σp)

    maximum_sharpe_ratio, maximum_sharpe_ratio_σp_E_rp, maximum_sharpe_ratio_σp =  get_maximum_sharpe_ratio(uncorrelated_weighted_portfolio_trails_simulation_df)
    pred_maximum_sharpe_ratio = predict_portfolio_expectded_return(uncorrelated_weighted_portfolio_trails_simulation_df, maximum_sharpe_ratio_σp)

    minimum_σp_E_rp_sharpe_ratio, minimum_σp_E_rp, minimum_σp = get_minimum_risk_portfolio(uncorrelated_weighted_portfolio_trails_simulation_df)
    pred_minimum_risk_portfolio = predict_portfolio_expectded_return(uncorrelated_weighted_portfolio_trails_simulation_df, minimum_σp)
    avg_risk = uncorrelated_weighted_portfolio_trails_simulation_df['σp'].mean()
    pred_avg_risk_Expected_return = predict_portfolio_expectded_return(uncorrelated_weighted_portfolio_trails_simulation_df, avg_risk)

    index = ['A', 'B', 'C', 'D', 'E', 'F']
    risk_tolerence_threshold_df =pd.DataFrame({'Portfolio Type': ['Random Portfolio', 'Maximun Return Portfolio','Maximun Risk Portfolio',
                                                                  'Maximum Sharpe Ratio(Tangent Portfolio)',
                                              'Minimum Risk Portfolio', 'Average Volatilty'],
                              'Predicted Expected Return': [pred_random_portfolio_return, pred_maximun_return_portfolio, pred_maximun_risk_portfolio,
                                                            pred_maximum_sharpe_ratio, pred_minimum_risk_portfolio, pred_avg_risk_Expected_return ],
                              'Portfolio Risk(volatility)':[portfolio_risk, max_E_rp_σp, max_σp, maximum_sharpe_ratio_σp, minimum_σp, avg_risk],
                              'Sharpe Ratio':[pred_random_portfolio_return/portfolio_risk, pred_maximun_return_portfolio/maximum_sharpe_ratio_σp,
                                              pred_maximun_risk_portfolio/max_σp, pred_maximum_sharpe_ratio/maximum_sharpe_ratio_σp ,
                                              pred_minimum_risk_portfolio/ minimum_σp, pred_avg_risk_Expected_return/avg_risk]},
                                           index=index)

    return risk_tolerence_threshold_df

```

```python
def plot_risk_tolerence_treshold(uncorrelated_weighted_portfolio_trails_simulation_df, portfolio_risk = 1.3):
    risk_tolerence_threshold_df = risk_tolerence_threshold(uncorrelated_weighted_portfolio_trails_simulation_df, portfolio_risk)

    print('\n                                              **********************************************************************\n'+
              '                                               Optimal Portfolio Table - Winning Model and Efficient Frontier\n'+
              '                                              **********************************************************************\n')
    display(risk_tolerence_threshold_df)
    portfolio_risk_values =  risk_tolerence_threshold_df['Portfolio Risk(volatility)'].values
    plotting_selected_efficient_frontier_predicted_portfolio(uncorrelated_weighted_portfolio_trails_simulation_df,portfolio_risk_values)

```

```python

def plot_suggested_portfolio_structure( log_returns, uncorrelated_weighted_portfolio_trails_simulation_df,
                                       most_diversify_portfolio_assets_list, portfolio_risk):

    risk_tolerence_threshold_df =  risk_tolerence_threshold(uncorrelated_weighted_portfolio_trails_simulation_df, portfolio_risk)

    for i in range(len(risk_tolerence_threshold_df)):
        portfolio_risk = risk_tolerence_threshold_df['Portfolio Risk(volatility)'][i]
        predicted_expected_return = risk_tolerence_threshold_df['Predicted Expected Return'][i]
        sharpe_ratio = risk_tolerence_threshold_df['Sharpe Ratio'][i]
        print('\n                                    *************************************\n'+
              '                                      Portfolio Risk(volatility)  : '+str(round(portfolio_risk,3))+'\n'+
              '                                      Predicted Expected Return   : '+str(round(predicted_expected_return,3))+'\n'+
              '                                      Sharpe Ratio                : '+str(round(sharpe_ratio,3))+'\n'
              '                                     *************************************\n')
        plot_predicted_portfolio_weight( log_returns, uncorrelated_weighted_portfolio_trails_simulation_df,
                                        most_diversify_portfolio_assets_list, portfolio_risk)
        plot_asset_return( log_returns, uncorrelated_weighted_portfolio_trails_simulation_df, most_diversify_portfolio_assets_list, portfolio_risk)
```

```python
def risk_tolerence_encoding(uncorrelated_weighted_portfolio_trails_simulation_df):

    risk_tolerence_threshold_df = risk_tolerence_threshold(uncorrelated_weighted_portfolio_trails_simulation_df)
    max_Erp = risk_tolerence_threshold_df['Portfolio Risk(volatility)']['B']
    max_riskp = risk_tolerence_threshold_df['Portfolio Risk(volatility)']['C']
    max_shape_ratiop = risk_tolerence_threshold_df['Portfolio Risk(volatility)']['D']
    min_riskp = risk_tolerence_threshold_df['Portfolio Risk(volatility)']['E']
    avg_riskp = risk_tolerence_threshold_df['Portfolio Risk(volatility)']['F']
    #σp	E_rp

    simulated_risk_list =  uncorelated_weighted_portfolio_trails_simulation_df['σp']
    pred_Expected_return_list = predict_portfolio_expectded_return(uncorrelated_weighted_portfolio_trails_simulation_df, simulated_risk_list)
    sharpe_ratio_list =pred_Expected_return_list/simulated_risk_list

    risk_profile_list = []
    risk_profile_encoding_list = []

    for i in range(len(simulated_risk_list)):
        portfolio_risk = simulated_risk_list[i]
        if portfolio_risk >= max_shape_ratiop and portfolio_risk <=avg_riskp :
            risk_profile_list.append('Moderate')
            risk_profile_encoding_list.append(1)
        elif portfolio_risk <max_shape_ratiop:
            risk_profile_list.append('Conservative')
            risk_profile_encoding_list.append(2)
        elif portfolio_risk  > avg_riskp:
            risk_profile_list.append('Aggressive')
            risk_profile_encoding_list.append(3)

    risk_tolerence_rating_df= pd.DataFrame({'Simulated Risk': simulated_risk_list, 'Predicted Expected Return':pred_Expected_return_list,
                                            'Sharpe Ratio':sharpe_ratio_list, 'Risk Profile':risk_profile_list,
                                            'Risk Profile Encoding Value':risk_profile_encoding_list})
    return  risk_tolerence_rating_df

risk_tolerence_encoding_df = risk_tolerence_encoding(uncorelated_weighted_portfolio_trails_simulation_df)
display(risk_tolerence_encoding_df)
```

```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
      Simulated Risk  Predicted Expected Return  Sharpe Ratio  Risk Profile  \
0           1.463924                   0.060298      0.041189  Conservative   
1           1.456257                   0.059883      0.041121  Conservative   
2           1.519654                   0.062076      0.040849  Conservative   
3           1.447017                   0.059329      0.041001  Conservative   
4           1.590675                   0.061184      0.038464    Aggressive   
...              ...                        ...           ...           ...   
9995        1.450366                   0.059537      0.041049  Conservative   
9996        1.499805                   0.061692      0.041134  Conservative   
9997        1.466001                   0.060404      0.041203  Conservative   
9998        1.368392                   0.052185      0.038136  Conservative   
9999        1.480896                   0.061070      0.041239  Conservative   

      Risk Profile Encoding Value  
0                               2  
1                               2  
2                               2  
3                               2  
4                               3  
...                           ...  
9995                            2  
9996                            2  
9997                            2  
9998                            2  
9999                            2  

[10000 rows x 5 columns]
```
```python
def get_risk_profile_matrix(uncorelated_weighted_portfolio_trails_simulation_df):
    risk_tolerence_encoding_df = risk_tolerence_encoding(uncorelated_weighted_portfolio_trails_simulation_df)
    risk_profile_matrix =  risk_tolerence_encoding_df.groupby('Risk Profile')[['Simulated Risk','Predicted Expected Return','Sharpe Ratio']].mean()
    return pd.DataFrame(risk_profile_matrix)

risk_profile_matrix = get_risk_profile_matrix(uncorelated_weighted_portfolio_trails_simulation_df)
risk_profile_matrix
```

```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
              Simulated Risk  Predicted Expected Return  Sharpe Ratio
Risk Profile                                                         
Aggressive          1.610688                   0.060046      0.037308
Conservative        1.455506                   0.058903      0.040447
```
```python

def plot_suggested_risk_profile_portfolio_structure( log_returns, uncorrelated_weighted_portfolio_trails_simulation_df,
                                                    most_diversify_portfolio_assets_list, portfolio_risk=1.3):

    #risk_tolerence_threshold_df =  risk_tolerence_threshold(uncorrelated_weighted_portfolio_trails_simulation_df, portfolio_risk)
    risk_profile_matrix = get_risk_profile_matrix(uncorelated_weighted_portfolio_trails_simulation_df)

    print('\n                                              **********************************************************************\n'+
              '                                                    Investment Profile Simulation And Portfolio Allocation \n'+
              '                                              **********************************************************************\n')
    display(risk_profile_matrix)
    for i in range(len(risk_profile_matrix)):
        risk_profile = risk_profile_matrix.index[i]
        portfolio_risk = risk_profile_matrix['Simulated Risk'][i]
        predicted_expected_return = risk_profile_matrix['Predicted Expected Return'][i]
        sharpe_ratio = risk_profile_matrix['Sharpe Ratio'][i]
        print('\n                                    *****************************************************\n'+
              '                                      Risk Profile                : '+risk_profile+' Investment \n'+
              '                                      Simulated Risk              : '+str(round(portfolio_risk,3))+'\n'+
              '                                      Predicted Expected Return   : '+str(round(predicted_expected_return,3))+'\n'+
              '                                      Sharpe Ratio                : '+str(round(sharpe_ratio,3))+'\n'
              '                                     *****************************************************\n')
        plot_predicted_portfolio_weight( log_returns, uncorrelated_weighted_portfolio_trails_simulation_df,
                                        most_diversify_portfolio_assets_list, portfolio_risk,risk_profile)
        plot_asset_return( log_returns, uncorrelated_weighted_portfolio_trails_simulation_df,
                          most_diversify_portfolio_assets_list, portfolio_risk,risk_profile)
```

```python
plot_risk_tolerence_treshold(uncorelated_weighted_portfolio_trails_simulation_df)
```

```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```

                                              **********************************************************************
                                               Optimal Portfolio Table - Winning Model and Efficient Frontier
                                              **********************************************************************


```
```
                            Portfolio Type  Predicted Expected Return  \
A                         Random Portfolio                   0.042446   
B                 Maximun Return Portfolio                   0.061299   
C                   Maximun Risk Portfolio                   0.050754   
D  Maximum Sharpe Ratio(Tangent Portfolio)                   0.061299   
E                   Minimum Risk Portfolio                   0.034486   
F                        Average Volatilty                   0.059935   

   Portfolio Risk(volatility)  Sharpe Ratio  
A                    1.300000      0.032651  
B                    1.587529      0.038613  
C                    1.717820      0.029546  
D                    1.587529      0.038613  
E                    1.256201      0.027452  
F                    1.457167      0.041131  
```
```
<Figure size 1200x500 with 2 Axes>
```
```python
plot_suggested_portfolio_structure( log_returns, uncorelated_weighted_portfolio_trails_simulation_df,
                                   most_diversify_portfolio_assets_list, 1.3)
```

```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```

                                    *************************************
                                      Portfolio Risk(volatility)  : 1.3
                                      Predicted Expected Return   : 0.042
                                      Sharpe Ratio                : 0.033
                                     *************************************


```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<pandas.io.formats.style.Styler at 0x2c5e97352d0>
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```

                                    *************************************
                                      Portfolio Risk(volatility)  : 1.588
                                      Predicted Expected Return   : 0.061
                                      Sharpe Ratio                : 0.039
                                     *************************************


```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<pandas.io.formats.style.Styler at 0x2c5e98b55d0>
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```

                                    *************************************
                                      Portfolio Risk(volatility)  : 1.718
                                      Predicted Expected Return   : 0.051
                                      Sharpe Ratio                : 0.03
                                     *************************************


```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<pandas.io.formats.style.Styler at 0x2c5e512ce50>
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```

                                    *************************************
                                      Portfolio Risk(volatility)  : 1.588
                                      Predicted Expected Return   : 0.061
                                      Sharpe Ratio                : 0.039
                                     *************************************


```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<pandas.io.formats.style.Styler at 0x2c5e26fcb90>
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```

                                    *************************************
                                      Portfolio Risk(volatility)  : 1.256
                                      Predicted Expected Return   : 0.034
                                      Sharpe Ratio                : 0.027
                                     *************************************


```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<pandas.io.formats.style.Styler at 0x2c5e2575350>
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```

                                    *************************************
                                      Portfolio Risk(volatility)  : 1.457
                                      Predicted Expected Return   : 0.06
                                      Sharpe Ratio                : 0.041
                                     *************************************


```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<pandas.io.formats.style.Styler at 0x2c5e1f00150>
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```python
plot_suggested_risk_profile_portfolio_structure( log_returns, uncorelated_weighted_portfolio_trails_simulation_df,
                                                most_diversify_portfolio_assets_list, portfolio_risk=1.3)
```

```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```

                                              **********************************************************************
                                                    Investment Profile Simulation And Portfolio Allocation 
                                              **********************************************************************


```
```
              Simulated Risk  Predicted Expected Return  Sharpe Ratio
Risk Profile                                                         
Aggressive          1.610688                   0.060046      0.037308
Conservative        1.455506                   0.058903      0.040447
```
```

                                    *****************************************************
                                      Risk Profile                : Aggressive Investment 
                                      Simulated Risk              : 1.611
                                      Predicted Expected Return   : 0.06
                                      Sharpe Ratio                : 0.037
                                     *****************************************************


```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<pandas.io.formats.style.Styler at 0x2c5e2a4d9d0>
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```

                                    *****************************************************
                                      Risk Profile                : Conservative Investment 
                                      Simulated Risk              : 1.456
                                      Predicted Expected Return   : 0.059
                                      Sharpe Ratio                : 0.04
                                     *****************************************************


```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<pandas.io.formats.style.Styler at 0x2c5e286ba90>
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
# K-means Clustering in Portfolio Analytics - Investment Risk Profiles Simulation



My main objective is to use  K-means clustering to found out the investment type of risk tolerence: very conservative, conservative, moderate, aggressive and very aggressive. From the Elbow model, I will  assume the optimal number of clusters is 5



In this section, I will combine K-means clustering with efficient frontier modeling to dig into the randomn generated portfolios.  In order to simulate the investors risk tolerence, I will use K-means clustering and optimal portfolio modeling on top of the privious covarience matrix technic that I used along with the covarience treshold  to reduce the volume of the assets. I used the covarience coefficient to filter uncorelated assets.   Then I will recommend an investment stratagy that optimize return for each type for risk tolerence.

I'm using the wining model 'y =0.07000 * x^2 + -0.01600 * x + -0.00900' to predict the portfolio expected returns. The simulated portfolio risk is combigned with the simulated the portfolio expected return and the predicted expected return to set the randomn efficient frontier data. The randomn efficient frontier data is then used as input for the  K-means cluster models





```python
def calculate_number_of_cluster(uncorrelated_weighted_portfolio_trails_simulation_df, n_components, ax):
    # Randomn efficient frontier data collection
    portfolio_risk_list = uncorrelated_weighted_portfolio_trails_simulation_df['σp']
    portefolio_return_list = uncorrelated_weighted_portfolio_trails_simulation_df['E_rp']
    predited_portfolio_return_list = predict_portfolio_expectded_return(uncorrelated_weighted_portfolio_trails_simulation_df, portfolio_risk_list)
    clipped_df = dataframe_clipping(portfolio_risk_list, portefolio_return_list, predited_portfolio_return_list )

    range_nbr_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # Step 2: Standardize the data
    scaler = StandardScaler()
    scaled_efficient_Frontier_data = scaler.fit_transform(clipped_df)


    #Determine the Number of clusters using Within Cluster Sum of Squares(wcss)
    wcss = [] # (Within Cluster Sum of Squares:inertia)
    silhouette_average_list = []

    for n2 in range_nbr_clusters:
        kmeans = KMeans(n_clusters=n2, init ='k-means++', max_iter=300,  n_init=10,random_state=0 )

        kmeans.fit(clipped_df)
        wcss.append(kmeans.inertia_)
        cluster_labels = kmeans.fit_predict(clipped_df)
        silhouette_average_list.append(silhouette_score(clipped_df, cluster_labels))

    ax1 = ax.twinx()
    ax.plot(range_nbr_clusters, wcss, 'b-', marker='o')
    ax1.plot(range_nbr_clusters,silhouette_average_list, 'g-', marker='o')

    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Within Cluster Sum of Squares(wcss)')
    ax1.set_ylabel('Silhouette score')
    ax.set_title('Elbow Method & Silhouette Analysis for Optimal Number of Clusters')
    #plt.show()

    return clipped_df

```

```python

def implement_k_means_clusters(clipped_df, fig, ax):
    # our main objective is to use K-means clustering to found out investment risk profile: very conservative, conservative, moderate, aggressive and very aggressive.
    # So from the Elbow model, let's assume the optimal number of clusters is 5

    kmeans = KMeans(n_clusters=5, init ='k-means++', max_iter=300, n_init=10,random_state=0 )
    #clusters = kmeans.fit_predict(pca_data)
    pred_clusters = kmeans.fit_predict(clipped_df)

    rand_data_point_and_cluster_df = clipped_df
    rand_data_point_and_cluster_df['cluster'] = pred_clusters


    #investment profile
    investment_profiles_index = ['Moderate', 'Conservative', 'Agressive', 'Very Aggressive', 'Very Conservative']
    investment_profiles_color = ['purple', 'gold', 'limegreen', 'green', 'yellow']
    display(rand_data_point_and_cluster_df)

    #plot cluster
    for i in range(len(investment_profiles_index)):
        cspl = ax.scatter(x=rand_data_point_and_cluster_df.loc[(rand_data_point_and_cluster_df['cluster'] ==i), ['σp']],
                      y=rand_data_point_and_cluster_df.loc[(rand_data_point_and_cluster_df['cluster'] ==i), ['E_rp']],
                      c= investment_profiles_color[i], cmap="viridis",label=investment_profiles_index[i])

    # find clusters centratides
    cluster_centers_df = pd.DataFrame(kmeans.cluster_centers_)
    cluster_centers_df = cluster_centers_df.set_axis( kmeans.feature_names_in_ , axis=1)
    cluster_centers_df.index =  investment_profiles_index
    cluster_centers_df.index.names = ['Investment Profile']

    #plot clusters centroid.
    ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker=".", s=100, c='red', label = 'Cluster Centroids')

    #plot efficienr frontier model
    model_poly_d2, popt_poly_d2, poly_d2_form = get_wining_model(clipped_df)
    xpoints,ypoints,top_sharpe_ratio_value_points = efficient_frontiere_optimal_portfolios_model_points(clipped_df,7)
    x_model_σp = np.linspace(xpoints.min(), xpoints.max(), len(clipped_df))
    y_model_E_rp_pred = model_poly_d2(x_model_σp, *popt_poly_d2)
    cspl = ax.scatter(x=x_model_σp, y=y_model_E_rp_pred, marker="*", c= y_model_E_rp_pred/x_model_σp,
                      cmap="viridis",label='Efficient Frontier:\n'+poly_d2_form)

    #find model predicted centroide expected return
    pred_centroide_Expr_list = []
    pred_centroide_Expr_list = model_poly_d2(cluster_centers_df['σp'], *popt_poly_d2)


    cluster_centers_df['Pred Centroide Expr'] =pred_centroide_Expr_list
    cluster_centers_df['Pred Centroide Sharpe Ratio'] =pred_centroide_Expr_list/cluster_centers_df['σp']
    display(cluster_centers_df)

    #plotting model centroide
    ax.scatter(cluster_centers_df['σp'], pred_centroide_Expr_list, marker=".", s=100, c='blue', label = 'Model Centroids')

    ax.set_title('Simulated Porfolio Clusters')
    ax.set_xlabel('Volatility(Risk)')
    ax.set_ylabel('Expected Return ')
    ax.legend(prop = { "size": 8 })
    plt.show()

    # Silhouette Score to evaluate the clustering
    sil_score = silhouette_score(clipped_df, pred_clusters)
    print(f'Silhouette Score: {sil_score}')
    return cluster_centers_df

```

```python

def plot_predicted_clusters_risk_profile_portfolio_allocation( log_returns, uncorrelated_weighted_portfolio_trails_simulation_df,
                                                              most_diversify_portfolio_assets_list, cluster_centers_df):


    print('\n                           **********************************************************************\n'+
              '                                 Investment Profile Simulation And Portfolio Allocation \n'+
              '                           **********************************************************************\n')
    for i in range(len(cluster_centers_df)):
        risk_profile = cluster_centers_df.index[i]
        portfolio_risk = cluster_centers_df['σp'][i]
        predicted_expected_return = cluster_centers_df['Pred Centroide Expr'][i]
        sharpe_ratio = cluster_centers_df['Pred Centroide Sharpe Ratio'][i]
        print('\n                                    *****************************************************\n'+
              '                                      Risk Profile                : '+risk_profile+' Investment \n'+
              '                                      Simulated Risk              : '+str(round(portfolio_risk,3))+'\n'+
              '                                      Predicted Expected Return   : '+str(round(predicted_expected_return,3))+'\n'+
              '                                      Sharpe Ratio                : '+str(round(sharpe_ratio,3))+'\n'
              '                                     *****************************************************\n')
        plot_predicted_portfolio_weight( log_returns, uncorrelated_weighted_portfolio_trails_simulation_df,
                                        most_diversify_portfolio_assets_list, portfolio_risk,risk_profile)
        plot_asset_return( log_returns, uncorrelated_weighted_portfolio_trails_simulation_df,
                          most_diversify_portfolio_assets_list, portfolio_risk,risk_profile)
```

```python
def implement_investement_profile_simulation(log_returns, uncorrelated_weighted_portfolio_trails_simulation_df,
                                             most_diversify_portfolio_assets_list, n_components):
    fig, ax =plt.subplots(1,2,figsize=(21, 5))
    clipped_df = calculate_number_of_cluster(uncorelated_weighted_portfolio_trails_simulation_df,n_components, ax[0])
    print('\n                                 *********************************************************************************\n'+
              '                                  Investement profile simulation  - Optimal Portfolio - Efficient Frontier Model \n'+
              '                                 *********************************************************************************\n')
    cluster_centers_df = implement_k_means_clusters(clipped_df, fig, ax[1])
    plot_predicted_clusters_risk_profile_portfolio_allocation( log_returns, uncorrelated_weighted_portfolio_trails_simulation_df,
                                                              most_diversify_portfolio_assets_list, cluster_centers_df)

implement_investement_profile_simulation(log_returns, uncorelated_weighted_portfolio_trails_simulation_df,
                                         most_diversify_portfolio_assets_list, 2)
```

```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```

                                 *********************************************************************************
                                  Investement profile simulation  - Optimal Portfolio - Efficient Frontier Model 
                                 *********************************************************************************


```
```
            σp      E_rp  y_E_rp_pred     error  y_optimal_E_rp  \
9088  1.518008  0.036252     0.062054  0.025803        0.036252   
9450  1.479494  0.037468     0.061014  0.023546        0.037468   
5096  1.487795  0.037854     0.061326  0.023472        0.037854   
371   1.482975  0.037747     0.061151  0.023403        0.037747   
8735  1.438078  0.035996     0.058735  0.022739        0.035996   
...        ...       ...          ...       ...             ...   
4488  1.330845  0.047093     0.047244  0.000151        0.047093   
9034  1.445931  0.059122     0.059259  0.000138        0.059122   
2486  1.409447  0.056408     0.056456  0.000048        0.056408   
2325  1.322570  0.045990     0.046022  0.000033        0.045990   
2839  1.366888  0.051998     0.052006  0.000008        0.051998   

      sharpes_ratio  cluster  
9088       0.023881        4  
9450       0.025325        1  
5096       0.025443        4  
371        0.025454        4  
8735       0.025030        0  
...             ...      ...  
4488       0.035386        2  
9034       0.040888        1  
2486       0.040021        0  
2325       0.034773        2  
2839       0.038041        2  

[9924 rows x 7 columns]
```
```
C:\Users\atsuv\AppData\Local\Temp\ipykernel_13792\1421094688.py:20: UserWarning:

No data for colormapping provided via 'c'. Parameters 'cmap' will be ignored

C:\Users\atsuv\AppData\Local\Temp\ipykernel_13792\1421094688.py:20: UserWarning:

No data for colormapping provided via 'c'. Parameters 'cmap' will be ignored

C:\Users\atsuv\AppData\Local\Temp\ipykernel_13792\1421094688.py:20: UserWarning:

No data for colormapping provided via 'c'. Parameters 'cmap' will be ignored

C:\Users\atsuv\AppData\Local\Temp\ipykernel_13792\1421094688.py:20: UserWarning:

No data for colormapping provided via 'c'. Parameters 'cmap' will be ignored

C:\Users\atsuv\AppData\Local\Temp\ipykernel_13792\1421094688.py:20: UserWarning:

No data for colormapping provided via 'c'. Parameters 'cmap' will be ignored


```
```
                          σp      E_rp  y_E_rp_pred     error  y_optimal_E_rp  \
Investment Profile                                                              
Moderate            1.422478  0.047610     0.057513  0.009903        0.047610   
Conservative        1.462226  0.048986     0.060165  0.011178        0.048986   
Agressive           1.376375  0.045750     0.052988  0.007238        0.045750   
Very Aggressive     1.558043  0.051773     0.061842  0.010068        0.051773   
Very Conservative   1.502600  0.050109     0.061699  0.011591        0.050109   

                    sharpes_ratio  Pred Centroide Expr  \
Investment Profile                                       
Moderate                 0.033469             0.057564   
Conservative             0.033502             0.060210   
Agressive                0.033237             0.053108   
Very Aggressive          0.033231             0.062033   
Very Conservative        0.033349             0.061763   

                    Pred Centroide Sharpe Ratio  
Investment Profile                               
Moderate                               0.040468  
Conservative                           0.041177  
Agressive                              0.038585  
Very Aggressive                        0.039815  
Very Conservative                      0.041104  
```
```
<Figure size 2100x500 with 3 Axes>
```
```
Silhouette Score: 0.9811219752806893

                           **********************************************************************
                                 Investment Profile Simulation And Portfolio Allocation 
                           **********************************************************************


                                    *****************************************************
                                      Risk Profile                : Moderate Investment 
                                      Simulated Risk              : 1.422
                                      Predicted Expected Return   : 0.058
                                      Sharpe Ratio                : 0.04
                                     *****************************************************


```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<pandas.io.formats.style.Styler at 0x2c5e841f850>
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```

                                    *****************************************************
                                      Risk Profile                : Conservative Investment 
                                      Simulated Risk              : 1.462
                                      Predicted Expected Return   : 0.06
                                      Sharpe Ratio                : 0.041
                                     *****************************************************


```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<pandas.io.formats.style.Styler at 0x2c5eab5cc90>
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```

                                    *****************************************************
                                      Risk Profile                : Agressive Investment 
                                      Simulated Risk              : 1.376
                                      Predicted Expected Return   : 0.053
                                      Sharpe Ratio                : 0.039
                                     *****************************************************


```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<pandas.io.formats.style.Styler at 0x2c5e9848150>
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```

                                    *****************************************************
                                      Risk Profile                : Very Aggressive Investment 
                                      Simulated Risk              : 1.558
                                      Predicted Expected Return   : 0.062
                                      Sharpe Ratio                : 0.04
                                     *****************************************************


```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<pandas.io.formats.style.Styler at 0x2c5e84dcc50>
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```

                                    *****************************************************
                                      Risk Profile                : Very Conservative Investment 
                                      Simulated Risk              : 1.503
                                      Predicted Expected Return   : 0.062
                                      Sharpe Ratio                : 0.041
                                     *****************************************************


```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<pandas.io.formats.style.Styler at 0x2c5e3df3590>
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
```
<Figure size 1200x600 with 1 Axes>
```
```
'y =0.07000 * x^2 + -0.01600 * x + -0.00900'
```
## Portfolio Stress Testing


### Macroeconomics Key Performance Indicators(KPIs) Data Collection and Preprocessing



In this section, we will use  Statistic Canads API stats-can to integrate the Canadian economic factors. We will then use Principal Components Analysis(PCA) technique to select the most importance economic factors.


```python
warnings.filterwarnings("ignore")
#                      --------------------------------------------------------------------------------------------
#                        Trade Balance: Labour force characteristics by province, monthly, seasonally adjusted
#                      --------------------------------------------------------------------------------------------

def get_trade_balance_rate(reporting_year_period, frequency_date_column ):
        frequency = frequency_date_column[0].upper()
        df = sc.table_to_df("12-10-0011-01")
        df1 = df.loc[(df['REF_DATE'] >= reporting_year_period) & (df['Trade'] =='Trade Balance') &
            (df['Principal trading partners']  == 'All countries'), ['REF_DATE','Trade','Principal trading partners','VALUE']]
        df1[frequency_date_column] = df1['REF_DATE'].dt.to_period(frequency)

        trade_balance_df = df1[[frequency_date_column, 'VALUE']]
        trade_balance_df['Trade Balance Rate'] = trade_balance_df['VALUE'].pct_change() * 100

        trade_balance_rate_df= trade_balance_df.groupby(frequency_date_column).mean()
        #trade_balance_rate_df = trade_balance_rate_df.rename(columns={'VALUE': 'Unemployment rate'})
        trade_balance_rate_df['Trade Balance Rate'] = round(trade_balance_rate_df['Trade Balance Rate'],1)
        trade_balance_rate_df = trade_balance_rate_df[['Trade Balance Rate']]
        trade_balance_rate_df = trade_balance_rate_df.dropna()
        return trade_balance_rate_df



#                      --------------------------------------------------------------------------------------------
#                        unemployment rate: Labour force characteristics by province, monthly, seasonally adjusted
#                      --------------------------------------------------------------------------------------------

def get_unemployment_rate(reporting_year_period, frequency_date_column ):
        frequency = frequency_date_column[0].upper()
        df = sc.table_to_df("14-10-0287-03")
        df1 = df.loc[(df['REF_DATE'] >= reporting_year_period) & (df['Labour force characteristics'] =='Unemployment rate') &
            (df['UOM']  == 'Percentage'), ['REF_DATE','Labour force characteristics','UOM','VALUE']]
        df1[frequency_date_column] = df1['REF_DATE'].dt.to_period(frequency)

        unemployment_rate_df = df1[[frequency_date_column, 'VALUE']]
        unemployment_rate_df= unemployment_rate_df.groupby(frequency_date_column).mean()
        unemployment_rate_df = unemployment_rate_df.rename(columns={'VALUE': 'Unemployment rate'})
        unemployment_rate_df['Unemployment rate'] = round(unemployment_rate_df['Unemployment rate'],1)
        unemployment_rate_df = unemployment_rate_df.dropna()
        return unemployment_rate_df


#                      --------------------------------------------------------------------------------------------------
#                                Financial market statistics, last Wednesday unless otherwise stated, Bank of Canada
#                      --------------------------------------------------------------------------------------------------

def get_Government_of_Canada_bonds_or_T_bill(reporting_year_period, col_value_name, rate_statement, frequency_date_column ):
    frequency = frequency_date_column[0].upper()

    df = sc.table_to_df("10-10-0122-01")
    df2 = df.loc[(df['REF_DATE'] >= reporting_year_period) & (df['UOM']  == 'Percent') &  (df['Rates'].str.contains(rate_statement)),
             ['REF_DATE','Rates','UOM','VALUE']]


    df2[frequency_date_column] = df2['REF_DATE'].dt.to_period(frequency)
    df2 = df2.dropna()
    goc_bonds_or_T_bill_df = df2[[frequency_date_column, 'VALUE']]
    #goc_bonds_or_T_bill_df['VALUE'] = round(goc_bonds_or_T_bill_df['VALUE'],1)
    goc_bonds_or_T_bill_df= goc_bonds_or_T_bill_df.groupby(frequency_date_column).mean()
    goc_bonds_or_T_bill_df= goc_bonds_or_T_bill_df.rename(columns={'VALUE': col_value_name})
    goc_bonds_or_T_bill_df[col_value_name] = round(goc_bonds_or_T_bill_df[col_value_name],1)

    return goc_bonds_or_T_bill_df

#                   ---------------------------------------------------------------------------------------------------------------------
#                            CPI Inflaction:The CPI measures the average change over time in the prices paid by urban consumers
#                            for a market basket of consumer goods and services,
#                            and it's a key indicator of inflation (ING Think) (Inflation Calculator).
#                   ---------------------------------------------------------------------------------------------------------------------


def get_CPI_inflaction_rate(reporting_year_period, frequency_date_column):
    frequency = frequency_date_column[0].upper()
    alternative_measures = 'Measure of core inflation based on a factor model, CPI-common (year-over-year percent change)'
    df = sc.table_to_df("18-10-0256-01")
    df2 = df.loc[(df['REF_DATE'] >= reporting_year_period) & (df['UOM']  == 'Percent') &
            (df['Alternative measures'] == alternative_measures),
           ['REF_DATE','Alternative measures','UOM','VALUE']]
    df2[frequency_date_column] = df2['REF_DATE'].dt.to_period(frequency)
    df2 = df2.dropna()
    CPI_inflaction_rate_df = df2[[frequency_date_column, 'VALUE']]
    CPI_inflaction_rate_df= CPI_inflaction_rate_df.groupby(frequency_date_column).mean()
    CPI_inflaction_rate_df= CPI_inflaction_rate_df.rename(columns={'VALUE': 'CPI Inflaction Rate'})
    CPI_inflaction_rate_df['CPI Inflaction Rate'] = round(CPI_inflaction_rate_df['CPI Inflaction Rate'],1)
    return CPI_inflaction_rate_df

#                                    -----------------------------------------------------------------------------------
                                                                   #morgage rate
#                                    -----------------------------------------------------------------------------------


def get_morgage_rate(reporting_year_period, frequency_date_column):
    frequency = frequency_date_column[0].upper()
    df = sc.table_to_df("34-10-0145-01")
    df1 = df.loc[(df['REF_DATE'] >= reporting_year_period), ['REF_DATE', 'UOM','VALUE']]
    df1[frequency_date_column] = df1['REF_DATE'].dt.to_period(frequency)

    get_morgage_rate_df = df1[[frequency_date_column, 'VALUE']]
    get_morgage_rate_df = get_morgage_rate_df.groupby(frequency_date_column).mean()
    get_morgage_rate_df = get_morgage_rate_df.rename(columns={'VALUE': 'Morgage Rate'})
    get_morgage_rate_df['Morgage Rate'] = round(get_morgage_rate_df['Morgage Rate'],1)
    get_morgage_rate_df = get_morgage_rate_df.dropna()
    return get_morgage_rate_df

#          -------------------------------------------------------------------------------------------------------------------------------------
#                                                                  prime rate
#                The prime interest rate is the percentage that U.S. commercial banks charge their most creditworthy customers for loans.
#                Like all loan rates, the prime interest rate is derived from the federal funds' overnight rate, set by the Federal Reserve at
#                meetings held eight times a year. The prime interest rate is the benchmark banks and other lenders
#                use when setting their interest rates for every category of loan from credit cards to car loans and mortgages.
#          -------------------------------------------------------------------------------------------------------------------------------------

def get_prime_rate(reporting_year_period, frequency_date_column):
    frequency = frequency_date_column[0].upper()
    df = sc.table_to_df("10-10-0145-01")
    df1 = df.loc[(df['REF_DATE'] >= reporting_year_period), ['REF_DATE', 'UOM','VALUE']]
    df1[frequency_date_column] = df1['REF_DATE'].dt.to_period(frequency)
    get_prime_rate_df = df1[[frequency_date_column, 'VALUE']]
    get_prime_rate_df.set_index(frequency_date_column, inplace=True)
    get_prime_rate_df = get_prime_rate_df.groupby(frequency_date_column).mean()
    get_prime_rate_df = get_prime_rate_df.rename(columns={'VALUE': 'Prime Rate'})
    get_prime_rate_df['Prime Rate'] = round(get_prime_rate_df['Prime Rate'],1)
    get_prime_rate_df = get_prime_rate_df.dropna()
    return get_prime_rate_df

#                       ----------------------------------------------------------------------------------------------------
#                                               House Price Index (house and land)
#                       ----------------------------------------------------------------------------------------------------

def get_house_price_index(reporting_year_period, frequency_date_column):
    frequency = frequency_date_column[0].upper()
    df = sc.table_to_df("18-10-0205-02")
    df1 = df.loc[(df['REF_DATE'] >= reporting_year_period) & (df['GEO'] =='Canada') &
                 (df['New housing price indexes'] =='Total (house and land)')
                 , ['REF_DATE','New housing price indexes', 'VALUE']]
    df1[frequency_date_column] = df1['REF_DATE'].dt.to_period(frequency)

    get_house_price_index_df = df1[[frequency_date_column, 'VALUE']]
    get_house_price_index_df.set_index(frequency_date_column, inplace=True)
    get_house_price_index_df = ((get_house_price_index_df / get_house_price_index_df.shift(1)) - 1)*100

    get_house_price_index_df= get_house_price_index_df.groupby(frequency_date_column).mean()
    get_house_price_index_df = get_house_price_index_df.rename(columns={'VALUE': 'House Price Index(house and land)'})
    get_house_price_index_df['House Price Index(house and land)'] = round(get_house_price_index_df[
        'House Price Index(house and land)'],1)
    get_house_price_index_df = get_house_price_index_df.dropna()
    return get_house_price_index_df.tail(60)

#                             -----------------------------------------------------------------------------------------
#                                                     Real GDP growth Seasonal adjustment
#                             -----------------------------------------------------------------------------------------

def get_Real_GDP_growth(reporting_year_period, frequency_date_column):
    frequency = frequency_date_column[0].upper()
    df = sc.table_to_df("36-10-0434-02")
    df1 = df.loc[(df['REF_DATE'] >= reporting_year_period) & (df['GEO'] =='Canada') &
                 (df['North American Industry Classification System (NAICS)'] =='All industries [T001]'),
                 ['REF_DATE','Seasonal adjustment', 'VALUE']]
    df1[frequency_date_column] = df1['REF_DATE'].dt.to_period(frequency)

    get_Real_GDP_growth_df = df1[[frequency_date_column, 'VALUE']]
    get_Real_GDP_growth_df.set_index(frequency_date_column, inplace=True)

    #get_Real_GDP_growth_df= get_Real_GDP_growth_df.groupby('MONTH_YEAR').sum()
    get_Real_GDP_growth_df= get_Real_GDP_growth_df.groupby(frequency_date_column).mean()
    get_Real_GDP_growth_df = ((get_Real_GDP_growth_df / get_Real_GDP_growth_df.shift(1)) - 1)*100
    get_Real_GDP_growth_df = get_Real_GDP_growth_df.rename(columns={'VALUE': 'Real GDP growth Seasonal adjustment'})
    get_Real_GDP_growth_df['Real GDP growth Seasonal adjustment'] = round(get_Real_GDP_growth_df[
        'Real GDP growth Seasonal adjustment'],1)
    get_Real_GDP_growth_df = get_Real_GDP_growth_df.dropna()
    return get_Real_GDP_growth_df.tail(60)

#                         ------------------------------------------------------------------------------------------------------
#                                                               Marcket Valatility


#                          oronto Stock Exchange statistics1: S&P/TSX 60 VIX Index (VIXI.TS)
#                          The S&P/TSX 60 is a market-capitalization-weighted index that tracks the performance of the 60 largest
#                          companies listed on the Toronto Stock Exchange (TSX). The S&P/TSX Composite, on the other hand,
#                          is a broader index that includes all common stocks and income trust units listed on the TSX

#                          The S&P/TSX Composite provides a more comprehensive view of the Canadian stock market.
#                          3It includes a wider range of companies, from small-cap to large-cap. This makes it a good
#                          choice for investors who want to diversify their portfolio across different sectors and market capitalizations.
#                          ://www.spglobal.com/spdji/en/indices/equity/sp-tsx-composite-index/#overview
#                          Toronto Stock Exchange statisticand  :S&P/TSX 60 VIX Index (VIXI.TS),
#                          S&P/TSX Venture Composite Index (^SPCDNX) and S&P/TSX Composite index (^GSPTSE)
#                          The S&P 500 index, or Standard & Poor’s 500, is a very important index that tracks
#                          the performance of the stocks of 500 large-cap companies in the U.S. The ticker symbol for the S&P 500 index is ^GSPC.
#                          The DJIA tracks the stock prices of 30 of the biggest American companies.
#                          The S&P 500 tracks 500 large-cap American stocks. Both offer a big-picture view of the state of the
#                          stock markets in general
#                          https://www.investopedia.com/ask/answers/difference-between-dow-jones-industrial-average-and-sp-500/#:
#                          ~:text=Key%20Takeaways,the%20stock%20markets%20in%20general.
#                        ---------------------------------------------------------------------------------------------------------------

def get_market_index_volatility(reporting_year_period, frequency_date_column, market_index_list = ['^GSPTSE', '^GSPC', '^DJI']):
    frequency = frequency_date_column[0].upper()

    start_date = reporting_year_period
    end_date = date.today()
    #index_yahoo_adj_close_price_data = yf.download(market_index_list, start_date, end_date, ['Adj Close'], period ='max')
    #market_adj_close_price_df = index_yahoo_adj_close_price_data['Adj Close']
    market_adj_close_price_df = create_adj_close_price_df(reporting_year_period, market_index_list)

    market_adj_close_price_log_return_df = np.log(market_adj_close_price_df/ market_adj_close_price_df.shift(1))
    # drop columns with all NaN's
    market_adj_close_price_log_return_df = market_adj_close_price_log_return_df.dropna(axis=0)

    #Market volatility

    market_volatility_df = market_adj_close_price_log_return_df.rolling(center=False,window= 252).std() * np.sqrt(252)
    for col in list(market_volatility_df.columns):
        market_volatility_df = market_volatility_df.rename(columns={col: 'Market '+col+' Volatility Index'})

    market_volatility_df = market_volatility_df.dropna(axis=0)

    market_volatility_df[frequency_date_column] = pd.to_datetime(market_volatility_df.index, format = '%m/%Y')
    market_volatility_df[frequency_date_column] = market_volatility_df[frequency_date_column].dt.to_period(frequency)

    #market_adj_close_price_log_return_frequency_df = market_volatility_df
    market_volatility_df.set_index(frequency_date_column, inplace=True)
    market_volatility_index_df = market_volatility_df.groupby(frequency_date_column).mean()
    market_volatility_index_df = round(market_volatility_index_df,1)
    market_volatility_index_df = market_volatility_index_df.dropna(axis=0)
    return market_volatility_index_df
    #if frequency == 'M' :
    #    return market_volatility_index_df.tail(60)
    #else:
    #    return market_volatility_index_df.tail(20)

#-------------------------------------------------------Governement of Canada Bonds average----------------------------------------------

def goc_bonds_average(reporting_year_period, frequency_date_column):
    goc_bonds_average_yield_1_3_df = get_Government_of_Canada_bonds_or_T_bill(reporting_year_period, 'GOC Marketable Bonds Average Yield: 1-3 year',
        'Government of Canada marketable bonds, average yield: 1-3 year', frequency_date_column)
    goc_bonds_average_yield_5_10_df = get_Government_of_Canada_bonds_or_T_bill(reporting_year_period, 'GOC Marketable Bonds Average Yield: 5-10 year',
        'Government of Canada marketable bonds, average yield: 5-10 year', frequency_date_column)
    goc_bonds_average_yield_3_5_df = get_Government_of_Canada_bonds_or_T_bill(reporting_year_period, 'GOC Marketable Bonds Average Yield: 3-5 year',
        'Government of Canada marketable bonds, average yield: 3-5 year', frequency_date_column)
    goc_bonds_average_yield_over_10_years_df = get_Government_of_Canada_bonds_or_T_bill(reporting_year_period,
                                                                                        'GOC Marketable Bonds Average Yield: over 10 years',
         'Government of Canada marketable bonds, average yield: over 10 years', frequency_date_column)

    goc_bonds_average_df = goc_bonds_average_yield_1_3_df.merge(goc_bonds_average_yield_5_10_df,
                                                                on= frequency_date_column, how='inner') \
                                                         .merge(goc_bonds_average_yield_3_5_df, on= frequency_date_column, how='inner') \
                                                         .merge(goc_bonds_average_yield_over_10_years_df, on= frequency_date_column, how='inner')
    return goc_bonds_average_df

#------------------------- Governement of Canada Benchmark Bonds Yield -------------------------------------------------------------------

def goc_benchmark_bonds_yield(reporting_year_period, frequency_date_column):
    goc_benchmark_bonds_yield_over_2_year_df = \
            get_Government_of_Canada_bonds_or_T_bill(reporting_year_period, 'GOC benchmark bond yields: 2 year',
                                            'Selected Government of Canada benchmark bond yields: 2 year' , frequency_date_column)
    goc_benchmark_bonds_yield_over_3_year_df = \
            get_Government_of_Canada_bonds_or_T_bill(reporting_year_period, 'GOC benchmark bond yields: 3 year',
                                                'Selected Government of Canada benchmark bond yields: 3 year', frequency_date_column)
    goc_benchmark_bonds_yield_over_5_year_df = \
            get_Government_of_Canada_bonds_or_T_bill(reporting_year_period, 'GOC benchmark bond yields: 5 year',
                                    'Selected Government of Canada benchmark bond yields: 5 year', frequency_date_column)
    goc_benchmark_bonds_yield_over_7_year_df = \
            get_Government_of_Canada_bonds_or_T_bill(reporting_year_period, 'GOC benchmark bond yields: 7 year',
                                    'Selected Government of Canada benchmark bond yields: 7 year', frequency_date_column)
    goc_benchmark_bonds_yield_over_10_years_df = \
            get_Government_of_Canada_bonds_or_T_bill(reporting_year_period, 'GOC benchmark bond yields: 10 years',
                                    'Selected Government of Canada benchmark bond yields: 10 years', frequency_date_column)
    goc_benchmark_bonds_yield_over_long_term_df = \
            get_Government_of_Canada_bonds_or_T_bill(reporting_year_period, 'GOC benchmark bond yields: long term',
                                         'Selected Government of Canada benchmark bond yields: long term', frequency_date_column)
    goc_benchmark_bonds_yield_df = \
            goc_benchmark_bonds_yield_over_2_year_df.merge(goc_benchmark_bonds_yield_over_3_year_df,
                                                       on= frequency_date_column, how='inner') \
                                .merge(goc_benchmark_bonds_yield_over_5_year_df, on= frequency_date_column, how='inner') \
                                .merge(goc_benchmark_bonds_yield_over_7_year_df, on= frequency_date_column, how='inner') \
                                .merge(goc_benchmark_bonds_yield_over_10_years_df, on= frequency_date_column, how='inner') \
                                .merge(goc_benchmark_bonds_yield_over_long_term_df, on= frequency_date_column, how='inner')

    return goc_benchmark_bonds_yield_df

 #------------------------------------------------------------Governement of Canada Treasurt Bills --------------------------------------------
def Treasury_bills(reporting_year_period, frequency_date_column):

    Treasury_bills_1_month_df = get_Government_of_Canada_bonds_or_T_bill(reporting_year_period, 'Treasury bills: 1 month',
                                'Treasury bills: 1 month', frequency_date_column)
    Treasury_bills_2_month_df = get_Government_of_Canada_bonds_or_T_bill(reporting_year_period, 'Treasury bills: 2 month',
                                'Treasury bills: 2 month', frequency_date_column)
    Treasury_bills_3_month_df = get_Government_of_Canada_bonds_or_T_bill(reporting_year_period, 'Treasury bills: 3 month',
                                 'Treasury bills: 3 month', frequency_date_column)
    Treasury_bills_6_month_df = get_Government_of_Canada_bonds_or_T_bill(reporting_year_period, 'Treasury bills: 6 month',
                                'Treasury bills: 6 month', frequency_date_column)
    Treasury_bills_1_year_df = get_Government_of_Canada_bonds_or_T_bill(reporting_year_period, 'Treasury bills: 1 year',
                                'Treasury bills: 1 year', frequency_date_column)
    Treasury_bills_df = Treasury_bills_1_month_df.merge(Treasury_bills_2_month_df, on= frequency_date_column, how='inner') \
                                            .merge(Treasury_bills_3_month_df, on= frequency_date_column, how='inner') \
                                            .merge(Treasury_bills_6_month_df, on= frequency_date_column, how='inner') \
                                            .merge(Treasury_bills_1_year_df, on=frequency_date_column, how='inner')
    return Treasury_bills_df

#-----------------------------------------  Other Economic Factors ------------------------------------------------------------------

def other_economic_factors(reporting_year_period, frequency_date_column):
    unemployment_rate_df = get_unemployment_rate(reporting_year_period, frequency_date_column)
    CPI_inflaction_rate_df = get_CPI_inflaction_rate(reporting_year_period, frequency_date_column)
    get_morgage_rate_df = get_morgage_rate(reporting_year_period, frequency_date_column)
    get_prime_rate_df = get_prime_rate(reporting_year_period, frequency_date_column)
    get_house_price_index_df = get_house_price_index(reporting_year_period, frequency_date_column)
    get_Real_GDP_growth_df = get_Real_GDP_growth(reporting_year_period, frequency_date_column)
    market_index_volatility_df = get_market_index_volatility(reporting_year_period, frequency_date_column)
    trade_balance_rate_df = get_trade_balance_rate(reporting_year_period, frequency_date_column)

    other_economic_factors_df = CPI_inflaction_rate_df.merge(get_morgage_rate_df, on= frequency_date_column, how='inner') \
                                                    .merge(get_prime_rate_df, on= frequency_date_column, how='inner') \
                                                .merge(get_house_price_index_df, on= frequency_date_column, how='inner') \
                                                .merge(unemployment_rate_df, on= frequency_date_column, how='inner') \
                                                .merge(get_Real_GDP_growth_df, on= frequency_date_column, how='inner') \
                                                .merge(market_index_volatility_df, on= frequency_date_column, how='inner')

    return other_economic_factors_df

#-----------------------------------------------------------All the Economic Factors -----------------------------------------
def get_economic_factors_df(reporting_year_period, reporting_frequency):
     #set reporting frequency

    if reporting_frequency.capitalize() == 'Month' or reporting_frequency.capitalize() == 'Quarter':
        frequency_date_column = reporting_frequency.capitalize() + '_Year'
        #frequency = reporting_frequency[0].upper()

        goc_bonds_average_df = goc_bonds_average(reporting_year_period, frequency_date_column)
        goc_benchmark_bonds_yield_df = goc_benchmark_bonds_yield(reporting_year_period, frequency_date_column)
        Treasury_bills_df = Treasury_bills(reporting_year_period, frequency_date_column)
        other_economic_factors_df = other_economic_factors(reporting_year_period, frequency_date_column)

        economic_factors_df = goc_bonds_average_df.merge(goc_benchmark_bonds_yield_df, on= frequency_date_column, how='inner') \
                                            .merge(Treasury_bills_df, on= frequency_date_column, how='inner') \
                                            .merge(other_economic_factors_df, on= frequency_date_column, how='inner')


        return economic_factors_df
    else:
        return 'The reporting frequency should be alphanbetic, Month or Qurater'

#-------------------------------------------------------------Macroeconomics factors Plotting---------------------------------------

def annotate_bars(ax):# this function is generated by ChatGPT
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy()
        ax.annotate(f'{height:.1f}', (x + width/2, y + height/2), ha='center', va='center', fontsize=10, color='black')


def get_economic_factors_barplotting(goc_bonds_average_df, goc_benchmark_bonds_yield_df,Treasury_bills_df, other_economic_factors_df ):
    fig, axes =plt.subplots(4,1,figsize=(20, 35), constrained_layout=True)


    bar_width = 0.7

    bar0 = goc_bonds_average_df.plot(kind='bar', width=bar_width, stacked=True, ax = axes[0])
    bar0.set_title('Governement of Canada Bonds Average',color='black')
    bar0.legend(loc='best')
    annotate_bars(axes[0])

    bar1 = goc_benchmark_bonds_yield_df.plot(kind='bar', width=bar_width, stacked=True, ax = axes[1])
    bar1.set_title('Governement of Canada Benchmark Bonds Yield',color='black')
    bar1.legend(loc='best')
    annotate_bars(axes[1])

    bar2 = Treasury_bills_df.plot(kind='bar', width=bar_width, stacked=True, ax = axes[2])
    bar2.set_title("Governement of Canada Treasury Bills",color='black')
    bar2.legend(loc='best')
    annotate_bars(axes[2])

    bar3 = other_economic_factors_df.plot(kind='bar', width=bar_width, stacked=True, ax = axes[3])
    bar3.set_title('Governement of Canada Other Economic Factirs',color='black')
    bar3.legend(loc='best')
    annotate_bars(axes[3])


#----------------------------Principal Components Analysis(PCA) to select most importance economic factors ---------------------------------

def selecting_importent_economic_factors_treshold_method_PCA(df,threshold):

    return df[(df.abs() > threshold).any(axis=1)].index.to_list()

def setting_PCA_for_economic_factors(economic_factors_df):
    # economic indicators dataset
   # economic_factors_df = get_economic_factors_df(reporting_year_period, reporting_frequency)

    # Standardizing the data
    scaler = StandardScaler()
    scaled_data_df = scaler.fit_transform(economic_factors_df)

    # Applying PCA
    all_pca = PCA(n_components=None)  # Use all components to find the best number of important indicators
    all_principal_components = all_pca.fit_transform(scaled_data_df)

    # Explained variance
    explained_variance = all_pca.explained_variance_ratio_

    # Principal Component Loadings(coefficients)
    loadings_matrix = all_pca.components_

    # Create a DataFrame for loadings
    loadings_matrix_df = pd.DataFrame(loadings_matrix.T, columns=[f'PC{i+1}' for i in range(loadings_matrix.shape[0])],
                                      index=economic_factors_df.columns)
    return loadings_matrix_df, explained_variance

def get_num_components(explained_variance,cumulative_variance_treshold = 0.9):
    # Determine the number of components explaining the cumulative varience treshold of the variance
    cumulative_variance = explained_variance.cumsum()
    return  (cumulative_variance <= cumulative_variance_treshold).sum() + 1

def select_top_components_df(loadings_matrix_df, num_components, threshold_for_high_loadings = 0.5):
    # Select top components
    return loadings_matrix_df.iloc[:, :num_components]

def select_top_indicators_df(loadings_matrix_df, num_components, threshold_for_high_loadings = 0.5):
    # Select top components
    selected_components_df = loadings_matrix_df.iloc[:, :num_components]
    # Find indicators with high loadings
    return selected_components_df[(selected_components_df.abs() > threshold_for_high_loadings).any(axis=1)]


def plot_explained_variance_(economic_factors_df):

    loadings_matrix_df, explained_variance =  setting_PCA_for_economic_factors(economic_factors_df)

    # Print explained variance

    explained_variance_df = pd.DataFrame(explained_variance).T
    explained_variance_df.columns = loadings_matrix_df.columns
    display(explained_variance_df)


    # Plotting the explained variance
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='individual explained variance')
    plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', label='cumulative explained variance')
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance by Principal Components')
    plt.legend(loc='best')
    plt.show()


def plotting_corr_matrix(economic_factors_matrix, title):

    g = sns.clustermap(economic_factors_matrix ,  method = 'complete', cmap   = 'RdBu', annot  = True, annot_kws = {'size': 15},figsize=(20, 15))
    g.fig.suptitle(title, y=0.9, fontsize=12)
    g.cax.set_position([1.02, 0.2, 0.03, 0.4])  # [left, bottom, width, height]
    plt.subplots_adjust(top=0.85)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    plt.setp(g.ax_heatmap.get_yticklabels(), rotation=360)

def get_most_important_economic_factors_list(economic_factors_df,
                                             cumulative_variance_treshold = 1, threshold_for_highest_loadings = 0.5):

        #plot_explained_variance_(reporting_year_period, reporting_frequency)
        #plot_explained_variance_(economic_factors_df)
        loadings_matrix_df, explained_variance = setting_PCA_for_economic_factors(economic_factors_df)
        #print('\nloadings_matrix_df\n')
        #display(loadings_matrix_df)
        num_components = get_num_components(explained_variance,cumulative_variance_treshold)
        top_components_df = select_top_components_df(loadings_matrix_df, num_components, threshold_for_highest_loadings)
        #print('\ntop_components_df\n')
        #display(top_components_df)
        #print('\ntop_indicators_df\n')
        top_indicators_df = select_top_indicators_df(loadings_matrix_df, num_components, threshold_for_highest_loadings)
        #display(top_indicators_df)
        most_important_economic_factors_list = selecting_importent_economic_factors_treshold_method_PCA(top_indicators_df,
                                                                                 threshold_for_highest_loadings)

        return most_important_economic_factors_list


def plotting_most_important_economic_factors_list(economic_factors_df,
                                             cumulative_variance_treshold = 1, threshold_for_highest_loadings = 0.5):

        #plot_explained_variance_(reporting_year_period, reporting_frequency)
        plot_explained_variance_(economic_factors_df)
        loadings_matrix_df, explained_variance = setting_PCA_for_economic_factors(economic_factors_df)
        print('\nloadings_matrix_df\n')
        display(loadings_matrix_df)
        num_components = get_num_components(explained_variance,cumulative_variance_treshold)
        top_components_df = select_top_components_df(loadings_matrix_df, num_components, threshold_for_highest_loadings)
        print('\ntop_components_df\n')
        display(top_components_df)
        print('\ntop_indicators_df\n')
        top_indicators_df = select_top_indicators_df(loadings_matrix_df, num_components, threshold_for_highest_loadings)
        display(top_indicators_df)
        most_important_economic_factors_list = selecting_importent_economic_factors_treshold_method_PCA(top_indicators_df,
                                                                                 threshold_for_highest_loadings)



def get_most_important_economic_factors_df(economic_factors_df, most_important_economic_factors_list):
        return economic_factors_df[most_important_economic_factors_list]


def get_most_important_economic_factors_matrix(most_important_economic_factors_df):
    return  generate_correlation_matrix(most_important_economic_factors_df)

def plotting_most_important_economic_factors_corr_clustermap(most_important_economic_factors_matrix):

        #PCA couple with covarience matrice to select most important factors

        plotting_corr_matrix(most_important_economic_factors_matrix,'Most Important Economic Factors Correlation Matrix Cluster Map using PCA')


#----------------------------------------------------------------------Main Data Setting-------------------------------------------
reporting_year_period = start_date(365*5)
reporting_frequency = 'Quarter'

cumulative_variance_treshold = 1.0
threshold_for_highest_loadings = 0.5
correlation_coefficient_treshold = 0.3

#Economic Factors Data Frames
goc_bonds_average_df = goc_bonds_average(reporting_year_period, reporting_frequency)
goc_benchmark_bonds_yield_df = goc_benchmark_bonds_yield(reporting_year_period, reporting_frequency)
Treasury_bills_df = Treasury_bills(reporting_year_period, reporting_frequency)
other_economic_factors_df = other_economic_factors(reporting_year_period, reporting_frequency)
trade_balance_rate_df = get_trade_balance_rate(reporting_year_period, reporting_frequency)
economic_factors_df = get_economic_factors_df(reporting_year_period, reporting_frequency)

#All the economic factors correlation matrice
economic_factors_matrix =  generate_correlation_matrix(economic_factors_df)

#Principal Components Analysis(PCA) to select  Most Important Economic Factors
most_important_economic_factors_list = get_most_important_economic_factors_list(economic_factors_df,  cumulative_variance_treshold,
                                                                                threshold_for_highest_loadings)
most_important_economic_factors_df = get_most_important_economic_factors_df(economic_factors_df, most_important_economic_factors_list)
most_important_economic_factors_matrix = get_most_important_economic_factors_matrix(most_important_economic_factors_df)

#-------------------------------------------------Data Visualization------------------------------------------------------------------
def print_economic_factors_data_table():
    print('\n                             **********************************************************\n'+
             '                              All the Economic Factors Data Tables\n'+
              '                         *********************************************************\n')

    display(economic_factors_df)
    get_economic_factors_barplotting(goc_bonds_average_df, goc_benchmark_bonds_yield_df,Treasury_bills_df, other_economic_factors_df )


def print_economic_factors_data_corr_matrix():
    print('\n                             **********************************************************\n'+
             '                              All the Economic Factors Correlation Matrix\n'+
              '                         *********************************************************\n')

    display(economic_factors_matrix)
    plotting_corr_matrix(economic_factors_matrix, 'All the economic factors')

def print_most_important_economic_factors():

    print('\n                        *****************************************************************************************\n'+
             '                               Principal Components Analysis(PCA) to select  Most Important Economic Factors \n'+
              '                          ****************************************************************************************\n')

    print('Principal Components Analysis(PCA) to select  Most Important Economic Factors \n')
    plotting_most_important_economic_factors_list(economic_factors_df, cumulative_variance_treshold, threshold_for_highest_loadings)
    print('\n most_important_economic_factors_df\n')
    display(most_important_economic_factors_df)
    print('\n most_important_economic_factors_matrix\n')
    display(most_important_economic_factors_matrix)
    plotting_corr_matrix(most_important_economic_factors_matrix, 'Most Important Economic Factors correlation Matrix - PCA Method')

```

```
[*********************100%%**********************]  3 of 3 completed
[*********************100%%**********************]  3 of 3 completed

```
#### Data Visualization


```python
print_economic_factors_data_table()
```

```

                             **********************************************************
                              All the Economic Factors Data Tables
                         *********************************************************


```
```
              GOC Marketable Bonds Average Yield: 1-3 year  \
Quarter_Year                                                 
2020Q1                                                 1.2   
2020Q2                                                 0.3   
2020Q3                                                 0.2   
2020Q4                                                 0.2   
2021Q1                                                 0.2   
2021Q2                                                 0.3   
2021Q3                                                 0.4   
2021Q4                                                 1.0   
2022Q1                                                 1.6   
2022Q2                                                 2.7   
2022Q3                                                 3.5   
2022Q4                                                 3.9   
2023Q1                                                 3.9   
2023Q2                                                 4.1   
2023Q3                                                 4.8   
2023Q4                                                 4.3   
2024Q1                                                 4.2   
2024Q2                                                 4.3   

              GOC Marketable Bonds Average Yield: 5-10 year  \
Quarter_Year                                                  
2020Q1                                                  1.1   
2020Q2                                                  0.5   
2020Q3                                                  0.5   
2020Q4                                                  0.6   
2021Q1                                                  1.0   
2021Q2                                                  1.3   
2021Q3                                                  1.2   
2021Q4                                                  1.5   
2022Q1                                                  2.0   
2022Q2                                                  2.9   
2022Q3                                                  3.0   
2022Q4                                                  3.2   
2023Q1                                                  3.0   
2023Q2                                                  3.1   
2023Q3                                                  3.8   
2023Q4                                                  3.6   
2024Q1                                                  3.4   
2024Q2                                                  3.7   

              GOC Marketable Bonds Average Yield: 3-5 year  \
Quarter_Year                                                 
2020Q1                                                 1.1   
2020Q2                                                 0.4   
2020Q3                                                 0.3   
2020Q4                                                 0.4   
2021Q1                                                 0.5   
2021Q2                                                 0.8   
2021Q3                                                 0.8   
2021Q4                                                 1.3   
2022Q1                                                 1.9   
2022Q2                                                 2.8   
2022Q3                                                 3.2   
2022Q4                                                 3.5   
2023Q1                                                 3.3   
2023Q2                                                 3.4   
2023Q3                                                 4.1   
2023Q4                                                 3.7   
2024Q1                                                 3.6   
2024Q2                                                 3.8   

              GOC Marketable Bonds Average Yield: over 10 years  \
Quarter_Year                                                      
2020Q1                                                      1.3   
2020Q2                                                      1.0   
2020Q3                                                      1.0   
2020Q4                                                      1.1   
2021Q1                                                      1.7   
2021Q2                                                      1.9   
2021Q3                                                      1.8   
2021Q4                                                      1.9   
2022Q1                                                      2.3   
2022Q2                                                      3.0   
2022Q3                                                      3.0   
2022Q4                                                      3.3   
2023Q1                                                      3.1   
2023Q2                                                      3.1   
2023Q3                                                      3.6   
2023Q4                                                      3.4   
2024Q1                                                      3.4   
2024Q2                                                      3.6   

              GOC benchmark bond yields: 2 year  \
Quarter_Year                                      
2020Q1                                      1.1   
2020Q2                                      0.3   
2020Q3                                      0.3   
2020Q4                                      0.2   
2021Q1                                      0.2   
2021Q2                                      0.4   
2021Q3                                      0.5   
2021Q4                                      1.0   
2022Q1                                      1.7   
2022Q2                                      2.7   
2022Q3                                      3.5   
2022Q4                                      3.9   
2023Q1                                      3.8   
2023Q2                                      4.1   
2023Q3                                      4.8   
2023Q4                                      4.3   
2024Q1                                      4.1   
2024Q2                                      4.2   

              GOC benchmark bond yields: 3 year  \
Quarter_Year                                      
2020Q1                                      1.1   
2020Q2                                      0.3   
2020Q3                                      0.3   
2020Q4                                      0.3   
2021Q1                                      0.3   
2021Q2                                      0.5   
2021Q3                                      0.6   
2021Q4                                      1.1   
2022Q1                                      1.8   
2022Q2                                      2.8   
2022Q3                                      3.4   
2022Q4                                      3.7   
2023Q1                                      3.6   
2023Q2                                      3.8   
2023Q3                                      4.5   
2023Q4                                      4.1   
2024Q1                                      3.9   
2024Q2                                      4.1   

              GOC benchmark bond yields: 5 year  \
Quarter_Year                                      
2020Q1                                      1.1   
2020Q2                                      0.4   
2020Q3                                      0.4   
2020Q4                                      0.4   
2021Q1                                      0.7   
2021Q2                                      0.9   
2021Q3                                      0.9   
2021Q4                                      1.4   
2022Q1                                      2.0   
2022Q2                                      2.8   
2022Q3                                      3.1   
2022Q4                                      3.3   
2023Q1                                      3.2   
2023Q2                                      3.3   
2023Q3                                      4.0   
2023Q4                                      3.7   
2024Q1                                      3.5   
2024Q2                                      3.7   

              GOC benchmark bond yields: 7 year  \
Quarter_Year                                      
2020Q1                                      1.1   
2020Q2                                      0.4   
2020Q3                                      0.4   
2020Q4                                      0.5   
2021Q1                                      0.9   
2021Q2                                      1.2   
2021Q3                                      1.1   
2021Q4                                      1.5   
2022Q1                                      2.0   
2022Q2                                      2.8   
2022Q3                                      3.0   
2022Q4                                      3.1   
2023Q1                                      3.0   
2023Q2                                      3.1   
2023Q3                                      3.8   
2023Q4                                      3.6   
2024Q1                                      3.4   
2024Q2                                      3.7   

              GOC benchmark bond yields: 10 years  \
Quarter_Year                                        
2020Q1                                        1.1   
2020Q2                                        0.5   
2020Q3                                        0.6   
2020Q4                                        0.7   
2021Q1                                        1.2   
2021Q2                                        1.5   
2021Q3                                        1.3   
2021Q4                                        1.6   
2022Q1                                        2.1   
2022Q2                                        2.9   
2022Q3                                        3.0   
2022Q4                                        3.2   
2023Q1                                        3.0   
2023Q2                                        3.1   
2023Q3                                        3.7   
2023Q4                                        3.6   
2024Q1                                        3.4   
2024Q2                                        3.7   

              GOC benchmark bond yields: long term  ...  \
Quarter_Year                                        ...   
2020Q1                                         1.4  ...   
2020Q2                                         1.1  ...   
2020Q3                                         1.1  ...   
2020Q4                                         1.2  ...   
2021Q1                                         1.8  ...   
2021Q2                                         2.0  ...   
2021Q3                                         1.8  ...   
2021Q4                                         1.9  ...   
2022Q1                                         2.2  ...   
2022Q2                                         2.9  ...   
2022Q3                                         2.9  ...   
2022Q4                                         3.2  ...   
2023Q1                                         3.1  ...   
2023Q2                                         3.1  ...   
2023Q3                                         3.5  ...   
2023Q4                                         3.4  ...   
2024Q1                                         3.3  ...   
2024Q2                                         3.6  ...   

              Treasury bills: 2 month  Treasury bills: 3 month  \
Quarter_Year                                                     
2020Q1                            1.3                      1.2   
2020Q2                            0.2                      0.2   
2020Q3                            0.2                      0.1   
2020Q4                            0.1                      0.1   
2021Q1                            0.1                      0.1   
2021Q2                            0.1                      0.1   
2021Q3                            0.2                      0.2   
2021Q4                            0.1                      0.1   
2022Q1                            0.3                      0.4   
2022Q2                            1.4                      1.6   
2022Q3                            2.9                      3.1   
2022Q4                            3.9                      4.1   
2023Q1                            4.4                      4.4   
2023Q2                            4.6                      4.6   
2023Q3                            5.0                      5.0   
2023Q4                            5.0                      5.0   
2024Q1                            5.0                      5.0   
2024Q2                            4.8                      4.8   

              Treasury bills: 6 month  Treasury bills: 1 year  \
Quarter_Year                                                    
2020Q1                            1.2                     1.2   
2020Q2                            0.3                     0.3   
2020Q3                            0.2                     0.2   
2020Q4                            0.1                     0.2   
2021Q1                            0.1                     0.1   
2021Q2                            0.2                     0.2   
2021Q3                            0.2                     0.3   
2021Q4                            0.3                     0.7   
2022Q1                            0.9                     1.4   
2022Q2                            2.1                     2.6   
2022Q3                            3.4                     3.7   
2022Q4                            4.2                     4.4   
2023Q1                            4.4                     4.4   
2023Q2                            4.7                     4.7   
2023Q3                            5.1                     5.2   
2023Q4                            5.0                     4.8   
2024Q1                            4.9                     4.8   
2024Q2                            4.8                     4.6   

              CPI Inflaction Rate  Morgage Rate  Prime Rate  \
Quarter_Year                                                  
2020Q1                        2.0           4.0         1.8   
2020Q2                        1.6           3.9         1.2   
2020Q3                        1.4           3.6         1.1   
2020Q4                        1.7           3.4         1.0   
2021Q1                        1.7           3.3         1.1   
2021Q2                        2.4           3.3         1.2   
2021Q3                        2.9           3.2         1.2   
2021Q4                        3.1           3.4         1.3   
2022Q1                        4.0           3.6         1.6   
2022Q2                        5.3           4.6         2.5   
2022Q3                        5.8           5.6         3.2   
2022Q4                        6.0           5.8         3.7   
2023Q1                        5.9           5.8         3.9   
2023Q2                        5.3           5.8         4.0   
2023Q3                        4.6           6.1         4.3   
2023Q4                        4.0           6.4         4.3   
2024Q1                        3.1           6.2         4.1   
2024Q2                        2.5           6.1         4.1   

              House Price Index(house and land)  Unemployment rate  \
Quarter_Year                                                         
2020Q1                                      0.2                4.6   
2020Q2                                      0.1                7.8   
2020Q3                                      0.7                5.9   
2020Q4                                      0.5                5.2   
2021Q1                                      1.2                5.5   
2021Q2                                      1.3                5.1   
2021Q3                                      0.5                4.6   
2021Q4                                      0.6                4.1   
2022Q1                                      1.1                4.3   
2022Q2                                      0.3                3.7   
2022Q3                                      0.0                3.6   
2022Q4                                     -0.1                3.5   
2023Q1                                     -0.2                3.8   
2023Q2                                      0.0                3.7   
2023Q3                                     -0.1                3.7   
2023Q4                                     -0.1                3.6   
2024Q1                                      0.0                4.0   
2024Q2                                      0.2                4.0   

              Real GDP growth Seasonal adjustment  
Quarter_Year                                       
2020Q1                                       -2.1  
2020Q2                                      -10.6  
2020Q3                                        8.9  
2020Q4                                        2.1  
2021Q1                                        1.2  
2021Q2                                       -0.1  
2021Q3                                        1.6  
2021Q4                                        1.6  
2022Q1                                        0.8  
2022Q2                                        1.1  
2022Q3                                        0.5  
2022Q4                                       -0.0  
2023Q1                                        0.6  
2023Q2                                        0.2  
2023Q3                                       -0.1  
2023Q4                                        0.1  
2024Q1                                        0.5  
2024Q2                                        0.4  

[18 rows x 21 columns]
```
```
<Figure size 2000x3500 with 4 Axes>
```
```python
print_economic_factors_data_corr_matrix()
```

```

                             **********************************************************
                              All the Economic Factors Correlation Matrix
                         *********************************************************


```
```
                                                   GOC Marketable Bonds Average Yield: 1-3 year  \
GOC Marketable Bonds Average Yield: 1-3 year                                           1.000000   
GOC Marketable Bonds Average Yield: 5-10 year                                          0.976199   
GOC Marketable Bonds Average Yield: 3-5 year                                           0.993209   
GOC Marketable Bonds Average Yield: over 10 years                                      0.951231   
GOC benchmark bond yields: 2 year                                                      0.999462   
GOC benchmark bond yields: 3 year                                                      0.998315   
GOC benchmark bond yields: 5 year                                                      0.989338   
GOC benchmark bond yields: 7 year                                                      0.977915   
GOC benchmark bond yields: 10 years                                                    0.967320   
GOC benchmark bond yields: long term                                                   0.952575   
Treasury bills: 1 month                                                                0.958881   
Treasury bills: 2 month                                                                0.966132   
Treasury bills: 3 month                                                                0.973636   
Treasury bills: 6 month                                                                0.987148   
Treasury bills: 1 year                                                                 0.996983   
CPI Inflaction Rate                                                                    0.725500   
Morgage Rate                                                                           0.972380   
Prime Rate                                                                             0.989560   
House Price Index(house and land)                                                     -0.744187   
Unemployment rate                                                                     -0.746375   
Real GDP growth Seasonal adjustment                                                   -0.031706   

                                                   GOC Marketable Bonds Average Yield: 5-10 year  \
GOC Marketable Bonds Average Yield: 1-3 year                                            0.976199   
GOC Marketable Bonds Average Yield: 5-10 year                                           1.000000   
GOC Marketable Bonds Average Yield: 3-5 year                                            0.992777   
GOC Marketable Bonds Average Yield: over 10 years                                       0.992448   
GOC benchmark bond yields: 2 year                                                       0.978538   
GOC benchmark bond yields: 3 year                                                       0.985369   
GOC benchmark bond yields: 5 year                                                       0.995928   
GOC benchmark bond yields: 7 year                                                       0.999459   
GOC benchmark bond yields: 10 years                                                     0.998761   
GOC benchmark bond yields: long term                                                    0.991043   
Treasury bills: 1 month                                                                 0.903818   
Treasury bills: 2 month                                                                 0.913679   
Treasury bills: 3 month                                                                 0.925168   
Treasury bills: 6 month                                                                 0.945192   
Treasury bills: 1 year                                                                  0.962425   
CPI Inflaction Rate                                                                     0.739110   
Morgage Rate                                                                            0.920224   
Prime Rate                                                                              0.952833   
House Price Index(house and land)                                                      -0.625150   
Unemployment rate                                                                      -0.802838   
Real GDP growth Seasonal adjustment                                                     0.009622   

                                                   GOC Marketable Bonds Average Yield: 3-5 year  \
GOC Marketable Bonds Average Yield: 1-3 year                                           0.993209   
GOC Marketable Bonds Average Yield: 5-10 year                                          0.992777   
GOC Marketable Bonds Average Yield: 3-5 year                                           1.000000   
GOC Marketable Bonds Average Yield: over 10 years                                      0.975615   
GOC benchmark bond yields: 2 year                                                      0.994331   
GOC benchmark bond yields: 3 year                                                      0.997760   
GOC benchmark bond yields: 5 year                                                      0.998818   
GOC benchmark bond yields: 7 year                                                      0.993509   
GOC benchmark bond yields: 10 years                                                    0.987007   
GOC benchmark bond yields: long term                                                   0.973534   
Treasury bills: 1 month                                                                0.927301   
Treasury bills: 2 month                                                                0.936388   
Treasury bills: 3 month                                                                0.947076   
Treasury bills: 6 month                                                                0.966219   
Treasury bills: 1 year                                                                 0.983328   
CPI Inflaction Rate                                                                    0.756007   
Morgage Rate                                                                           0.946726   
Prime Rate                                                                             0.971334   
House Price Index(house and land)                                                     -0.693431   
Unemployment rate                                                                     -0.785146   
Real GDP growth Seasonal adjustment                                                   -0.017584   

                                                   GOC Marketable Bonds Average Yield: over 10 years  \
GOC Marketable Bonds Average Yield: 1-3 year                                                0.951231   
GOC Marketable Bonds Average Yield: 5-10 year                                               0.992448   
GOC Marketable Bonds Average Yield: 3-5 year                                                0.975615   
GOC Marketable Bonds Average Yield: over 10 years                                           1.000000   
GOC benchmark bond yields: 2 year                                                           0.954739   
GOC benchmark bond yields: 3 year                                                           0.963174   
GOC benchmark bond yields: 5 year                                                           0.980846   
GOC benchmark bond yields: 7 year                                                           0.989952   
GOC benchmark bond yields: 10 years                                                         0.995923   
GOC benchmark bond yields: long term                                                        0.998236   
Treasury bills: 1 month                                                                     0.876058   
Treasury bills: 2 month                                                                     0.885694   
Treasury bills: 3 month                                                                     0.898744   
Treasury bills: 6 month                                                                     0.918627   
Treasury bills: 1 year                                                                      0.937023   
CPI Inflaction Rate                                                                         0.746162   
Morgage Rate                                                                                0.886569   
Prime Rate                                                                                  0.927391   
House Price Index(house and land)                                                          -0.561822   
Unemployment rate                                                                          -0.805263   
Real GDP growth Seasonal adjustment                                                         0.029892   

                                                   GOC benchmark bond yields: 2 year  \
GOC Marketable Bonds Average Yield: 1-3 year                                0.999462   
GOC Marketable Bonds Average Yield: 5-10 year                               0.978538   
GOC Marketable Bonds Average Yield: 3-5 year                                0.994331   
GOC Marketable Bonds Average Yield: over 10 years                           0.954739   
GOC benchmark bond yields: 2 year                                           1.000000   
GOC benchmark bond yields: 3 year                                           0.998947   
GOC benchmark bond yields: 5 year                                           0.990991   
GOC benchmark bond yields: 7 year                                           0.979915   
GOC benchmark bond yields: 10 years                                         0.970160   
GOC benchmark bond yields: long term                                        0.955397   
Treasury bills: 1 month                                                     0.955022   
Treasury bills: 2 month                                                     0.962590   
Treasury bills: 3 month                                                     0.970533   
Treasury bills: 6 month                                                     0.985163   
Treasury bills: 1 year                                                      0.996038   
CPI Inflaction Rate                                                         0.734083   
Morgage Rate                                                                0.969309   
Prime Rate                                                                  0.987723   
House Price Index(house and land)                                          -0.735073   
Unemployment rate                                                          -0.750504   
Real GDP growth Seasonal adjustment                                        -0.021169   

                                                   GOC benchmark bond yields: 3 year  \
GOC Marketable Bonds Average Yield: 1-3 year                                0.998315   
GOC Marketable Bonds Average Yield: 5-10 year                               0.985369   
GOC Marketable Bonds Average Yield: 3-5 year                                0.997760   
GOC Marketable Bonds Average Yield: over 10 years                           0.963174   
GOC benchmark bond yields: 2 year                                           0.998947   
GOC benchmark bond yields: 3 year                                           1.000000   
GOC benchmark bond yields: 5 year                                           0.995408   
GOC benchmark bond yields: 7 year                                           0.986585   
GOC benchmark bond yields: 10 years                                         0.977878   
GOC benchmark bond yields: long term                                        0.962796   
Treasury bills: 1 month                                                     0.944663   
Treasury bills: 2 month                                                     0.953059   
Treasury bills: 3 month                                                     0.962051   
Treasury bills: 6 month                                                     0.978696   
Treasury bills: 1 year                                                      0.992004   
CPI Inflaction Rate                                                         0.740256   
Morgage Rate                                                                0.962145   
Prime Rate                                                                  0.982265   
House Price Index(house and land)                                          -0.720636   
Unemployment rate                                                          -0.765741   
Real GDP growth Seasonal adjustment                                        -0.014608   

                                                   GOC benchmark bond yields: 5 year  \
GOC Marketable Bonds Average Yield: 1-3 year                                0.989338   
GOC Marketable Bonds Average Yield: 5-10 year                               0.995928   
GOC Marketable Bonds Average Yield: 3-5 year                                0.998818   
GOC Marketable Bonds Average Yield: over 10 years                           0.980846   
GOC benchmark bond yields: 2 year                                           0.990991   
GOC benchmark bond yields: 3 year                                           0.995408   
GOC benchmark bond yields: 5 year                                           1.000000   
GOC benchmark bond yields: 7 year                                           0.996909   
GOC benchmark bond yields: 10 years                                         0.991605   
GOC benchmark bond yields: long term                                        0.978755   
Treasury bills: 1 month                                                     0.919827   
Treasury bills: 2 month                                                     0.929439   
Treasury bills: 3 month                                                     0.940004   
Treasury bills: 6 month                                                     0.960019   
Treasury bills: 1 year                                                      0.977644   
CPI Inflaction Rate                                                         0.751887   
Morgage Rate                                                                0.937894   
Prime Rate                                                                  0.966160   
House Price Index(house and land)                                          -0.668349   
Unemployment rate                                                          -0.793680   
Real GDP growth Seasonal adjustment                                        -0.001892   

                                                   GOC benchmark bond yields: 7 year  \
GOC Marketable Bonds Average Yield: 1-3 year                                0.977915   
GOC Marketable Bonds Average Yield: 5-10 year                               0.999459   
GOC Marketable Bonds Average Yield: 3-5 year                                0.993509   
GOC Marketable Bonds Average Yield: over 10 years                           0.989952   
GOC benchmark bond yields: 2 year                                           0.979915   
GOC benchmark bond yields: 3 year                                           0.986585   
GOC benchmark bond yields: 5 year                                           0.996909   
GOC benchmark bond yields: 7 year                                           1.000000   
GOC benchmark bond yields: 10 years                                         0.997936   
GOC benchmark bond yields: long term                                        0.988589   
Treasury bills: 1 month                                                     0.906120   
Treasury bills: 2 month                                                     0.915759   
Treasury bills: 3 month                                                     0.926639   
Treasury bills: 6 month                                                     0.946566   
Treasury bills: 1 year                                                      0.963724   
CPI Inflaction Rate                                                         0.735470   
Morgage Rate                                                                0.920864   
Prime Rate                                                                  0.954083   
House Price Index(house and land)                                          -0.626895   
Unemployment rate                                                          -0.806113   
Real GDP growth Seasonal adjustment                                         0.007730   

                                                   GOC benchmark bond yields: 10 years  \
GOC Marketable Bonds Average Yield: 1-3 year                                  0.967320   
GOC Marketable Bonds Average Yield: 5-10 year                                 0.998761   
GOC Marketable Bonds Average Yield: 3-5 year                                  0.987007   
GOC Marketable Bonds Average Yield: over 10 years                             0.995923   
GOC benchmark bond yields: 2 year                                             0.970160   
GOC benchmark bond yields: 3 year                                             0.977878   
GOC benchmark bond yields: 5 year                                             0.991605   
GOC benchmark bond yields: 7 year                                             0.997936   
GOC benchmark bond yields: 10 years                                           1.000000   
GOC benchmark bond yields: long term                                          0.994981   
Treasury bills: 1 month                                                       0.894311   
Treasury bills: 2 month                                                       0.904050   
Treasury bills: 3 month                                                       0.915834   
Treasury bills: 6 month                                                       0.935894   
Treasury bills: 1 year                                                        0.953085   
CPI Inflaction Rate                                                           0.736259   
Morgage Rate                                                                  0.907991   
Prime Rate                                                                    0.943930   
House Price Index(house and land)                                            -0.591373   
Unemployment rate                                                            -0.811841   
Real GDP growth Seasonal adjustment                                           0.030765   

                                                   GOC benchmark bond yields: long term  \
GOC Marketable Bonds Average Yield: 1-3 year                                   0.952575   
GOC Marketable Bonds Average Yield: 5-10 year                                  0.991043   
GOC Marketable Bonds Average Yield: 3-5 year                                   0.973534   
GOC Marketable Bonds Average Yield: over 10 years                              0.998236   
GOC benchmark bond yields: 2 year                                              0.955397   
GOC benchmark bond yields: 3 year                                              0.962796   
GOC benchmark bond yields: 5 year                                              0.978755   
GOC benchmark bond yields: 7 year                                              0.988589   
GOC benchmark bond yields: 10 years                                            0.994981   
GOC benchmark bond yields: long term                                           1.000000   
Treasury bills: 1 month                                                        0.889460   
Treasury bills: 2 month                                                        0.898326   
Treasury bills: 3 month                                                        0.909881   
Treasury bills: 6 month                                                        0.926861   
Treasury bills: 1 year                                                         0.940557   
CPI Inflaction Rate                                                            0.726525   
Morgage Rate                                                                   0.894927   
Prime Rate                                                                     0.935540   
House Price Index(house and land)                                             -0.562839   
Unemployment rate                                                             -0.794732   
Real GDP growth Seasonal adjustment                                            0.025609   

                                                   ...  \
GOC Marketable Bonds Average Yield: 1-3 year       ...   
GOC Marketable Bonds Average Yield: 5-10 year      ...   
GOC Marketable Bonds Average Yield: 3-5 year       ...   
GOC Marketable Bonds Average Yield: over 10 years  ...   
GOC benchmark bond yields: 2 year                  ...   
GOC benchmark bond yields: 3 year                  ...   
GOC benchmark bond yields: 5 year                  ...   
GOC benchmark bond yields: 7 year                  ...   
GOC benchmark bond yields: 10 years                ...   
GOC benchmark bond yields: long term               ...   
Treasury bills: 1 month                            ...   
Treasury bills: 2 month                            ...   
Treasury bills: 3 month                            ...   
Treasury bills: 6 month                            ...   
Treasury bills: 1 year                             ...   
CPI Inflaction Rate                                ...   
Morgage Rate                                       ...   
Prime Rate                                         ...   
House Price Index(house and land)                  ...   
Unemployment rate                                  ...   
Real GDP growth Seasonal adjustment                ...   

                                                   Treasury bills: 2 month  \
GOC Marketable Bonds Average Yield: 1-3 year                      0.966132   
GOC Marketable Bonds Average Yield: 5-10 year                     0.913679   
GOC Marketable Bonds Average Yield: 3-5 year                      0.936388   
GOC Marketable Bonds Average Yield: over 10 years                 0.885694   
GOC benchmark bond yields: 2 year                                 0.962590   
GOC benchmark bond yields: 3 year                                 0.953059   
GOC benchmark bond yields: 5 year                                 0.929439   
GOC benchmark bond yields: 7 year                                 0.915759   
GOC benchmark bond yields: 10 years                               0.904050   
GOC benchmark bond yields: long term                              0.898326   
Treasury bills: 1 month                                           0.999360   
Treasury bills: 2 month                                           1.000000   
Treasury bills: 3 month                                           0.999162   
Treasury bills: 6 month                                           0.994025   
Treasury bills: 1 year                                            0.980071   
CPI Inflaction Rate                                               0.587856   
Morgage Rate                                                      0.982410   
Prime Rate                                                        0.991599   
House Price Index(house and land)                                -0.768520   
Unemployment rate                                                -0.644867   
Real GDP growth Seasonal adjustment                              -0.045324   

                                                   Treasury bills: 3 month  \
GOC Marketable Bonds Average Yield: 1-3 year                      0.973636   
GOC Marketable Bonds Average Yield: 5-10 year                     0.925168   
GOC Marketable Bonds Average Yield: 3-5 year                      0.947076   
GOC Marketable Bonds Average Yield: over 10 years                 0.898744   
GOC benchmark bond yields: 2 year                                 0.970533   
GOC benchmark bond yields: 3 year                                 0.962051   
GOC benchmark bond yields: 5 year                                 0.940004   
GOC benchmark bond yields: 7 year                                 0.926639   
GOC benchmark bond yields: 10 years                               0.915834   
GOC benchmark bond yields: long term                              0.909881   
Treasury bills: 1 month                                           0.997743   
Treasury bills: 2 month                                           0.999162   
Treasury bills: 3 month                                           1.000000   
Treasury bills: 6 month                                           0.997109   
Treasury bills: 1 year                                            0.986143   
CPI Inflaction Rate                                               0.612922   
Morgage Rate                                                      0.986454   
Prime Rate                                                        0.995128   
House Price Index(house and land)                                -0.770248   
Unemployment rate                                                -0.658274   
Real GDP growth Seasonal adjustment                              -0.048666   

                                                   Treasury bills: 6 month  \
GOC Marketable Bonds Average Yield: 1-3 year                      0.987148   
GOC Marketable Bonds Average Yield: 5-10 year                     0.945192   
GOC Marketable Bonds Average Yield: 3-5 year                      0.966219   
GOC Marketable Bonds Average Yield: over 10 years                 0.918627   
GOC benchmark bond yields: 2 year                                 0.985163   
GOC benchmark bond yields: 3 year                                 0.978696   
GOC benchmark bond yields: 5 year                                 0.960019   
GOC benchmark bond yields: 7 year                                 0.946566   
GOC benchmark bond yields: 10 years                               0.935894   
GOC benchmark bond yields: long term                              0.926861   
Treasury bills: 1 month                                           0.990470   
Treasury bills: 2 month                                           0.994025   
Treasury bills: 3 month                                           0.997109   
Treasury bills: 6 month                                           1.000000   
Treasury bills: 1 year                                            0.995380   
CPI Inflaction Rate                                               0.654033   
Morgage Rate                                                      0.989194   
Prime Rate                                                        0.999187   
House Price Index(house and land)                                -0.764929   
Unemployment rate                                                -0.682239   
Real GDP growth Seasonal adjustment                              -0.045914   

                                                   Treasury bills: 1 year  \
GOC Marketable Bonds Average Yield: 1-3 year                     0.996983   
GOC Marketable Bonds Average Yield: 5-10 year                    0.962425   
GOC Marketable Bonds Average Yield: 3-5 year                     0.983328   
GOC Marketable Bonds Average Yield: over 10 years                0.937023   
GOC benchmark bond yields: 2 year                                0.996038   
GOC benchmark bond yields: 3 year                                0.992004   
GOC benchmark bond yields: 5 year                                0.977644   
GOC benchmark bond yields: 7 year                                0.963724   
GOC benchmark bond yields: 10 years                              0.953085   
GOC benchmark bond yields: long term                             0.940557   
Treasury bills: 1 month                                          0.974308   
Treasury bills: 2 month                                          0.980071   
Treasury bills: 3 month                                          0.986143   
Treasury bills: 6 month                                          0.995380   
Treasury bills: 1 year                                           1.000000   
CPI Inflaction Rate                                              0.709570   
Morgage Rate                                                     0.982448   
Prime Rate                                                       0.995893   
House Price Index(house and land)                               -0.761639   
Unemployment rate                                               -0.720879   
Real GDP growth Seasonal adjustment                             -0.036478   

                                                   CPI Inflaction Rate  \
GOC Marketable Bonds Average Yield: 1-3 year                  0.725500   
GOC Marketable Bonds Average Yield: 5-10 year                 0.739110   
GOC Marketable Bonds Average Yield: 3-5 year                  0.756007   
GOC Marketable Bonds Average Yield: over 10 years             0.746162   
GOC benchmark bond yields: 2 year                             0.734083   
GOC benchmark bond yields: 3 year                             0.740256   
GOC benchmark bond yields: 5 year                             0.751887   
GOC benchmark bond yields: 7 year                             0.735470   
GOC benchmark bond yields: 10 years                           0.736259   
GOC benchmark bond yields: long term                          0.726525   
Treasury bills: 1 month                                       0.569049   
Treasury bills: 2 month                                       0.587856   
Treasury bills: 3 month                                       0.612922   
Treasury bills: 6 month                                       0.654033   
Treasury bills: 1 year                                        0.709570   
CPI Inflaction Rate                                           1.000000   
Morgage Rate                                                  0.629513   
Prime Rate                                                    0.666817   
House Price Index(house and land)                            -0.541372   
Unemployment rate                                            -0.751794   
Real GDP growth Seasonal adjustment                           0.017866   

                                                   Morgage Rate  Prime Rate  \
GOC Marketable Bonds Average Yield: 1-3 year           0.972380    0.989560   
GOC Marketable Bonds Average Yield: 5-10 year          0.920224    0.952833   
GOC Marketable Bonds Average Yield: 3-5 year           0.946726    0.971334   
GOC Marketable Bonds Average Yield: over 10 years      0.886569    0.927391   
GOC benchmark bond yields: 2 year                      0.969309    0.987723   
GOC benchmark bond yields: 3 year                      0.962145    0.982265   
GOC benchmark bond yields: 5 year                      0.937894    0.966160   
GOC benchmark bond yields: 7 year                      0.920864    0.954083   
GOC benchmark bond yields: 10 years                    0.907991    0.943930   
GOC benchmark bond yields: long term                   0.894927    0.935540   
Treasury bills: 1 month                                0.978951    0.987231   
Treasury bills: 2 month                                0.982410    0.991599   
Treasury bills: 3 month                                0.986454    0.995128   
Treasury bills: 6 month                                0.989194    0.999187   
Treasury bills: 1 year                                 0.982448    0.995893   
CPI Inflaction Rate                                    0.629513    0.666817   
Morgage Rate                                           1.000000    0.987620   
Prime Rate                                             0.987620    1.000000   
House Price Index(house and land)                     -0.805028   -0.762774   
Unemployment rate                                     -0.617461   -0.694479   
Real GDP growth Seasonal adjustment                   -0.090534   -0.048387   

                                                   House Price Index(house and land)  \
GOC Marketable Bonds Average Yield: 1-3 year                               -0.744187   
GOC Marketable Bonds Average Yield: 5-10 year                              -0.625150   
GOC Marketable Bonds Average Yield: 3-5 year                               -0.693431   
GOC Marketable Bonds Average Yield: over 10 years                          -0.561822   
GOC benchmark bond yields: 2 year                                          -0.735073   
GOC benchmark bond yields: 3 year                                          -0.720636   
GOC benchmark bond yields: 5 year                                          -0.668349   
GOC benchmark bond yields: 7 year                                          -0.626895   
GOC benchmark bond yields: 10 years                                        -0.591373   
GOC benchmark bond yields: long term                                       -0.562839   
Treasury bills: 1 month                                                    -0.765780   
Treasury bills: 2 month                                                    -0.768520   
Treasury bills: 3 month                                                    -0.770248   
Treasury bills: 6 month                                                    -0.764929   
Treasury bills: 1 year                                                     -0.761639   
CPI Inflaction Rate                                                        -0.541372   
Morgage Rate                                                               -0.805028   
Prime Rate                                                                 -0.762774   
House Price Index(house and land)                                           1.000000   
Unemployment rate                                                           0.388159   
Real GDP growth Seasonal adjustment                                         0.275270   

                                                   Unemployment rate  \
GOC Marketable Bonds Average Yield: 1-3 year               -0.746375   
GOC Marketable Bonds Average Yield: 5-10 year              -0.802838   
GOC Marketable Bonds Average Yield: 3-5 year               -0.785146   
GOC Marketable Bonds Average Yield: over 10 years          -0.805263   
GOC benchmark bond yields: 2 year                          -0.750504   
GOC benchmark bond yields: 3 year                          -0.765741   
GOC benchmark bond yields: 5 year                          -0.793680   
GOC benchmark bond yields: 7 year                          -0.806113   
GOC benchmark bond yields: 10 years                        -0.811841   
GOC benchmark bond yields: long term                       -0.794732   
Treasury bills: 1 month                                    -0.633458   
Treasury bills: 2 month                                    -0.644867   
Treasury bills: 3 month                                    -0.658274   
Treasury bills: 6 month                                    -0.682239   
Treasury bills: 1 year                                     -0.720879   
CPI Inflaction Rate                                        -0.751794   
Morgage Rate                                               -0.617461   
Prime Rate                                                 -0.694479   
House Price Index(house and land)                           0.388159   
Unemployment rate                                           1.000000   
Real GDP growth Seasonal adjustment                        -0.353492   

                                                   Real GDP growth Seasonal adjustment  
GOC Marketable Bonds Average Yield: 1-3 year                                 -0.031706  
GOC Marketable Bonds Average Yield: 5-10 year                                 0.009622  
GOC Marketable Bonds Average Yield: 3-5 year                                 -0.017584  
GOC Marketable Bonds Average Yield: over 10 years                             0.029892  
GOC benchmark bond yields: 2 year                                            -0.021169  
GOC benchmark bond yields: 3 year                                            -0.014608  
GOC benchmark bond yields: 5 year                                            -0.001892  
GOC benchmark bond yields: 7 year                                             0.007730  
GOC benchmark bond yields: 10 years                                           0.030765  
GOC benchmark bond yields: long term                                          0.025609  
Treasury bills: 1 month                                                      -0.047291  
Treasury bills: 2 month                                                      -0.045324  
Treasury bills: 3 month                                                      -0.048666  
Treasury bills: 6 month                                                      -0.045914  
Treasury bills: 1 year                                                       -0.036478  
CPI Inflaction Rate                                                           0.017866  
Morgage Rate                                                                 -0.090534  
Prime Rate                                                                   -0.048387  
House Price Index(house and land)                                             0.275270  
Unemployment rate                                                            -0.353492  
Real GDP growth Seasonal adjustment                                           1.000000  

[21 rows x 21 columns]
```
```
<Figure size 2000x1500 with 4 Axes>
```
```python
print_most_important_economic_factors()
```

```

                        *****************************************************************************************
                               Principal Components Analysis(PCA) to select  Most Important Economic Factors 
                          ****************************************************************************************

Principal Components Analysis(PCA) to select  Most Important Economic Factors 


```
```
       PC1       PC2      PC3       PC4       PC5       PC6       PC7  \
0  0.85935  0.068931  0.03383  0.024056  0.007083  0.004152  0.001816   

        PC8       PC9      PC10      PC11      PC12      PC13      PC14  \
0  0.000543  0.000109  0.000046  0.000031  0.000021  0.000012  0.000007   

       PC15      PC16      PC17          PC18  
0  0.000005  0.000004  0.000002  4.703571e-34  
```
```
<Figure size 1000x600 with 1 Axes>
```
```

loadings_matrix_df


```
```
                                                        PC1       PC2  \
GOC Marketable Bonds Average Yield: 1-3 year       0.234760  0.026131   
GOC Marketable Bonds Average Yield: 5-10 year      0.231555 -0.080234   
GOC Marketable Bonds Average Yield: 3-5 year       0.234132 -0.029851   
GOC Marketable Bonds Average Yield: over 10 years  0.227174 -0.122381   
GOC benchmark bond yields: 2 year                  0.234779  0.013805   
GOC benchmark bond yields: 3 year                  0.234758 -0.005337   
GOC benchmark bond yields: 5 year                  0.233485 -0.051348   
GOC benchmark bond yields: 7 year                  0.231707 -0.078183   
GOC benchmark bond yields: 10 years                0.230046 -0.107253   
GOC benchmark bond yields: long term               0.227639 -0.108980   
Treasury bills: 1 month                            0.225278  0.128380   
Treasury bills: 2 month                            0.226989  0.119869   
Treasury bills: 3 month                            0.228918  0.111167   
Treasury bills: 6 month                            0.231878  0.088394   
Treasury bills: 1 year                             0.234082  0.053455   
CPI Inflaction Rate                                0.173762 -0.180793   
Morgage Rate                                       0.227671  0.150522   
Prime Rate                                         0.232769  0.079864   
House Price Index(house and land)                 -0.171404 -0.381353   
Unemployment rate                                 -0.181182  0.436201   
Real GDP growth Seasonal adjustment               -0.003512 -0.702397   

                                                        PC3       PC4  \
GOC Marketable Bonds Average Yield: 1-3 year       0.016201 -0.012893   
GOC Marketable Bonds Average Yield: 5-10 year     -0.095232  0.153528   
GOC Marketable Bonds Average Yield: 3-5 year      -0.072510  0.042401   
GOC Marketable Bonds Average Yield: over 10 years -0.136092  0.217986   
GOC benchmark bond yields: 2 year                  0.008394 -0.010948   
GOC benchmark bond yields: 3 year                 -0.016723  0.008134   
GOC benchmark bond yields: 5 year                 -0.076537  0.077274   
GOC benchmark bond yields: 7 year                 -0.090904  0.152294   
GOC benchmark bond yields: 10 years               -0.098440  0.189327   
GOC benchmark bond yields: long term              -0.102382  0.240735   
Treasury bills: 1 month                            0.259529  0.019614   
Treasury bills: 2 month                            0.237662  0.011307   
Treasury bills: 3 month                            0.202732  0.004675   
Treasury bills: 6 month                            0.144440 -0.000347   
Treasury bills: 1 year                             0.065061 -0.036268   
CPI Inflaction Rate                               -0.575582 -0.504796   
Morgage Rate                                       0.160041 -0.041466   
Prime Rate                                         0.118729  0.003283   
House Price Index(house and land)                 -0.085620  0.663168   
Unemployment rate                                  0.172573  0.184132   
Real GDP growth Seasonal adjustment                0.578009 -0.267773   

                                                        PC5       PC6  \
GOC Marketable Bonds Average Yield: 1-3 year      -0.024366  0.061716   
GOC Marketable Bonds Average Yield: 5-10 year      0.031926  0.220653   
GOC Marketable Bonds Average Yield: 3-5 year      -0.000326  0.170944   
GOC Marketable Bonds Average Yield: over 10 years -0.057406  0.134132   
GOC benchmark bond yields: 2 year                 -0.061867  0.068719   
GOC benchmark bond yields: 3 year                 -0.029301  0.139984   
GOC benchmark bond yields: 5 year                  0.002426  0.185022   
GOC benchmark bond yields: 7 year                  0.065766  0.199976   
GOC benchmark bond yields: 10 years                0.017254  0.166038   
GOC benchmark bond yields: long term              -0.052869  0.050951   
Treasury bills: 1 month                            0.025262 -0.352326   
Treasury bills: 2 month                            0.007964 -0.309849   
Treasury bills: 3 month                           -0.015006 -0.274253   
Treasury bills: 6 month                           -0.064215 -0.175812   
Treasury bills: 1 year                            -0.075239 -0.075118   
CPI Inflaction Rate                               -0.493601 -0.289764   
Morgage Rate                                      -0.149447  0.043348   
Prime Rate                                        -0.045882 -0.147697   
House Price Index(house and land)                 -0.268190 -0.450428   
Unemployment rate                                 -0.757262  0.304913   
Real GDP growth Seasonal adjustment               -0.242523  0.190165   

                                                        PC7       PC8  \
GOC Marketable Bonds Average Yield: 1-3 year       0.291109 -0.139295   
GOC Marketable Bonds Average Yield: 5-10 year     -0.045993  0.043579   
GOC Marketable Bonds Average Yield: 3-5 year       0.217199 -0.078918   
GOC Marketable Bonds Average Yield: over 10 years -0.429816 -0.114250   
GOC benchmark bond yields: 2 year                  0.297248 -0.173324   
GOC benchmark bond yields: 3 year                  0.287062 -0.090465   
GOC benchmark bond yields: 5 year                  0.216081 -0.095769   
GOC benchmark bond yields: 7 year                  0.051898  0.010020   
GOC benchmark bond yields: 10 years               -0.119412  0.138684   
GOC benchmark bond yields: long term              -0.511097 -0.023798   
Treasury bills: 1 month                           -0.132600 -0.161266   
Treasury bills: 2 month                           -0.107582 -0.145893   
Treasury bills: 3 month                           -0.119133 -0.032772   
Treasury bills: 6 month                            0.048952 -0.000216   
Treasury bills: 1 year                             0.192900 -0.096072   
CPI Inflaction Rate                               -0.058610  0.010737   
Morgage Rate                                       0.053773  0.891069   
Prime Rate                                        -0.017152  0.019519   
House Price Index(house and land)                  0.298552  0.103229   
Unemployment rate                                 -0.061990 -0.167199   
Real GDP growth Seasonal adjustment               -0.055179 -0.037586   

                                                        PC9      PC10  \
GOC Marketable Bonds Average Yield: 1-3 year       0.133451  0.157802   
GOC Marketable Bonds Average Yield: 5-10 year     -0.190157  0.094682   
GOC Marketable Bonds Average Yield: 3-5 year       0.286689 -0.068549   
GOC Marketable Bonds Average Yield: over 10 years  0.444101 -0.171831   
GOC benchmark bond yields: 2 year                  0.017306 -0.047074   
GOC benchmark bond yields: 3 year                  0.039591  0.137423   
GOC benchmark bond yields: 5 year                 -0.199210 -0.309908   
GOC benchmark bond yields: 7 year                 -0.347636 -0.330405   
GOC benchmark bond yields: 10 years               -0.133025  0.001400   
GOC benchmark bond yields: long term              -0.019783  0.353407   
Treasury bills: 1 month                            0.086808 -0.541618   
Treasury bills: 2 month                           -0.201502 -0.078164   
Treasury bills: 3 month                            0.104932  0.042636   
Treasury bills: 6 month                           -0.009291  0.301415   
Treasury bills: 1 year                             0.369251  0.282200   
CPI Inflaction Rate                               -0.074097 -0.086223   
Morgage Rate                                       0.145476 -0.116283   
Prime Rate                                        -0.508541  0.292196   
House Price Index(house and land)                  0.016639 -0.009538   
Unemployment rate                                 -0.067842 -0.048952   
Real GDP growth Seasonal adjustment               -0.019916 -0.010577   

                                                       PC11      PC12  \
GOC Marketable Bonds Average Yield: 1-3 year       0.515279  0.182155   
GOC Marketable Bonds Average Yield: 5-10 year     -0.343397 -0.213668   
GOC Marketable Bonds Average Yield: 3-5 year       0.178631 -0.298389   
GOC Marketable Bonds Average Yield: over 10 years -0.208001  0.320040   
GOC benchmark bond yields: 2 year                 -0.160140 -0.375752   
GOC benchmark bond yields: 3 year                 -0.004180 -0.218928   
GOC benchmark bond yields: 5 year                  0.116107  0.498475   
GOC benchmark bond yields: 7 year                 -0.001623  0.058658   
GOC benchmark bond yields: 10 years               -0.256611 -0.100427   
GOC benchmark bond yields: long term               0.502917 -0.195265   
Treasury bills: 1 month                            0.152998 -0.267793   
Treasury bills: 2 month                           -0.020385  0.101861   
Treasury bills: 3 month                           -0.161700 -0.009663   
Treasury bills: 6 month                           -0.265119 -0.029288   
Treasury bills: 1 year                            -0.215109  0.348370   
CPI Inflaction Rate                               -0.010197 -0.031054   
Morgage Rate                                       0.110641  0.020574   
Prime Rate                                         0.064961  0.185499   
House Price Index(house and land)                  0.011201 -0.002338   
Unemployment rate                                 -0.015296 -0.015750   
Real GDP growth Seasonal adjustment                0.004074 -0.000240   

                                                       PC13      PC14  \
GOC Marketable Bonds Average Yield: 1-3 year       0.241015 -0.101838   
GOC Marketable Bonds Average Yield: 5-10 year     -0.071449  0.159755   
GOC Marketable Bonds Average Yield: 3-5 year      -0.702733  0.011258   
GOC Marketable Bonds Average Yield: over 10 years -0.023063  0.076492   
GOC benchmark bond yields: 2 year                  0.524676  0.397393   
GOC benchmark bond yields: 3 year                 -0.028056 -0.277546   
GOC benchmark bond yields: 5 year                 -0.034998  0.289604   
GOC benchmark bond yields: 7 year                  0.114303 -0.590334   
GOC benchmark bond yields: 10 years                0.049083  0.024542   
GOC benchmark bond yields: long term               0.169617  0.019484   
Treasury bills: 1 month                           -0.031216  0.162620   
Treasury bills: 2 month                           -0.005137 -0.203259   
Treasury bills: 3 month                            0.049030 -0.144779   
Treasury bills: 6 month                           -0.099210 -0.238959   
Treasury bills: 1 year                             0.066918  0.001790   
CPI Inflaction Rate                               -0.013493 -0.045574   
Morgage Rate                                       0.072023  0.044917   
Prime Rate                                        -0.310753  0.371407   
House Price Index(house and land)                 -0.013328  0.005792   
Unemployment rate                                 -0.028583 -0.042345   
Real GDP growth Seasonal adjustment               -0.015595 -0.011826   

                                                       PC15      PC16  \
GOC Marketable Bonds Average Yield: 1-3 year       0.360692 -0.184581   
GOC Marketable Bonds Average Yield: 5-10 year      0.474708 -0.266721   
GOC Marketable Bonds Average Yield: 3-5 year       0.143179  0.156651   
GOC Marketable Bonds Average Yield: over 10 years  0.068157 -0.254903   
GOC benchmark bond yields: 2 year                  0.059814  0.058629   
GOC benchmark bond yields: 3 year                 -0.505661 -0.334065   
GOC benchmark bond yields: 5 year                 -0.361245 -0.134422   
GOC benchmark bond yields: 7 year                  0.179727  0.203702   
GOC benchmark bond yields: 10 years               -0.254798  0.375737   
GOC benchmark bond yields: long term              -0.164477  0.134039   
Treasury bills: 1 month                           -0.103672  0.115781   
Treasury bills: 2 month                            0.208998  0.083060   
Treasury bills: 3 month                           -0.006277 -0.375099   
Treasury bills: 6 month                           -0.209504 -0.082658   
Treasury bills: 1 year                             0.046062  0.547883   
CPI Inflaction Rate                                0.023215 -0.007326   
Morgage Rate                                       0.027882 -0.039727   
Prime Rate                                         0.050503 -0.003452   
House Price Index(house and land)                  0.027148 -0.033986   
Unemployment rate                                  0.013702  0.025426   
Real GDP growth Seasonal adjustment                0.009457  0.008515   

                                                       PC17      PC18  
GOC Marketable Bonds Average Yield: 1-3 year      -0.003957  0.170916  
GOC Marketable Bonds Average Yield: 5-10 year      0.230909 -0.350206  
GOC Marketable Bonds Average Yield: 3-5 year       0.048250  0.146997  
GOC Marketable Bonds Average Yield: over 10 years -0.325177  0.221735  
GOC benchmark bond yields: 2 year                 -0.202109  0.198754  
GOC benchmark bond yields: 3 year                  0.068860 -0.036051  
GOC benchmark bond yields: 5 year                  0.216742 -0.104021  
GOC benchmark bond yields: 7 year                 -0.278951 -0.212126  
GOC benchmark bond yields: 10 years                0.279357  0.379159  
GOC benchmark bond yields: long term               0.065868 -0.173841  
Treasury bills: 1 month                           -0.119419 -0.296309  
Treasury bills: 2 month                            0.322462  0.497798  
Treasury bills: 3 month                            0.378795 -0.164712  
Treasury bills: 6 month                           -0.459009  0.058260  
Treasury bills: 1 year                             0.120218 -0.373485  
CPI Inflaction Rate                                0.039455 -0.009281  
Morgage Rate                                      -0.025172  0.026488  
Prime Rate                                        -0.308323  0.006737  
House Price Index(house and land)                  0.003009 -0.002548  
Unemployment rate                                  0.041119 -0.014845  
Real GDP growth Seasonal adjustment                0.003751 -0.009416  
```
```

top_components_df


```
```
                                                        PC1       PC2  \
GOC Marketable Bonds Average Yield: 1-3 year       0.234760  0.026131   
GOC Marketable Bonds Average Yield: 5-10 year      0.231555 -0.080234   
GOC Marketable Bonds Average Yield: 3-5 year       0.234132 -0.029851   
GOC Marketable Bonds Average Yield: over 10 years  0.227174 -0.122381   
GOC benchmark bond yields: 2 year                  0.234779  0.013805   
GOC benchmark bond yields: 3 year                  0.234758 -0.005337   
GOC benchmark bond yields: 5 year                  0.233485 -0.051348   
GOC benchmark bond yields: 7 year                  0.231707 -0.078183   
GOC benchmark bond yields: 10 years                0.230046 -0.107253   
GOC benchmark bond yields: long term               0.227639 -0.108980   
Treasury bills: 1 month                            0.225278  0.128380   
Treasury bills: 2 month                            0.226989  0.119869   
Treasury bills: 3 month                            0.228918  0.111167   
Treasury bills: 6 month                            0.231878  0.088394   
Treasury bills: 1 year                             0.234082  0.053455   
CPI Inflaction Rate                                0.173762 -0.180793   
Morgage Rate                                       0.227671  0.150522   
Prime Rate                                         0.232769  0.079864   
House Price Index(house and land)                 -0.171404 -0.381353   
Unemployment rate                                 -0.181182  0.436201   
Real GDP growth Seasonal adjustment               -0.003512 -0.702397   

                                                        PC3       PC4  \
GOC Marketable Bonds Average Yield: 1-3 year       0.016201 -0.012893   
GOC Marketable Bonds Average Yield: 5-10 year     -0.095232  0.153528   
GOC Marketable Bonds Average Yield: 3-5 year      -0.072510  0.042401   
GOC Marketable Bonds Average Yield: over 10 years -0.136092  0.217986   
GOC benchmark bond yields: 2 year                  0.008394 -0.010948   
GOC benchmark bond yields: 3 year                 -0.016723  0.008134   
GOC benchmark bond yields: 5 year                 -0.076537  0.077274   
GOC benchmark bond yields: 7 year                 -0.090904  0.152294   
GOC benchmark bond yields: 10 years               -0.098440  0.189327   
GOC benchmark bond yields: long term              -0.102382  0.240735   
Treasury bills: 1 month                            0.259529  0.019614   
Treasury bills: 2 month                            0.237662  0.011307   
Treasury bills: 3 month                            0.202732  0.004675   
Treasury bills: 6 month                            0.144440 -0.000347   
Treasury bills: 1 year                             0.065061 -0.036268   
CPI Inflaction Rate                               -0.575582 -0.504796   
Morgage Rate                                       0.160041 -0.041466   
Prime Rate                                         0.118729  0.003283   
House Price Index(house and land)                 -0.085620  0.663168   
Unemployment rate                                  0.172573  0.184132   
Real GDP growth Seasonal adjustment                0.578009 -0.267773   

                                                        PC5       PC6  \
GOC Marketable Bonds Average Yield: 1-3 year      -0.024366  0.061716   
GOC Marketable Bonds Average Yield: 5-10 year      0.031926  0.220653   
GOC Marketable Bonds Average Yield: 3-5 year      -0.000326  0.170944   
GOC Marketable Bonds Average Yield: over 10 years -0.057406  0.134132   
GOC benchmark bond yields: 2 year                 -0.061867  0.068719   
GOC benchmark bond yields: 3 year                 -0.029301  0.139984   
GOC benchmark bond yields: 5 year                  0.002426  0.185022   
GOC benchmark bond yields: 7 year                  0.065766  0.199976   
GOC benchmark bond yields: 10 years                0.017254  0.166038   
GOC benchmark bond yields: long term              -0.052869  0.050951   
Treasury bills: 1 month                            0.025262 -0.352326   
Treasury bills: 2 month                            0.007964 -0.309849   
Treasury bills: 3 month                           -0.015006 -0.274253   
Treasury bills: 6 month                           -0.064215 -0.175812   
Treasury bills: 1 year                            -0.075239 -0.075118   
CPI Inflaction Rate                               -0.493601 -0.289764   
Morgage Rate                                      -0.149447  0.043348   
Prime Rate                                        -0.045882 -0.147697   
House Price Index(house and land)                 -0.268190 -0.450428   
Unemployment rate                                 -0.757262  0.304913   
Real GDP growth Seasonal adjustment               -0.242523  0.190165   

                                                        PC7       PC8  \
GOC Marketable Bonds Average Yield: 1-3 year       0.291109 -0.139295   
GOC Marketable Bonds Average Yield: 5-10 year     -0.045993  0.043579   
GOC Marketable Bonds Average Yield: 3-5 year       0.217199 -0.078918   
GOC Marketable Bonds Average Yield: over 10 years -0.429816 -0.114250   
GOC benchmark bond yields: 2 year                  0.297248 -0.173324   
GOC benchmark bond yields: 3 year                  0.287062 -0.090465   
GOC benchmark bond yields: 5 year                  0.216081 -0.095769   
GOC benchmark bond yields: 7 year                  0.051898  0.010020   
GOC benchmark bond yields: 10 years               -0.119412  0.138684   
GOC benchmark bond yields: long term              -0.511097 -0.023798   
Treasury bills: 1 month                           -0.132600 -0.161266   
Treasury bills: 2 month                           -0.107582 -0.145893   
Treasury bills: 3 month                           -0.119133 -0.032772   
Treasury bills: 6 month                            0.048952 -0.000216   
Treasury bills: 1 year                             0.192900 -0.096072   
CPI Inflaction Rate                               -0.058610  0.010737   
Morgage Rate                                       0.053773  0.891069   
Prime Rate                                        -0.017152  0.019519   
House Price Index(house and land)                  0.298552  0.103229   
Unemployment rate                                 -0.061990 -0.167199   
Real GDP growth Seasonal adjustment               -0.055179 -0.037586   

                                                        PC9      PC10  \
GOC Marketable Bonds Average Yield: 1-3 year       0.133451  0.157802   
GOC Marketable Bonds Average Yield: 5-10 year     -0.190157  0.094682   
GOC Marketable Bonds Average Yield: 3-5 year       0.286689 -0.068549   
GOC Marketable Bonds Average Yield: over 10 years  0.444101 -0.171831   
GOC benchmark bond yields: 2 year                  0.017306 -0.047074   
GOC benchmark bond yields: 3 year                  0.039591  0.137423   
GOC benchmark bond yields: 5 year                 -0.199210 -0.309908   
GOC benchmark bond yields: 7 year                 -0.347636 -0.330405   
GOC benchmark bond yields: 10 years               -0.133025  0.001400   
GOC benchmark bond yields: long term              -0.019783  0.353407   
Treasury bills: 1 month                            0.086808 -0.541618   
Treasury bills: 2 month                           -0.201502 -0.078164   
Treasury bills: 3 month                            0.104932  0.042636   
Treasury bills: 6 month                           -0.009291  0.301415   
Treasury bills: 1 year                             0.369251  0.282200   
CPI Inflaction Rate                               -0.074097 -0.086223   
Morgage Rate                                       0.145476 -0.116283   
Prime Rate                                        -0.508541  0.292196   
House Price Index(house and land)                  0.016639 -0.009538   
Unemployment rate                                 -0.067842 -0.048952   
Real GDP growth Seasonal adjustment               -0.019916 -0.010577   

                                                       PC11      PC12  \
GOC Marketable Bonds Average Yield: 1-3 year       0.515279  0.182155   
GOC Marketable Bonds Average Yield: 5-10 year     -0.343397 -0.213668   
GOC Marketable Bonds Average Yield: 3-5 year       0.178631 -0.298389   
GOC Marketable Bonds Average Yield: over 10 years -0.208001  0.320040   
GOC benchmark bond yields: 2 year                 -0.160140 -0.375752   
GOC benchmark bond yields: 3 year                 -0.004180 -0.218928   
GOC benchmark bond yields: 5 year                  0.116107  0.498475   
GOC benchmark bond yields: 7 year                 -0.001623  0.058658   
GOC benchmark bond yields: 10 years               -0.256611 -0.100427   
GOC benchmark bond yields: long term               0.502917 -0.195265   
Treasury bills: 1 month                            0.152998 -0.267793   
Treasury bills: 2 month                           -0.020385  0.101861   
Treasury bills: 3 month                           -0.161700 -0.009663   
Treasury bills: 6 month                           -0.265119 -0.029288   
Treasury bills: 1 year                            -0.215109  0.348370   
CPI Inflaction Rate                               -0.010197 -0.031054   
Morgage Rate                                       0.110641  0.020574   
Prime Rate                                         0.064961  0.185499   
House Price Index(house and land)                  0.011201 -0.002338   
Unemployment rate                                 -0.015296 -0.015750   
Real GDP growth Seasonal adjustment                0.004074 -0.000240   

                                                       PC13      PC14  \
GOC Marketable Bonds Average Yield: 1-3 year       0.241015 -0.101838   
GOC Marketable Bonds Average Yield: 5-10 year     -0.071449  0.159755   
GOC Marketable Bonds Average Yield: 3-5 year      -0.702733  0.011258   
GOC Marketable Bonds Average Yield: over 10 years -0.023063  0.076492   
GOC benchmark bond yields: 2 year                  0.524676  0.397393   
GOC benchmark bond yields: 3 year                 -0.028056 -0.277546   
GOC benchmark bond yields: 5 year                 -0.034998  0.289604   
GOC benchmark bond yields: 7 year                  0.114303 -0.590334   
GOC benchmark bond yields: 10 years                0.049083  0.024542   
GOC benchmark bond yields: long term               0.169617  0.019484   
Treasury bills: 1 month                           -0.031216  0.162620   
Treasury bills: 2 month                           -0.005137 -0.203259   
Treasury bills: 3 month                            0.049030 -0.144779   
Treasury bills: 6 month                           -0.099210 -0.238959   
Treasury bills: 1 year                             0.066918  0.001790   
CPI Inflaction Rate                               -0.013493 -0.045574   
Morgage Rate                                       0.072023  0.044917   
Prime Rate                                        -0.310753  0.371407   
House Price Index(house and land)                 -0.013328  0.005792   
Unemployment rate                                 -0.028583 -0.042345   
Real GDP growth Seasonal adjustment               -0.015595 -0.011826   

                                                       PC15      PC16  \
GOC Marketable Bonds Average Yield: 1-3 year       0.360692 -0.184581   
GOC Marketable Bonds Average Yield: 5-10 year      0.474708 -0.266721   
GOC Marketable Bonds Average Yield: 3-5 year       0.143179  0.156651   
GOC Marketable Bonds Average Yield: over 10 years  0.068157 -0.254903   
GOC benchmark bond yields: 2 year                  0.059814  0.058629   
GOC benchmark bond yields: 3 year                 -0.505661 -0.334065   
GOC benchmark bond yields: 5 year                 -0.361245 -0.134422   
GOC benchmark bond yields: 7 year                  0.179727  0.203702   
GOC benchmark bond yields: 10 years               -0.254798  0.375737   
GOC benchmark bond yields: long term              -0.164477  0.134039   
Treasury bills: 1 month                           -0.103672  0.115781   
Treasury bills: 2 month                            0.208998  0.083060   
Treasury bills: 3 month                           -0.006277 -0.375099   
Treasury bills: 6 month                           -0.209504 -0.082658   
Treasury bills: 1 year                             0.046062  0.547883   
CPI Inflaction Rate                                0.023215 -0.007326   
Morgage Rate                                       0.027882 -0.039727   
Prime Rate                                         0.050503 -0.003452   
House Price Index(house and land)                  0.027148 -0.033986   
Unemployment rate                                  0.013702  0.025426   
Real GDP growth Seasonal adjustment                0.009457  0.008515   

                                                       PC17  
GOC Marketable Bonds Average Yield: 1-3 year      -0.003957  
GOC Marketable Bonds Average Yield: 5-10 year      0.230909  
GOC Marketable Bonds Average Yield: 3-5 year       0.048250  
GOC Marketable Bonds Average Yield: over 10 years -0.325177  
GOC benchmark bond yields: 2 year                 -0.202109  
GOC benchmark bond yields: 3 year                  0.068860  
GOC benchmark bond yields: 5 year                  0.216742  
GOC benchmark bond yields: 7 year                 -0.278951  
GOC benchmark bond yields: 10 years                0.279357  
GOC benchmark bond yields: long term               0.065868  
Treasury bills: 1 month                           -0.119419  
Treasury bills: 2 month                            0.322462  
Treasury bills: 3 month                            0.378795  
Treasury bills: 6 month                           -0.459009  
Treasury bills: 1 year                             0.120218  
CPI Inflaction Rate                                0.039455  
Morgage Rate                                      -0.025172  
Prime Rate                                        -0.308323  
House Price Index(house and land)                  0.003009  
Unemployment rate                                  0.041119  
Real GDP growth Seasonal adjustment                0.003751  
```
```

top_indicators_df


```
```
                                                   PC1       PC2       PC3  \
GOC Marketable Bonds Average Yield: 1-3 year  0.234760  0.026131  0.016201   
GOC Marketable Bonds Average Yield: 3-5 year  0.234132 -0.029851 -0.072510   
GOC benchmark bond yields: 2 year             0.234779  0.013805  0.008394   
GOC benchmark bond yields: 3 year             0.234758 -0.005337 -0.016723   
GOC benchmark bond yields: 7 year             0.231707 -0.078183 -0.090904   
GOC benchmark bond yields: long term          0.227639 -0.108980 -0.102382   
Treasury bills: 1 month                       0.225278  0.128380  0.259529   
Treasury bills: 1 year                        0.234082  0.053455  0.065061   
CPI Inflaction Rate                           0.173762 -0.180793 -0.575582   
Morgage Rate                                  0.227671  0.150522  0.160041   
Prime Rate                                    0.232769  0.079864  0.118729   
House Price Index(house and land)            -0.171404 -0.381353 -0.085620   
Unemployment rate                            -0.181182  0.436201  0.172573   
Real GDP growth Seasonal adjustment          -0.003512 -0.702397  0.578009   

                                                   PC4       PC5       PC6  \
GOC Marketable Bonds Average Yield: 1-3 year -0.012893 -0.024366  0.061716   
GOC Marketable Bonds Average Yield: 3-5 year  0.042401 -0.000326  0.170944   
GOC benchmark bond yields: 2 year            -0.010948 -0.061867  0.068719   
GOC benchmark bond yields: 3 year             0.008134 -0.029301  0.139984   
GOC benchmark bond yields: 7 year             0.152294  0.065766  0.199976   
GOC benchmark bond yields: long term          0.240735 -0.052869  0.050951   
Treasury bills: 1 month                       0.019614  0.025262 -0.352326   
Treasury bills: 1 year                       -0.036268 -0.075239 -0.075118   
CPI Inflaction Rate                          -0.504796 -0.493601 -0.289764   
Morgage Rate                                 -0.041466 -0.149447  0.043348   
Prime Rate                                    0.003283 -0.045882 -0.147697   
House Price Index(house and land)             0.663168 -0.268190 -0.450428   
Unemployment rate                             0.184132 -0.757262  0.304913   
Real GDP growth Seasonal adjustment          -0.267773 -0.242523  0.190165   

                                                   PC7       PC8       PC9  \
GOC Marketable Bonds Average Yield: 1-3 year  0.291109 -0.139295  0.133451   
GOC Marketable Bonds Average Yield: 3-5 year  0.217199 -0.078918  0.286689   
GOC benchmark bond yields: 2 year             0.297248 -0.173324  0.017306   
GOC benchmark bond yields: 3 year             0.287062 -0.090465  0.039591   
GOC benchmark bond yields: 7 year             0.051898  0.010020 -0.347636   
GOC benchmark bond yields: long term         -0.511097 -0.023798 -0.019783   
Treasury bills: 1 month                      -0.132600 -0.161266  0.086808   
Treasury bills: 1 year                        0.192900 -0.096072  0.369251   
CPI Inflaction Rate                          -0.058610  0.010737 -0.074097   
Morgage Rate                                  0.053773  0.891069  0.145476   
Prime Rate                                   -0.017152  0.019519 -0.508541   
House Price Index(house and land)             0.298552  0.103229  0.016639   
Unemployment rate                            -0.061990 -0.167199 -0.067842   
Real GDP growth Seasonal adjustment          -0.055179 -0.037586 -0.019916   

                                                  PC10      PC11      PC12  \
GOC Marketable Bonds Average Yield: 1-3 year  0.157802  0.515279  0.182155   
GOC Marketable Bonds Average Yield: 3-5 year -0.068549  0.178631 -0.298389   
GOC benchmark bond yields: 2 year            -0.047074 -0.160140 -0.375752   
GOC benchmark bond yields: 3 year             0.137423 -0.004180 -0.218928   
GOC benchmark bond yields: 7 year            -0.330405 -0.001623  0.058658   
GOC benchmark bond yields: long term          0.353407  0.502917 -0.195265   
Treasury bills: 1 month                      -0.541618  0.152998 -0.267793   
Treasury bills: 1 year                        0.282200 -0.215109  0.348370   
CPI Inflaction Rate                          -0.086223 -0.010197 -0.031054   
Morgage Rate                                 -0.116283  0.110641  0.020574   
Prime Rate                                    0.292196  0.064961  0.185499   
House Price Index(house and land)            -0.009538  0.011201 -0.002338   
Unemployment rate                            -0.048952 -0.015296 -0.015750   
Real GDP growth Seasonal adjustment          -0.010577  0.004074 -0.000240   

                                                  PC13      PC14      PC15  \
GOC Marketable Bonds Average Yield: 1-3 year  0.241015 -0.101838  0.360692   
GOC Marketable Bonds Average Yield: 3-5 year -0.702733  0.011258  0.143179   
GOC benchmark bond yields: 2 year             0.524676  0.397393  0.059814   
GOC benchmark bond yields: 3 year            -0.028056 -0.277546 -0.505661   
GOC benchmark bond yields: 7 year             0.114303 -0.590334  0.179727   
GOC benchmark bond yields: long term          0.169617  0.019484 -0.164477   
Treasury bills: 1 month                      -0.031216  0.162620 -0.103672   
Treasury bills: 1 year                        0.066918  0.001790  0.046062   
CPI Inflaction Rate                          -0.013493 -0.045574  0.023215   
Morgage Rate                                  0.072023  0.044917  0.027882   
Prime Rate                                   -0.310753  0.371407  0.050503   
House Price Index(house and land)            -0.013328  0.005792  0.027148   
Unemployment rate                            -0.028583 -0.042345  0.013702   
Real GDP growth Seasonal adjustment          -0.015595 -0.011826  0.009457   

                                                  PC16      PC17  
GOC Marketable Bonds Average Yield: 1-3 year -0.184581 -0.003957  
GOC Marketable Bonds Average Yield: 3-5 year  0.156651  0.048250  
GOC benchmark bond yields: 2 year             0.058629 -0.202109  
GOC benchmark bond yields: 3 year            -0.334065  0.068860  
GOC benchmark bond yields: 7 year             0.203702 -0.278951  
GOC benchmark bond yields: long term          0.134039  0.065868  
Treasury bills: 1 month                       0.115781 -0.119419  
Treasury bills: 1 year                        0.547883  0.120218  
CPI Inflaction Rate                          -0.007326  0.039455  
Morgage Rate                                 -0.039727 -0.025172  
Prime Rate                                   -0.003452 -0.308323  
House Price Index(house and land)            -0.033986  0.003009  
Unemployment rate                             0.025426  0.041119  
Real GDP growth Seasonal adjustment           0.008515  0.003751  
```
```

 most_important_economic_factors_df


```
```
              GOC Marketable Bonds Average Yield: 1-3 year  \
Quarter_Year                                                 
2020Q1                                                 1.2   
2020Q2                                                 0.3   
2020Q3                                                 0.2   
2020Q4                                                 0.2   
2021Q1                                                 0.2   
2021Q2                                                 0.3   
2021Q3                                                 0.4   
2021Q4                                                 1.0   
2022Q1                                                 1.6   
2022Q2                                                 2.7   
2022Q3                                                 3.5   
2022Q4                                                 3.9   
2023Q1                                                 3.9   
2023Q2                                                 4.1   
2023Q3                                                 4.8   
2023Q4                                                 4.3   
2024Q1                                                 4.2   
2024Q2                                                 4.3   

              GOC Marketable Bonds Average Yield: 3-5 year  \
Quarter_Year                                                 
2020Q1                                                 1.1   
2020Q2                                                 0.4   
2020Q3                                                 0.3   
2020Q4                                                 0.4   
2021Q1                                                 0.5   
2021Q2                                                 0.8   
2021Q3                                                 0.8   
2021Q4                                                 1.3   
2022Q1                                                 1.9   
2022Q2                                                 2.8   
2022Q3                                                 3.2   
2022Q4                                                 3.5   
2023Q1                                                 3.3   
2023Q2                                                 3.4   
2023Q3                                                 4.1   
2023Q4                                                 3.7   
2024Q1                                                 3.6   
2024Q2                                                 3.8   

              GOC benchmark bond yields: 2 year  \
Quarter_Year                                      
2020Q1                                      1.1   
2020Q2                                      0.3   
2020Q3                                      0.3   
2020Q4                                      0.2   
2021Q1                                      0.2   
2021Q2                                      0.4   
2021Q3                                      0.5   
2021Q4                                      1.0   
2022Q1                                      1.7   
2022Q2                                      2.7   
2022Q3                                      3.5   
2022Q4                                      3.9   
2023Q1                                      3.8   
2023Q2                                      4.1   
2023Q3                                      4.8   
2023Q4                                      4.3   
2024Q1                                      4.1   
2024Q2                                      4.2   

              GOC benchmark bond yields: 3 year  \
Quarter_Year                                      
2020Q1                                      1.1   
2020Q2                                      0.3   
2020Q3                                      0.3   
2020Q4                                      0.3   
2021Q1                                      0.3   
2021Q2                                      0.5   
2021Q3                                      0.6   
2021Q4                                      1.1   
2022Q1                                      1.8   
2022Q2                                      2.8   
2022Q3                                      3.4   
2022Q4                                      3.7   
2023Q1                                      3.6   
2023Q2                                      3.8   
2023Q3                                      4.5   
2023Q4                                      4.1   
2024Q1                                      3.9   
2024Q2                                      4.1   

              GOC benchmark bond yields: 7 year  \
Quarter_Year                                      
2020Q1                                      1.1   
2020Q2                                      0.4   
2020Q3                                      0.4   
2020Q4                                      0.5   
2021Q1                                      0.9   
2021Q2                                      1.2   
2021Q3                                      1.1   
2021Q4                                      1.5   
2022Q1                                      2.0   
2022Q2                                      2.8   
2022Q3                                      3.0   
2022Q4                                      3.1   
2023Q1                                      3.0   
2023Q2                                      3.1   
2023Q3                                      3.8   
2023Q4                                      3.6   
2024Q1                                      3.4   
2024Q2                                      3.7   

              GOC benchmark bond yields: long term  Treasury bills: 1 month  \
Quarter_Year                                                                  
2020Q1                                         1.4                      1.3   
2020Q2                                         1.1                      0.2   
2020Q3                                         1.1                      0.2   
2020Q4                                         1.2                      0.1   
2021Q1                                         1.8                      0.1   
2021Q2                                         2.0                      0.1   
2021Q3                                         1.8                      0.2   
2021Q4                                         1.9                      0.1   
2022Q1                                         2.2                      0.2   
2022Q2                                         2.9                      1.1   
2022Q3                                         2.9                      2.8   
2022Q4                                         3.2                      3.9   
2023Q1                                         3.1                      4.3   
2023Q2                                         3.1                      4.5   
2023Q3                                         3.5                      4.9   
2023Q4                                         3.4                      4.9   
2024Q1                                         3.3                      5.0   
2024Q2                                         3.6                      4.8   

              Treasury bills: 1 year  CPI Inflaction Rate  Morgage Rate  \
Quarter_Year                                                              
2020Q1                           1.2                  2.0           4.0   
2020Q2                           0.3                  1.6           3.9   
2020Q3                           0.2                  1.4           3.6   
2020Q4                           0.2                  1.7           3.4   
2021Q1                           0.1                  1.7           3.3   
2021Q2                           0.2                  2.4           3.3   
2021Q3                           0.3                  2.9           3.2   
2021Q4                           0.7                  3.1           3.4   
2022Q1                           1.4                  4.0           3.6   
2022Q2                           2.6                  5.3           4.6   
2022Q3                           3.7                  5.8           5.6   
2022Q4                           4.4                  6.0           5.8   
2023Q1                           4.4                  5.9           5.8   
2023Q2                           4.7                  5.3           5.8   
2023Q3                           5.2                  4.6           6.1   
2023Q4                           4.8                  4.0           6.4   
2024Q1                           4.8                  3.1           6.2   
2024Q2                           4.6                  2.5           6.1   

              Prime Rate  House Price Index(house and land)  \
Quarter_Year                                                  
2020Q1               1.8                                0.2   
2020Q2               1.2                                0.1   
2020Q3               1.1                                0.7   
2020Q4               1.0                                0.5   
2021Q1               1.1                                1.2   
2021Q2               1.2                                1.3   
2021Q3               1.2                                0.5   
2021Q4               1.3                                0.6   
2022Q1               1.6                                1.1   
2022Q2               2.5                                0.3   
2022Q3               3.2                                0.0   
2022Q4               3.7                               -0.1   
2023Q1               3.9                               -0.2   
2023Q2               4.0                                0.0   
2023Q3               4.3                               -0.1   
2023Q4               4.3                               -0.1   
2024Q1               4.1                                0.0   
2024Q2               4.1                                0.2   

              Unemployment rate  Real GDP growth Seasonal adjustment  
Quarter_Year                                                          
2020Q1                      4.6                                 -2.1  
2020Q2                      7.8                                -10.6  
2020Q3                      5.9                                  8.9  
2020Q4                      5.2                                  2.1  
2021Q1                      5.5                                  1.2  
2021Q2                      5.1                                 -0.1  
2021Q3                      4.6                                  1.6  
2021Q4                      4.1                                  1.6  
2022Q1                      4.3                                  0.8  
2022Q2                      3.7                                  1.1  
2022Q3                      3.6                                  0.5  
2022Q4                      3.5                                 -0.0  
2023Q1                      3.8                                  0.6  
2023Q2                      3.7                                  0.2  
2023Q3                      3.7                                 -0.1  
2023Q4                      3.6                                  0.1  
2024Q1                      4.0                                  0.5  
2024Q2                      4.0                                  0.4  
```
```

 most_important_economic_factors_matrix


```
```
                                              GOC Marketable Bonds Average Yield: 1-3 year  \
GOC Marketable Bonds Average Yield: 1-3 year                                      1.000000   
GOC Marketable Bonds Average Yield: 3-5 year                                      0.993209   
GOC benchmark bond yields: 2 year                                                 0.999462   
GOC benchmark bond yields: 3 year                                                 0.998315   
GOC benchmark bond yields: 7 year                                                 0.977915   
GOC benchmark bond yields: long term                                              0.952575   
Treasury bills: 1 month                                                           0.958881   
Treasury bills: 1 year                                                            0.996983   
CPI Inflaction Rate                                                               0.725500   
Morgage Rate                                                                      0.972380   
Prime Rate                                                                        0.989560   
House Price Index(house and land)                                                -0.744187   
Unemployment rate                                                                -0.746375   
Real GDP growth Seasonal adjustment                                              -0.031706   

                                              GOC Marketable Bonds Average Yield: 3-5 year  \
GOC Marketable Bonds Average Yield: 1-3 year                                      0.993209   
GOC Marketable Bonds Average Yield: 3-5 year                                      1.000000   
GOC benchmark bond yields: 2 year                                                 0.994331   
GOC benchmark bond yields: 3 year                                                 0.997760   
GOC benchmark bond yields: 7 year                                                 0.993509   
GOC benchmark bond yields: long term                                              0.973534   
Treasury bills: 1 month                                                           0.927301   
Treasury bills: 1 year                                                            0.983328   
CPI Inflaction Rate                                                               0.756007   
Morgage Rate                                                                      0.946726   
Prime Rate                                                                        0.971334   
House Price Index(house and land)                                                -0.693431   
Unemployment rate                                                                -0.785146   
Real GDP growth Seasonal adjustment                                              -0.017584   

                                              GOC benchmark bond yields: 2 year  \
GOC Marketable Bonds Average Yield: 1-3 year                           0.999462   
GOC Marketable Bonds Average Yield: 3-5 year                           0.994331   
GOC benchmark bond yields: 2 year                                      1.000000   
GOC benchmark bond yields: 3 year                                      0.998947   
GOC benchmark bond yields: 7 year                                      0.979915   
GOC benchmark bond yields: long term                                   0.955397   
Treasury bills: 1 month                                                0.955022   
Treasury bills: 1 year                                                 0.996038   
CPI Inflaction Rate                                                    0.734083   
Morgage Rate                                                           0.969309   
Prime Rate                                                             0.987723   
House Price Index(house and land)                                     -0.735073   
Unemployment rate                                                     -0.750504   
Real GDP growth Seasonal adjustment                                   -0.021169   

                                              GOC benchmark bond yields: 3 year  \
GOC Marketable Bonds Average Yield: 1-3 year                           0.998315   
GOC Marketable Bonds Average Yield: 3-5 year                           0.997760   
GOC benchmark bond yields: 2 year                                      0.998947   
GOC benchmark bond yields: 3 year                                      1.000000   
GOC benchmark bond yields: 7 year                                      0.986585   
GOC benchmark bond yields: long term                                   0.962796   
Treasury bills: 1 month                                                0.944663   
Treasury bills: 1 year                                                 0.992004   
CPI Inflaction Rate                                                    0.740256   
Morgage Rate                                                           0.962145   
Prime Rate                                                             0.982265   
House Price Index(house and land)                                     -0.720636   
Unemployment rate                                                     -0.765741   
Real GDP growth Seasonal adjustment                                   -0.014608   

                                              GOC benchmark bond yields: 7 year  \
GOC Marketable Bonds Average Yield: 1-3 year                           0.977915   
GOC Marketable Bonds Average Yield: 3-5 year                           0.993509   
GOC benchmark bond yields: 2 year                                      0.979915   
GOC benchmark bond yields: 3 year                                      0.986585   
GOC benchmark bond yields: 7 year                                      1.000000   
GOC benchmark bond yields: long term                                   0.988589   
Treasury bills: 1 month                                                0.906120   
Treasury bills: 1 year                                                 0.963724   
CPI Inflaction Rate                                                    0.735470   
Morgage Rate                                                           0.920864   
Prime Rate                                                             0.954083   
House Price Index(house and land)                                     -0.626895   
Unemployment rate                                                     -0.806113   
Real GDP growth Seasonal adjustment                                    0.007730   

                                              GOC benchmark bond yields: long term  \
GOC Marketable Bonds Average Yield: 1-3 year                              0.952575   
GOC Marketable Bonds Average Yield: 3-5 year                              0.973534   
GOC benchmark bond yields: 2 year                                         0.955397   
GOC benchmark bond yields: 3 year                                         0.962796   
GOC benchmark bond yields: 7 year                                         0.988589   
GOC benchmark bond yields: long term                                      1.000000   
Treasury bills: 1 month                                                   0.889460   
Treasury bills: 1 year                                                    0.940557   
CPI Inflaction Rate                                                       0.726525   
Morgage Rate                                                              0.894927   
Prime Rate                                                                0.935540   
House Price Index(house and land)                                        -0.562839   
Unemployment rate                                                        -0.794732   
Real GDP growth Seasonal adjustment                                       0.025609   

                                              Treasury bills: 1 month  \
GOC Marketable Bonds Average Yield: 1-3 year                 0.958881   
GOC Marketable Bonds Average Yield: 3-5 year                 0.927301   
GOC benchmark bond yields: 2 year                            0.955022   
GOC benchmark bond yields: 3 year                            0.944663   
GOC benchmark bond yields: 7 year                            0.906120   
GOC benchmark bond yields: long term                         0.889460   
Treasury bills: 1 month                                      1.000000   
Treasury bills: 1 year                                       0.974308   
CPI Inflaction Rate                                          0.569049   
Morgage Rate                                                 0.978951   
Prime Rate                                                   0.987231   
House Price Index(house and land)                           -0.765780   
Unemployment rate                                           -0.633458   
Real GDP growth Seasonal adjustment                         -0.047291   

                                              Treasury bills: 1 year  \
GOC Marketable Bonds Average Yield: 1-3 year                0.996983   
GOC Marketable Bonds Average Yield: 3-5 year                0.983328   
GOC benchmark bond yields: 2 year                           0.996038   
GOC benchmark bond yields: 3 year                           0.992004   
GOC benchmark bond yields: 7 year                           0.963724   
GOC benchmark bond yields: long term                        0.940557   
Treasury bills: 1 month                                     0.974308   
Treasury bills: 1 year                                      1.000000   
CPI Inflaction Rate                                         0.709570   
Morgage Rate                                                0.982448   
Prime Rate                                                  0.995893   
House Price Index(house and land)                          -0.761639   
Unemployment rate                                          -0.720879   
Real GDP growth Seasonal adjustment                        -0.036478   

                                              CPI Inflaction Rate  \
GOC Marketable Bonds Average Yield: 1-3 year             0.725500   
GOC Marketable Bonds Average Yield: 3-5 year             0.756007   
GOC benchmark bond yields: 2 year                        0.734083   
GOC benchmark bond yields: 3 year                        0.740256   
GOC benchmark bond yields: 7 year                        0.735470   
GOC benchmark bond yields: long term                     0.726525   
Treasury bills: 1 month                                  0.569049   
Treasury bills: 1 year                                   0.709570   
CPI Inflaction Rate                                      1.000000   
Morgage Rate                                             0.629513   
Prime Rate                                               0.666817   
House Price Index(house and land)                       -0.541372   
Unemployment rate                                       -0.751794   
Real GDP growth Seasonal adjustment                      0.017866   

                                              Morgage Rate  Prime Rate  \
GOC Marketable Bonds Average Yield: 1-3 year      0.972380    0.989560   
GOC Marketable Bonds Average Yield: 3-5 year      0.946726    0.971334   
GOC benchmark bond yields: 2 year                 0.969309    0.987723   
GOC benchmark bond yields: 3 year                 0.962145    0.982265   
GOC benchmark bond yields: 7 year                 0.920864    0.954083   
GOC benchmark bond yields: long term              0.894927    0.935540   
Treasury bills: 1 month                           0.978951    0.987231   
Treasury bills: 1 year                            0.982448    0.995893   
CPI Inflaction Rate                               0.629513    0.666817   
Morgage Rate                                      1.000000    0.987620   
Prime Rate                                        0.987620    1.000000   
House Price Index(house and land)                -0.805028   -0.762774   
Unemployment rate                                -0.617461   -0.694479   
Real GDP growth Seasonal adjustment              -0.090534   -0.048387   

                                              House Price Index(house and land)  \
GOC Marketable Bonds Average Yield: 1-3 year                          -0.744187   
GOC Marketable Bonds Average Yield: 3-5 year                          -0.693431   
GOC benchmark bond yields: 2 year                                     -0.735073   
GOC benchmark bond yields: 3 year                                     -0.720636   
GOC benchmark bond yields: 7 year                                     -0.626895   
GOC benchmark bond yields: long term                                  -0.562839   
Treasury bills: 1 month                                               -0.765780   
Treasury bills: 1 year                                                -0.761639   
CPI Inflaction Rate                                                   -0.541372   
Morgage Rate                                                          -0.805028   
Prime Rate                                                            -0.762774   
House Price Index(house and land)                                      1.000000   
Unemployment rate                                                      0.388159   
Real GDP growth Seasonal adjustment                                    0.275270   

                                              Unemployment rate  \
GOC Marketable Bonds Average Yield: 1-3 year          -0.746375   
GOC Marketable Bonds Average Yield: 3-5 year          -0.785146   
GOC benchmark bond yields: 2 year                     -0.750504   
GOC benchmark bond yields: 3 year                     -0.765741   
GOC benchmark bond yields: 7 year                     -0.806113   
GOC benchmark bond yields: long term                  -0.794732   
Treasury bills: 1 month                               -0.633458   
Treasury bills: 1 year                                -0.720879   
CPI Inflaction Rate                                   -0.751794   
Morgage Rate                                          -0.617461   
Prime Rate                                            -0.694479   
House Price Index(house and land)                      0.388159   
Unemployment rate                                      1.000000   
Real GDP growth Seasonal adjustment                   -0.353492   

                                              Real GDP growth Seasonal adjustment  
GOC Marketable Bonds Average Yield: 1-3 year                            -0.031706  
GOC Marketable Bonds Average Yield: 3-5 year                            -0.017584  
GOC benchmark bond yields: 2 year                                       -0.021169  
GOC benchmark bond yields: 3 year                                       -0.014608  
GOC benchmark bond yields: 7 year                                        0.007730  
GOC benchmark bond yields: long term                                     0.025609  
Treasury bills: 1 month                                                 -0.047291  
Treasury bills: 1 year                                                  -0.036478  
CPI Inflaction Rate                                                      0.017866  
Morgage Rate                                                            -0.090534  
Prime Rate                                                              -0.048387  
House Price Index(house and land)                                        0.275270  
Unemployment rate                                                       -0.353492  
Real GDP growth Seasonal adjustment                                      1.000000  
```
```
<Figure size 2000x1500 with 4 Axes>
```
##### Scenario Analysis

Macroeconomics KPI Best Case Scenario - Worst Case Scenario and Normal Case Scenario


```python

```

##### Recalculate Portfolio Key Performence metricstrics

 recalculate Expected return, Standard deviation (risk), and Value-at-Risk (VaR).


```python

```

##### Visualize the Stress Test Results

Visualizing the impact of the stress scenario on the portfolio can help in understanding the potential risks.


#### Decision Trees in Portfolio Stress Testing

In this section, we will use Decision Tree to model how different scenarios might cascade through the portfolio, affecting asset values, returns, and overall portfolio performance.



```python

```

#####  Interpret the Results



 - Expected Return under Stress: Indicates how much the portfolio's return is expected to decrease under the stress scenario.

 - Portfolio Risk under Stress: Shows how much the risk (volatility) increases under the stress scenario.

 - VaR under Stress: Quantifies the potential loss in the portfolio's value at a specified confidence level under stressed conditions.


```python

```

```python

```

```python
#historical_data_int = pd.read_csv('~/Documents/FinancialMath/SummerSeminar2016/2016-supervisory-historical-data/SupervisoryhistoricalInternational.csv', date_parser=pd.Period)
```

```python
#historical_data_dom = pd.read_csv('~/Documents/FinancialMath/SummerSeminar2016/2016-supervisory-historical-data/SupervisoryhistoricalDomestic.csv', date_parser=pd.Period)
```

### Refences

##### Read and print the stock tickers that make up S&P/TSX_Composite_Index

https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index

tickersDJIA = pd.read_html(

    'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')[0]

    https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average

    https://en.wikipedia.org/wiki/New_York_Stock_Exchange

clusters

https://medium.com/pursuitnotes/k-means-clustering-model-in-6-steps-with-python-35b532cfa8ad

https://odsc.medium.com/unsupervised-learning-evaluating-clusters-bd47eed175ce

https://medium.com/@nusfintech.ml/ml-optimisation-for-portfolio-allocation-9da34e7fe6b1



https://www.scikit-yb.org/en/latest/

https://plotly.com/python/time-series/



print(tickersDJIA.head())



# Get the data for the tickers from yahoo finance

data = yf.download(tickersDJIA.Symbol.to_list(),'2021-1-1','2021-7-12', auto_adjust=True)['Close']

print(data.head())

1. Mean Absolute Error (MAE)

2. Root Mean Squared Error (RMSE)

3. Mean Absolute Percentage Error (MAPE)

4. R-Squared Score.

4.   https://www.math.utah.edu/~palais/pcr/spike/Evaluating%20the%20Goodness%20of%20Fit%20--%20Fitting%20Data%20(Curve%20Fitting%20Toolbox)%20copy.html

7.  To compute one standard deviation errors on the parameters, use perr = np.sqrt(np.diag(pcov))

8. https://statisticsbyjim.com/regression/curve-fitting-linear-nonlinear-regression/

9. Interpreting the results (coefficient, intercept) and calculating the accuracy of the model

10. Visualization (plotting a graph)

11. https://data36.com/linear-regression-in-python-numpy-polyfit/

12. Importing the Python libraries we will use

13. Getting the data

14. Defining x values (the input variable) and y values (the output variable)

15, Machine Learning: fitting the model

16. Interpreting the results (coefficient, intercept) and calculating the accuracy of the model

17. machine learning is really what comes before it (data preparation, data cleaning) and what comes after it (interpreting, testing, validating and fine-tuning the model).

18. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html

19. mean squared error (MSE), R-squared, or adjusted R-squared can be used to assess

20. https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html

https://builtin.com/data-science/step-step-explanation-principal-component-analysis


#solving lineair equation

**solving lineair equation

#confusion matrinx

#weight optimization

https://www.geeksforgeeks.org/data-science-solving-linear-equations-2/?ref=ml_lbp



https://lmfit.github.io/lmfit-py/examples/example_fit_with_bounds.html

http://wwwens.aero.jussieu.fr/lefrere/master/SPE/docs-python/scipy-doc/generated/scipy.optimize.curve_fit.html

https://lmfit.github.io/lmfit-py/examples/example_fit_with_bounds.html

https://datascience.stackexchange.com/questions/65136/get-the-polynomial-equation-with-two-variables-in-python



def func(x, a, b, c):

    return a * np.exp(-b * x) + c



xdata = np.linspace(0, 4, 50)

y = func(xdata, 2.5, 1.3, 0.5)

rng = np.random.default_rng()

y_noise = 0.2 * rng.normal(size=xdata.size)

ydata = y + y_noise

plt.plot(xdata, ydata, 'b-', label='data')



popt, pcov = curve_fit(func, xdata, ydata)

plt.plot(xdata, func(xdata, *popt), 'r-',

         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))



#Constrain the optimization to the region of 0 <= a <= 3, 0 <= b <= 1 and 0 <= c <= 0.5:

popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))

plt.plot(xdata, func(xdata, *popt), 'g--',

         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))



plt.xlabel('x')

plt.ylabel('y')

plt.legend()

plt.show()





#------------

## Kolmogorov-Smirnov Test

The Kolmogorov-Smirnov (K-S) test is a nonparametric test that can be used to evaluate whether a sample comes from a population with a specific continuous distribution.



To perform the K-S test in Python, we can use the scipy.stats.kstest function from the scipy module.

from scipy.stats import kstest



# Sample data

sample = [0.5, 0.4, 0.35, 0.3, 0.25]



# Perform the K-S test

statistic, p_value = kstest(sample, 'norm')



print(statistic)

print(p_value)


import matplotlib.pyplot as plt



# Create a figure and axes

fig, ax = plt.subplots()



# Create a scatter plot of data

scatter = ax.scatter(x, y)



# Add a hover tooltip to the scatter plot

annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",

                  bbox=dict(boxstyle="round", fc="w"),

                  arrowprops=dict(arrowstyle="->"))

annot.set_visible(False)



def update_annot(ind):

    pos = scatter.get_offsets()[ind["ind"][0]]

    annot.xy = pos

    text = f"{ind['ind']}: {pos}"

    annot.set_text(text)



def hover(event):

    vis = annot.get_visible()

    if event.inaxes == ax:

        cont, ind = scatter.contains(event)

        if cont:

            update_annot(ind)

            annot.set_visible(True)

            fig.canvas.draw_idle()

        else:

            if vis:

                annot.set_visible(False)

                fig.canvas.draw_idle()



fig.canvas.mpl_connect("motion_notify_event", hover)



plt.show()



\https://www.quora.com/How-can-you-display-a-tooltip-with-additional-information-about-the-bounding-box-while-it-is-being-hovered-over-in-Python


```python



```

```python

```
