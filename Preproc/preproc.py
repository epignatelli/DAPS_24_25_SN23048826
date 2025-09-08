"""
This python file aims to pre process the datasets so that they can be explored further
"""
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter
from matplotlib import dates as mdates
import seaborn as sns
from Utility.utils import create_directory

def preprocess_data(data_type, data):
    """
    Pre-process dataset using the given input (dataset name)
    """

    # Create process directory
    create_directory('processing')

    if data_type == "stock":
        return preprocess_stock_data(data)
    if data_type == "covid":
        return preprocess_covid_data(data)
    if data_type == "interest":
        return preprocess_interest_data(data)


def preprocess_interest_data(interest_data):

    """ 
    This function starts by performing data cleaning via linear interpolation to deal with 
    missing interest rate weekend data. Then it will fill the missing data 
    for the first 2 days of April.
    Once completed, it will provide various visualisation graphs from plots,boxplots and zscores. 
    Finally, it will perform normalization.        
    """

    # Convert the data extracted into a datafram with sort in ascending order according to date.
    interest_dataframe = pd.DataFrame.from_records(interest_data)
    interest_dataframe =interest_dataframe.sort_values(by=['effectiveDate'])

    # Remove interest rate unwanted columns and rows (The latter is for data that are not related to
    # effective federal funds rate (EFFR) interest rate type)
    interest_dataframe = interest_dataframe[interest_dataframe['type']=='EFFR']
    interest_dataframe = interest_dataframe.drop(columns=['_id','type',
                                                                    'percentPercentile1',
                                                                    'percentPercentile25',
                                                                    'percentPercentile75',
                                                                    'percentPercentile99',
                                                                    'targetRateFrom',
                                                                    'targetRateTo',
                                                                    'revisionIndicator',
                                                                    'average30day',
                                                                    'average90day',
                                                                    'average180day',
                                                                    'index','footnoteId'])

    # Convert the effectiveDate column from string into a datetime format and set index.
    interest_dataframe['effectiveDate'] = pd.to_datetime(interest_dataframe['effectiveDate'])
    interest_dataframe = interest_dataframe.set_index('effectiveDate')

    # plot the interest rate data
    for column in interest_dataframe:
        show_interest_data(interest_dataframe,column)


    # Perform linear interpolation to data clean and fill missing weekend data
    interest_dataframe = interest_dataframe.resample('1D')
    interest_dataframe = interest_dataframe.interpolate(method='time')

    # Fill up the missing first two days of April 2017
    first_line_df = interest_dataframe.iloc[:1]
    first_line_index = first_line_df.index[0]

    missing_day_one = first_line_df
    new_index = first_line_index - datetime.timedelta(days=2)
    missing_day_one = missing_day_one.rename(index={first_line_index:new_index},
                                            inplace=False)

    missing_day_two = first_line_df
    new_index = first_line_index - datetime.timedelta(days=1)
    missing_day_two = missing_day_two.rename(index={first_line_index:new_index},
                                             inplace=False)

    interest_dataframe = pd.concat([missing_day_one,missing_day_two,
                                         interest_dataframe.loc[:]])


    # Plot boxplot and zscore outliers
    for column in interest_dataframe:
        show_boxplot(interest_dataframe,column,'interest')

    for column in interest_dataframe:
        zscore_plots(interest_dataframe,column,'interest')

    # Transform data via normalization
    interest_dataframe = normalize(interest_dataframe)
    return interest_dataframe


def preprocess_stock_data(stock_data):

    """ 
    This function starts by performing data cleaning via interpolation to deal
    with missing stock weekend data.
    Then it will fill the missing data for the first 2 days of April
    Once completed, it will provide various visualisation graphs from plots,
    boxplots and zscores.
    Finally, it will perform normalization.
    """

    # Convert the data extracted into a datafram with sort in ascending order according to date.
    stock_dataframe = pd.DataFrame.from_records(stock_data)
    stock_dataframe = stock_dataframe.sort_values(by=['Date'])

    # Remove interest rate unwanted columns
    stock_dataframe = stock_dataframe.drop(columns=['_id','Open','High','Low',
                                                    'Dividends','Stock Splits','Volume'])

    # Convert the Date column from string into a datetime format and set index.
    stock_dataframe['Date'] = pd.to_datetime(stock_dataframe['Date'])
    stock_dataframe = stock_dataframe.set_index('Date')

    # plot the stock data
    for column in stock_dataframe[['Close']]:
        show_stock_data(stock_dataframe,column)

    # plot the close price stocks for each season
    show_stocks_close_season(stock_dataframe)

    # Perform linear interpolation to data clean and fill missing weekend data
    stock_dataframe = stock_dataframe.resample('1D')
    stock_dataframe = stock_dataframe.interpolate(method='time')

    # Fill up the missing first two days of April 2017
    first_line_df = stock_dataframe.iloc[:1]
    first_line_index = first_line_df.index[0]

    last_line_df = stock_dataframe.iloc[-1:]
    last_line_index = last_line_df.index[-1]

    missing_day_one = first_line_df
    new_index = first_line_index - datetime.timedelta(days=2)
    missing_day_one = missing_day_one.rename(index={first_line_index:new_index},
                                             inplace=False)

    missing_day_two = first_line_df
    new_index = first_line_index - datetime.timedelta(days=1)
    missing_day_two = missing_day_two.rename(index={first_line_index:new_index},
                                             inplace=False)

    # Fill up the missing last 3 days
    missing_day_last_one = last_line_df
    new_index = last_line_index + datetime.timedelta(days=1)
    missing_day_last_one = missing_day_last_one.rename(index={last_line_index:new_index},
                                                       inplace=False)

    missing_day_last_two = last_line_df
    new_index = last_line_index + datetime.timedelta(days=2)
    missing_day_last_two = missing_day_last_two.rename(index={last_line_index:new_index},
                                                       inplace=False)

    missing_day_last_three = last_line_df
    new_index = last_line_index + datetime.timedelta(days=3)
    missing_day_last_three = missing_day_last_three.rename(index={last_line_index:new_index},
                                                           inplace=False)

    # Update stock dataframe
    stock_dataframe = pd.concat([missing_day_one,missing_day_two,
                                 stock_dataframe.loc[:],
                                 missing_day_last_one,
                                 missing_day_last_two,
                                 missing_day_last_three])


    # Plot boxplot and zscore outliers
    for column in stock_dataframe:
        show_boxplot(stock_dataframe,column,'stocks')

    for column in stock_dataframe:
        zscore_plots(stock_dataframe,column,'stocks')

    # Transform data via normalization
    stock_dataframe = normalize(stock_dataframe)

    return stock_dataframe


def preprocess_covid_data(covid_data):

    """ 
    This function starts by performing data cleaning via imputing to deal
    with missing covid-19 data (from 04-2017 to 02-2020).
    Once completed, it will provide various visualisation graphs from plots,boxplots and zscores.
    Finally, it will perform normalization.
    """

    # Convert the data extracted into a datafram with sort in ascending order according to date.
    covid_dataframe = pd.DataFrame.from_records(covid_data)
    covid_dataframe = covid_dataframe.sort_values(by=['date_of_interest'])

    # Drop any data that are after 1st of April
    covid_dataframe = covid_dataframe[covid_dataframe['date_of_interest']<'2024-04-01']

    # Remove interest rate unwanted columns
    covid_dataframe = covid_dataframe.drop(columns=['_id','probable_case_count',
            'hospitalized_count', 'case_count_7day_avg',
            'all_case_count_7day_avg', 'hosp_count_7day_avg',
            'death_count_7day_avg', 'bx_case_count', 'bx_probable_case_count',
            'bx_hospitalized_count', 'bx_death_count', 'bx_case_count_7day_avg',
            'bx_probable_case_count_7day_avg', 'bx_all_case_count_7day_avg',
            'bx_hospitalized_count_7day_avg', 'bx_death_count_7day_avg',
            'bk_case_count', 'bk_probable_case_count', 'bk_hospitalized_count',
            'bk_death_count', 'bk_case_count_7day_avg',
            'bk_probable_case_count_7day_avg', 'bk_all_case_count_7day_avg',
            'bk_hospitalized_count_7day_avg', 'bk_death_count_7day_avg',
            'mn_case_count', 'mn_probable_case_count', 'mn_hospitalized_count',
            'mn_death_count', 'mn_case_count_7day_avg',
            'mn_probable_case_count_7day_avg', 'mn_all_case_count_7day_avg',
            'mn_hospitalized_count_7day_avg', 'mn_death_count_7day_avg',
            'qn_case_count', 'qn_probable_case_count', 'qn_hospitalized_count',
            'qn_death_count', 'qn_case_count_7day_avg',
            'qn_probable_case_count_7day_avg', 'qn_all_case_count_7day_avg',
            'qn_hospitalized_count_7day_avg', 'qn_death_count_7day_avg',
            'si_case_count', 'si_probable_case_count', 'si_hospitalized_count',
            'si_death_count', 'si_probable_case_count_7day_avg',
            'si_case_count_7day_avg', 'si_all_case_count_7day_avg',
            'si_hospitalized_count_7day_avg', 'si_death_count_7day_avg',
            'incomplete'])

    # Convert the date_of_interest column from string into a datetime format and set index.
    covid_dataframe = covid_dataframe.set_index('date_of_interest')

    # Fill up the missing data from April 2017 to Feburary 2020
    first_covid_data= covid_dataframe.iloc[:1].copy()
    for column in first_covid_data.columns:
        first_covid_data.loc[:,column] = 0
    initial_date = '2017-04-01'
    initial_date_string = datetime.datetime.strptime(initial_date,'%Y-%m-%d').date()
    covid_start_date = covid_dataframe.iloc[:1].index[0]
    first_covid_data = first_covid_data.rename(index={covid_start_date:initial_date},
                                               inplace=False)
    covid_start_date = datetime.datetime.strptime(covid_start_date,'%Y-%m-%d').date()

    while initial_date_string < covid_start_date:
        covid_dataframe = pd.concat([first_covid_data,covid_dataframe.loc[:]])
        first_covid_data = first_covid_data.rename(
            index={initial_date_string.strftime('%Y-%m-%d'):(initial_date_string
                                                             + datetime.timedelta(days=1)).strftime(
                                                                '%Y-%m-%d')},
                                                             inplace=False)
        initial_date_string = initial_date_string + datetime.timedelta(days=1)

    covid_dataframe = covid_dataframe.sort_index()
    covid_dataframe = covid_dataframe.set_index(covid_dataframe.index.astype(
                                        dtype='datetime64[ns]'))

    # plot covid data
    for column in covid_dataframe:
        show_covid_data(covid_dataframe,column,'_ori')

    # Gaussian Filter to smooth the data and remove the noise.
    covid_case_list = [int(x) for x in list(covid_dataframe['case_count'])]
    covid_death_list = [int(x) for x in list(covid_dataframe['death_count'])]

    # Smootthing the data via Gaussian Filter
    covid_dataframe['case_count'] = gaussian_filter(covid_case_list,1,truncate=3)
    covid_dataframe['death_count'] = gaussian_filter(covid_death_list,1,truncate=3)

    # plot smoothed covid data
    for column in covid_dataframe:
        show_covid_data(covid_dataframe,column,'_smooth')

    # plot covid data boxplot and zscore
    for column in covid_dataframe:
        show_boxplot(covid_dataframe,column,'covid')


    for column in covid_dataframe:
        zscore_plots(covid_dataframe,column,'covid')

    # perform data transformation using normalization
    covid_dataframe = normalize(covid_dataframe)
    return covid_dataframe



def show_stocks_close_season(stock_data):

    """
    Plot line charts for stock close prices data and save in local disk.
    """

    try:

        fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7,1,sharex=False,figsize=(15,15))

    # Plotting data on each subplot
        ax1.plot(stock_data['Close'].loc['2017-04-01 00:00:00':'2018-04-01 00:00:00'])

        ax2.plot(stock_data['Close'].loc['2018-04-01 00:00:00':'2019-04-01 00:00:00'])

        ax3.plot(stock_data['Close'].loc['2019-04-01 00:00:00':'2020-04-01 00:00:00'])

        ax4.plot(stock_data['Close'].loc['2020-04-01 00:00:00':'2021-04-01 00:00:00'])

        ax5.plot(stock_data['Close'].loc['2021-04-01 00:00:00':'2022-04-01 00:00:00'])

        ax6.plot(stock_data['Close'].loc['2022-04-01 00:00:00':'2023-04-01 00:00:00'])

        ax7.plot(stock_data['Close'].loc['2023-04-01 00:00:00':'2024-04-01 00:00:00'])


        # Method 2: Manually setting visibility of x-axis labels and ticks
        for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax7]:
            ax.set_title('S&P 500 Close Price Vs Datetime')
            ax.set_xlabel('DateTime')
            ax.set_ylabel('Stock Price')
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            ax.xaxis.set_tick_params(labelbottom=True)
            ax.grid()

        # Adjust layout
        plt.tight_layout()
        plt.savefig('processing/stock_close_season.png')
        plt.close()

    except ValueError as e:
        print(f"Failed to plot the seasonal stock close prices data. Error: {e}")

def show_interest_data(df_interest,column):
    """
    Plot line chart for interest rate data and save in local disk.
    """
    try :
        # Sets the label
        label = column + ' value'

        # Set the time limits (i.e 01-04-2017 to 01-04-2024)
        # start_time = datetime.date(2017, 4, 1)
        # end_time = datetime.date(2024, 4, 1)

        # Performs the plot
        plt.figure().set_figwidth(15)
        plt.plot(df_interest[column],label=label,linestyle='-',c='b')
        plt.xlabel('DateTime')
        plt.ylabel('Value')
        plt.grid()
        plt.legend()
        plt.title('US interest rate Vs Date')
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        plt.savefig('processing/interest_' + column.lower() + '.png')
        plt.close()

    except ValueError as e:
        print(f"Failed to plot the interest rate data. Error: {e}")



def show_stock_data(df_stock,column):
    """
    Plot line chart for stocks data and save in local disk.
    """
    try :
        # Sets the label
        label = column + ' value'

        # Set the time limits (i.e 01-04-2017 to 01-04-2024)


        # Performs the plot
        plt.figure().set_figwidth(15)
        plt.plot(df_stock[column],label=label,linestyle='-',c='b')
        plt.xlabel('DateTime')
        plt.ylabel('Value')
        plt.grid()
        plt.legend()
        plt.title('S&P 500 stock close price Vs Date')
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.gcf().autofmt_xdate()
        plt.savefig('processing/stock_'+column.lower()+'.png')
        plt.close()

    except ValueError as e:
        print(f"Failed to plot the stock data. Error: {e}")



def show_covid_data(df_covid,column,description):
    """
    Plot line chart for covid data and save in local disk.
    """
    try :
        # Sets the label
        label = column + ' value'

        # Set the time limits (i.e 01-04-2017 to 01-04-2024)
        start_time = datetime.date(2017,4,1)
        end_time = datetime.date(2024,4,1)

        # Performs the plot
        plt.figure()
        plt.plot(df_covid[column],label=label,linestyle='-',c='b')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid()
        plt.title(f"covid {column.lower()} Vs Date")
        plt.legend()
        plt.xlim(start_time,end_time)
        plt.savefig('processing/covid_' + column.lower() + description + '.jpeg')
        plt.close()

    except ValueError as e:
        print(f"Failed to plot the stock data. Error: {e}")


def normalize(data):
    """
    Normalises the data between 0 and 1 using the min-max normalization.
    """
    try:
        # make sure the input data is a numpy array
        result = data.copy()
        for feature_name in data.columns:
            maximum_value = data[feature_name].max()
            minimum_value = data[feature_name].min()
            result[feature_name] = (data[feature_name] - minimum_value) / (maximum_value 
                                                        - minimum_value)

        # return the results
        return result

    except ValueError as e:
        print(f"Failed to peform data transformation via normalization. Error: {e}")
        return None

def zscore_plots(data,column,category):
    """
    Detects outliers using the z-score method and returns the indices of the outliers.
    """

    try:
        # Set the time limits (i.e 01-04-2017 to 01-04-2024)
        start_time = datetime.date(2017,4,1)
        end_time = datetime.date(2024,4,1)

        label = column + ' zscore'

        # calculate the z-score
        z_score = np.abs(stats.zscore(data[column]))
        # Perform the plots
        plt.figure()
        plt.plot(z_score,label=label,linestyle='-',c='b')
        plt.grid()
        plt.title(column)
        plt.ylabel('Z score')
        plt.xlabel('Date')
        plt.xlim(start_time,end_time)
        plt.savefig('processing/' + category + '_' + column.lower() + '_zscore.png')
        plt.close()
    except ValueError as e:
        print(f"Failed to show the {category} zscore plot. Error: {e}")


def show_boxplot(data,column,category):
    """
    Provide boxplot and save to the local disk.
    """

    try:

        plt.figure()
        data_array = data[column]
        sns.boxplot(data_array)
        y_label = column + ' Value'
        plt.ylabel(y_label)
        plt.title(column)
        plt.savefig('processing/'+category+'_'+column.lower()+'_box.png')
        plt.close()

    except ValueError as e:
        print(f"Failed to show the {category} boxplot. Error: {e}")
