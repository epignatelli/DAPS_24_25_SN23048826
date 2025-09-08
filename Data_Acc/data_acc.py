import yfinance as yf
import pandas as pd
import requests

def acquire_datasets(name):
    """
    Acquire dataset using the given input (dataset name)
    """
    if name == "stock":
        return acquire_stock_data()
    if name == "covid":
        return acquire_covid_data()
    if name == "interest":
        return acquire_interest_data()


def acquire_stock_data():
    """
    Acquire stocks data for S&P 500 by using yfinance python library
    """
    # select S&P 500 group
    stock = yf.Ticker("^GSPC")
    # Acquire stock history with 1 day interval
    stock_data = stock.history(start="2017-04-01", end="2024-04-01", interval="1d")
    stock_data = stock_data.reset_index()

    # Update the date column format to %Y-%m-%d
    for i in range(len(stock_data)):
        current_date = stock_data.at[i, "Date"]
        datetime_conversion = pd.to_datetime(current_date)
        updated_date = datetime_conversion.strftime('%Y-%m-%d')
        stock_data.at[i, "Date"] = updated_date

    # Convert the data into a dictionary
    stock_data = stock_data.to_dict('records')
    return stock_data


def acquire_interest_data():
    """
    Acquire interest data for New York by using newyorkfed API.
    """
    # Access new york interest rate data
    start_date = "2017-04-01"
    end_date = "2024-04-01"
    interest_url = f"https://markets.newyorkfed.org/api/rates/all/" \
    f"search.json?startDate={start_date}&endDate={end_date}&type=rate"

    interest_json = requests.get(interest_url, timeout=10).json()
    interest_dataframe = pd.DataFrame.from_records(interest_json["refRates"])
    # Convert the data into a dictionary
    interest_data = interest_dataframe.to_dict('records')
    return interest_data



def acquire_covid_data():
    """
    Acquire covid data for New York by using cityofnewyork API.
    """
    # Access new york covid statistics data
    covid_url = "https://data.cityofnewyork.us/resource/rc75-m7u3.json?$limit=1500"

    covid_json = requests.get(covid_url, timeout=10).json()
    covid_dataframe = pd.DataFrame.from_records(covid_json)
    # Update the date column format to %Y-%m-%d
    for i in range(len(covid_dataframe)):
        current_date = covid_dataframe.at[i, "date_of_interest"]
        datetime_conversion = pd.to_datetime(current_date)
        updated_date = datetime_conversion.strftime('%Y-%m-%d')
        covid_dataframe.at[i, "date_of_interest"] = 0
        covid_dataframe.at[i, "date_of_interest"] = updated_date

    # Convert the data into a dictionary 
    covid_data = covid_dataframe.to_dict('records')
    return covid_data


