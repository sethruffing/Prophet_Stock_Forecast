# Importing packages
import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
from datetime import date
import matplotlib.pyplot as plt

# Starting the Streamlit window
def main():
    st.title('Forecast Stocks using Prophet by Meta')

    # Add some text
    st.write('Use the Prophet model by Meta to forecast stocks. Note that the model is highly dependent on the data it is trained on, so pick start and end dates that will give the model unbiased data. The training data comes from the Yahoo Finance API and the forecasted data comes from Prophet by Meta.')

    # ENTER STOCK TICKER
    ticker = st.text_input('Enter Stock Ticker', 'SPY')
    ticker = str(ticker)
    # Text input box for the start date
    start_date = st.text_input('Enter Start Date with YYYY-MM-DD format', '2000-01-01')
    start_date = str(start_date)
    # Text input box for the end date
    end_date = st.text_input('Enter End Date with YYYY-MM-DD format', date.today())
    end_date = str(end_date)
    # Button to trigger the forecast
    if st.button('Run Forecast'):
        # Loading the ticker data from yfinance
        try:
            ticker_yahoo = yf.Ticker(ticker)
            data = ticker_yahoo.history(start=start_date, end=end_date)
            if data.empty:
                st.error("Error: No data found for the specified date range.")
                return
            last_quote = data.iloc[-1] # gets current time
            data = data._append(pd.DataFrame(last_quote).transpose())
        except ValueError as e:
            st.error(f"Error: {e}")
            return
        except:
            st.error("Error: Unable to retrieve data from Yahoo Finance.")
            return

        # Adjusting columns to be readable by Prophet
        df = data["Close"].reset_index()
        df.rename(columns={'index': 'ds', 'Close': 'y'}, inplace=True)
        col = []
        for i in df['ds']:
            col.append(i.date())
        df['ds'] = col
        df = df[['ds', 'y']]

        # Making the time into datetime
        df['ds'] = pd.to_datetime(df['ds'], utc=False)

        # Using the prophet model
        model = Prophet()
        model.fit(df)

        # Plotting the Forecast
        future = model.make_future_dataframe(periods=365)
        forecast = model.predict(future)
        fig = model.plot(forecast)
        plt.title(str("Forecast for: " + ticker))
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        st.pyplot(fig)
        st.write("Below is the data produced by the forecast from Prophet. To save the data to a CSV, hover over the top right corner of the data, and click the download button.")
        st.write(forecast)

        st.write()
        st.write('This Streamlit Web App was developed by Seth Ruffing (TAMU Class of 2023) and Carter Powell (TAMU Class of 2024)')
if __name__ == "__main__":
    main()
