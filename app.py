from flask import Flask, render_template, jsonify, request
import pandas as pd
import requests
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yfinance as yf

app = Flask(__name__, template_folder="templates")

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = Path() / "static" / f"{fig_id}.{fig_extension}"
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
    return path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_api', methods=["POST"])
def predict_api():

    ticker = request.args.get('ticker')

    if 'ticker' in request.args.keys():

        # Get the ticker object
        stock = yf.Ticker(ticker)

        # Get real-time price data
        price = stock.history(period="2y")['Close']
        df = pd.DataFrame(price)

        # df = pd.read_csv('nsei-stock-data.csv')

        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.40)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.40): int(len(df))])

        scaler = joblib.load('stock-price-prediction-scaler.pkl')

        sequence_length = 200
        past_200_days = data_training.tail(sequence_length+1)
        final_df = pd.concat([past_200_days,data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test, y_test = [], []
        for i in range(sequence_length+1, input_data.shape[0]):
            x_test.append(input_data[i - sequence_length:i])
            y_test.append(input_data[i, 0])
        X_test, y_test = np.array(x_test), np.array(y_test)

        model = joblib.load("stock-price-prediction-keras-model.pkl")

        prediction = model.predict(X_test)

        prediction_unscaled = scaler.inverse_transform(prediction.reshape(-1, 1))
        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        mean_diff = np.mean(y_test_unscaled - prediction_unscaled)


        print(prediction_unscaled[-1].tolist()[0] + mean_diff)
        output = {
            'Prediction': prediction_unscaled[-1].tolist()[0],  
            'Acutal': y_test_unscaled[-1].tolist()[0],
            'Mean Diff': mean_diff,
            'Prediction + Mean Diff': prediction_unscaled[-1].tolist()[0] + mean_diff,
            'Prediction - Mean Diff': prediction_unscaled[-1].tolist()[0] - mean_diff
        }

        print(output)
        return jsonify(output)
    else :
        return "Please pass the parameter ticker"


@app.route('/predict', methods=["GET", "POST"])
def predict():

    ticker = request.args.get('ticker')
    
    if 'ticker' in request.args.keys():

        # # Get the ticker object
        stock = yf.Ticker(ticker)

        # Get real-time price data
        price = stock.history(period="2y")['Close']
        df = pd.DataFrame(price)

        # df = pd.read_csv('nsei-stock-data.csv')

        ema100 = df.Close.ewm(span=100, adjust=False).mean()
        ema200 = df.Close.ewm(span=200, adjust=False).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(df.Close, 'y', label='Closing Price')
        plt.plot(ema100, 'g', label='EMA 100')
        plt.plot(ema200, 'r', label='EMA 200')
        plt.title("Closing Price vs Time (100 & 200 Days EMA)")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        ema_chart_path_100_200 = save_fig("ema_100_200")

        data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.40)])
        data_testing = pd.DataFrame(df['Close'][int(len(df)*0.40): int(len(df))])

        scaler = joblib.load('stock-price-prediction-scaler.pkl')

        sequence_length = 200
        past_200_days = data_training.tail(sequence_length+1)
        final_df = pd.concat([past_200_days,data_testing], ignore_index=True)
        input_data = scaler.fit_transform(final_df)

        x_test, y_test = [], []
        for i in range(sequence_length+1, input_data.shape[0]):
            x_test.append(input_data[i - sequence_length:i])
            y_test.append(input_data[i, 0])
        X_test, y_test = np.array(x_test), np.array(y_test)

        model = joblib.load("stock-price-prediction-keras-model.pkl")

        prediction = model.predict(X_test)

        prediction_unscaled = scaler.inverse_transform(prediction.reshape(-1, 1))
        y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

        mean_diff = np.mean(y_test_unscaled - prediction_unscaled)

        ## Actual vs Predicted Chart
        plt.figure(figsize=(12, 6))
        plt.plot(y_test_unscaled, 'r', label='Actual Price')
        plt.plot(prediction_unscaled, 'g', label='Predicted Price')
        plt.title("Acutal Vs Predicted")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        actual_predicted_path = save_fig("acutal_predicted")

        mean_diff = np.mean(y_test_unscaled - prediction_unscaled)

        print(prediction_unscaled[-1].tolist()[0] + mean_diff)
        output = {
            'Prediction': np.round(prediction_unscaled[-1].tolist()[0], 2),  
            'Acutal': np.round(y_test_unscaled[-1].tolist()[0], 2),
            'Mean Diff': np.round(mean_diff, 2),
            'Prediction + Mean Diff': np.round(prediction_unscaled[-1].tolist()[0] + mean_diff, 2),
            'Prediction - Mean Diff': np.round(prediction_unscaled[-1].tolist()[0] - mean_diff, 2)
        }

        print(output)
        return render_template('index.html', ticker=ticker, output=output, actual_predicted = actual_predicted_path, ema_chart_100_200 = ema_chart_path_100_200)
    else :
        return render_template('index.html', message="Invalid Request")
    


if __name__ == "__main__":
    app.run(debug=True)