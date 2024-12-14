from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open("stock.pkl", "rb") as f:
    model = pickle.load(f)

# Prediction function
def predict_stock(
    prev_close, open_price, high, low, last, close,
    vwap, volume, turnover, trades, deliverable_volume, percent_deliverable
):
    temp_array = [
        prev_close, open_price, high, low, last, close,
        vwap, volume, turnover, trades, deliverable_volume
    ]
    
    # Convert to NumPy array and reshape
    temp_array = np.array([temp_array])  # Ensure input is 2D
    
    # Predict using the trained model
    predicted_value = model.predict(temp_array)
    return int(predicted_value[0])

# Routes
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        prev_close = float(request.form.get('Prev_Close'))
        open_price = float(request.form.get('Open'))
        high = float(request.form.get('High'))
        low = float(request.form.get('Low'))
        last = float(request.form.get('Last'))
        close = float(request.form.get('Close'))
        vwap = float(request.form.get('VWAP'))
        volume = int(request.form.get('Volume'))
        turnover = float(request.form.get('Turnover'))
        trades = int(request.form.get('Trades'))
        deliverable_volume = int(request.form.get('Deliverable_Volume'))
        percent_deliverable = float(request.form.get('percent_deliverable'))

        # Get prediction
        prediction = predict_stock(
            prev_close, open_price, high, low, last, close,
            vwap, volume, turnover, trades, deliverable_volume, percent_deliverable
        )
        print(prediction)

        # Render result
        return render_template('result.html', Prediction=prediction)
    
    # Render predict page for GET method
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True,port=4500, host="0.0.0.0")
