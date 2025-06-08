import os
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app= application

## Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = CustomData(
            price=request.form.get('price', type=float),
            h1=request.form.get('h1', type=float),        # 1h 
            h24=request.form.get('h24', type=float),      # 24h
            d7=request.form.get('d7', type=float),        # 7d
            volume_24h=request.form.get('volume_24h', type=float),
            mkt_cap=request.form.get('mkt_cap', type=float),
            price_change_ratio=request.form.get('price_change_ratio', type=float),
            volume_to_price=request.form.get('volume_to_price', type=float),
            is_stable_coin=request.form.get('is_stable_coin', type=int)
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)[0]  # Get the first element (scalar)

        return render_template("predict.html", prediction=round(prediction, 3))
    
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
