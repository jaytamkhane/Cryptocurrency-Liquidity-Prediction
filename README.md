
# 💸 Cryptocurrency Liquidity Prediction

A machine learning project to **predict the liquidity** of cryptocurrencies using market metrics like price changes, volume, and market cap. The model is deployed via a **Flask web app** on **Render**.

---

## 🧠 Problem Statement

Predict the `liquidity` of a crypto coin using features like:

- `coin`, `Price`, `1h`, `24h`, `7d`
- `24h_volume`, `market_cap`, `date`

---

## 📁 Dataset

Two CSVs from CoinGecko:

- `coin_gecko_2022-03-16.csv`
- `coin_gecko_2022-03-17.csv`

📂 Located in: `notebook/data/`

---

## 📊 EDA Summary

- Handled missing values using mean imputation
- Found strong correlation between `volume`, `market_cap`, and `liquidity`
- Combined and cleaned datasets for model input

📂 Notebook: `notebook/EDA.ipynb`

---

## 🤖 Models Used

Trained several regressors:

- Ridge, Lasso, Linear Regression 
- Decision Tree
- Random Forest
- XGBoost Regressor
- ✅ CatBoost Regressor (Best)
- AdaBoost
- SVR

📂 Code: `notebook/MODEL TRAINING.ipynb`

---

## 📈 Evaluation Results

| Model        | R² Train | R² Test  | 
|--------------|----------|----------|
| RandomForest | 0.9741   | 0.8309   |
| DecisionTree | 1.0000   | 0.4587   |
| AdaBoost     | 0.9390   | 0.8832   |
| XGBoost      | 1.0000   | 0.5955   |
| CatBoost     | 0.9999   | 0.9878   |
| SVR          | 0.2128   | 0.0369   |

✅ **Best Model:** catBoost

---

## 🌐 Live Deployment

🟢 Web App is hosted on **Render**

**🔗 Live Demo:** [Crypto_Liquidity_Prediction_Project](https://cryptocurrency-liquidity-prediction-a1qg.onrender.com)

📂 Deployment files:
- `app.py`
- `templates/`
- `render.yaml`, `runtime.txt`
- `requirements.txt`

---

## 🗂️ Project Structure

```
Cryptocurrency-Liquidity-Prediction/
├── artifacts/ # Saved models, datasets
│ ├── model.pkl
│ ├── preprocessor.pkl
│ ├── data.csv, train.csv, test.csv
├── notebook/
│ ├── EDA.ipynb
│ └── MODEL TRAINING.ipynb
│ └── data/
├── src/
│ ├── components/
│ │ ├── data_ingestion.py
│ │ ├── data_transformation.py
│ │ └── model_trainer.py
│ ├── pipeline/
│ │ ├── predict_pipeline.py
│ │ └── init.py
│ ├── logger.py, utils.py, exception.py
├── templates/ # HTML frontend
├── static/ # Static CSS
├── app.py # Flask app
├── render.yaml, runtime.txt, requirements.txt
├── setup.py
└── README.md
```

---

## 🧪 Run Locally

```bash
git clone https://github.com/jaytamkhane/Cryptocurrency-Liquidity-Prediction
cd Cryptocurrency-Liquidity-Prediction

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt

python app.py
```

---

## 👨‍💻 Developed by

**Jay Tamkhane**  
📧 jaytamkhane161@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/jay-tamkhane)
