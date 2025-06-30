# 🏡 BHPredictor

**BHPredictor** is a machine learning-powered web application that predicts the **median value of owner-occupied homes** in Boston suburbs. It is built using the [Boston Housing Dataset](https://archive.ics.uci.edu/ml/datasets/housing) (originally published on **July 7, 1993**) and provides real-time price predictions based on 13 key housing features.

## 🚀 Features

- Predict house prices in Boston with 13 input features
- Clean and interactive Streamlit UI
- Real-time inference using a trained ML model
- Based on the classic Boston Housing dataset from StatLib (Carnegie Mellon)

## 📊 Dataset Info

The dataset contains 506 samples with the following features:

| Feature     | Description |
|-------------|-------------|
| CRIM        | Crime rate per capita |
| ZN          | % of residential land zoned for large lots |
| INDUS       | % of non-retail business acres |
| CHAS        | Bounds Charles River (1=yes, 0=no) |
| NOX         | Nitric oxides concentration |
| RM          | Avg. number of rooms |
| AGE         | % built before 1940 |
| DIS         | Distance to employment centers |
| RAD         | Highway access index |
| TAX         | Property tax rate |
| PTRATIO     | Pupil-teacher ratio |
| B           | 1000(Bk - 0.63)^2 |
| LSTAT       | % lower status of population |
| MEDV        | (Target) Median value in $1000's |

## 🧠 ML Model

- **Algorithm:** RandomForestRegressor
- **Target:** `MEDV` (Median home value)
- **Libraries:** `scikit-learn`, `pandas`, `streamlit`, `joblib`

## 🖥️ Tech Stack

- Python
- Streamlit
- scikit-learn
- joblib
- pandas
- numpy
- matplotlib

## 🛠️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-username/BHPredictor.git
cd BHPredictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
