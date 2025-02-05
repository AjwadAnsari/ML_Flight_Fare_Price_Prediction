from fastapi import FastAPI
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load and preprocess data
data = pd.read_csv("Flight_Fare.csv")
data.dropna(subset=['Route', 'Total_Stops'], inplace=True)

# Outlier removal for 'Price'
Q1 = data['Price'].quantile(0.25)
Q3 = data['Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
data = data[(data['Price'] >= lower_bound) & (data['Price'] <= upper_bound)]

# Feature engineering
data['Total_Stops'] = data['Total_Stops'].map({'4 stops': 0, '3 stops': 1, '2 stops': 2, 'non-stop': 3, '1 stop': 4})
data = pd.get_dummies(data, columns=['Source', 'Destination'], drop_first=True)

df = data.drop(columns=['Route'])
X = df.drop("Price", axis=1)
y = df["Price"]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Define request body model
class FlightFeatures(BaseModel):
    features: list

# Prediction endpoint
@app.post("/predict")
def predict_price(input_data: FlightFeatures, model_type: str = "linear"):
    features_array = np.array(input_data.features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    
    if model_type == "linear":
        prediction = linear_model.predict(features_scaled)
    elif model_type == "random_forest":
        prediction = rf_model.predict(features_scaled)
    else:
        return {"error": "Invalid model type. Choose 'linear' or 'random_forest'."}
    
    return {"predicted_price": prediction[0]}

# Run with `uvicorn filename:app --reload`
