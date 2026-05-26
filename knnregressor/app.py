import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------
# PAGE TITLE
# ---------------------------------------------------

st.title("KNN Regressor Application")

st.write("House Price Prediction using K-Nearest Neighbors Regressor")

# ---------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------

data = fetch_california_housing()

df = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

df["Price"] = data.target

# ---------------------------------------------------
# DISPLAY DATASET
# ---------------------------------------------------

st.subheader("Dataset")

st.dataframe(df.head())

# ---------------------------------------------------
# FEATURES AND TARGET
# ---------------------------------------------------

X = df.drop("Price", axis=1)

y = df["Price"]

# ---------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ---------------------------------------------------
# KNN MODEL
# ---------------------------------------------------

k = st.slider("Select K Value", 1, 20, 5)

model = KNeighborsRegressor(n_neighbors=k)

model.fit(X_train, y_train)

# ---------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------

predictions = model.predict(X_test)

# ---------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------

mse = mean_squared_error(y_test, predictions)

r2 = r2_score(y_test, predictions)

st.subheader("Model Performance")

st.write(f"Mean Squared Error : {mse:.2f}")

st.write(f"R2 Score : {r2:.2f}")

# ---------------------------------------------------
# USER INPUTS
# ---------------------------------------------------

st.subheader("Predict House Price")

MedInc = st.number_input("Median Income", value=3.0)

HouseAge = st.number_input("House Age", value=20.0)

AveRooms = st.number_input("Average Rooms", value=5.0)

AveBedrms = st.number_input("Average Bedrooms", value=1.0)

Population = st.number_input("Population", value=1000.0)

AveOccup = st.number_input("Average Occupancy", value=3.0)

Latitude = st.number_input("Latitude", value=34.0)

Longitude = st.number_input("Longitude", value=-118.0)

# ---------------------------------------------------
# PREDICTION BUTTON
# ---------------------------------------------------

if st.button("Predict Price"):

    input_data = np.array([[
        MedInc,
        HouseAge,
        AveRooms,
        AveBedrms,
        Population,
        AveOccup,
        Latitude,
        Longitude
    ]])

    prediction = model.predict(input_data)

    st.success(
        f"Predicted House Price : {prediction[0]:.2f}"
    )

# ---------------------------------------------------
# GRAPH
# ---------------------------------------------------

st.subheader("Actual vs Predicted")

fig, ax = plt.subplots(figsize=(8,5))

ax.scatter(y_test, predictions)

ax.set_xlabel("Actual Prices")

ax.set_ylabel("Predicted Prices")

ax.set_title("KNN Regressor Predictions")

st.pyplot(fig)