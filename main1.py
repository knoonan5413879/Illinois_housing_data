import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import shap
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor

st.write("""
#Illinois House Price App

This app shows the home prices in Illinois**
""")
st.write('---')

# load the housing data
Illinois = pd.read_csv("C:\\Users\\Kevin\\OneDrive\\Desktop\\homes_data.csv")
x = pd.DataFrame(Illinois, columns=["bathrooms", "bedrooms", "sqr_ft"])
y = pd.DataFrame(Illinois, columns=["price"])

# Sidebar
# Header that specifies the input parameters
st.sidebar.header("Specify Input Parameters")


def user_input_features():
    bedrooms = st.sidebar.slider('bedrooms', x.bedrooms.min(), x.bedrooms.max(), x.bedrooms.mean())
    bathrooms = st.sidebar.slider('bathrooms', x.bathrooms.min(), x.bathrooms.max(), x.bathrooms.mean())
    sqr_ft = st.sidebar.slider('sqr_ft', x.sqr_ft.min(), x.sqr_ft.max(), x.sqr_ft.mean())
    data = {'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'sqr_ft': sqr_ft}
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

# print specified input parameters
st.header('Specified Input Parameters')
st.write(df)
st.write('---')

# Build a regression model to make a prediction
model = RandomForestRegressor()
model.fit(x, y)

# apply the model
prediction = model.predict(df)

st.header('Prediction of price')
st.write(prediction)
st.write('---')

# using SHAP values to explain the model's predictions
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x)

st.header('Feature Importance')
plt.title('Feature Importance from the SHAP values')
shap.summary_plot(shap_values, x)
st.pyplot(bbox_inches='tight')
st.write('---')

plt.title('Feature importance based on SHAP values bar chart')
shap.summary_plot(shap_values, x, plot_type="bar")
st.pyplot(bbox_inches='tight')







