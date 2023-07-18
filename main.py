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
x = pd.DataFrame({'bedrooms': [3, 1, 3, 3, 2, 4, 4, 5, 4, 4, 4, 3, 4, 4, 3, 4, 6, 3, 4, 4, 3, 5, 4, 4, 4, 4, 5, 6, 4, 5,
                               3, 5, 3, 4, 4,
                               4, 3, 5, 4, 3, 3, 4, 3, 3, 5, 4, 3, 5, 3, 4, 4, 4, 4, 4, 4, 3, 4, 5, 5, 3, 4, 4, 3, 3, 3,
                               5, 4, 5, 3, 3],
                  'bathrooms': [2, 1, 2, 2, 1, 3, 3, 4, 4, 3, 2, 3, 3, 3, 2, 3, 4, 2, 4, 3, 2, 4, 6, 4, 4, 6, 5, 7, 4,
                                7, 2, 3, 2, 3, 2,
                                2, 2, 3, 3, 2, 2, 2, 3, 2, 6, 2, 3, 5, 3, 2, 4, 3, 3, 3, 3, 3, 3, 4, 5, 3, 4, 3, 3, 2,
                                2, 6, 3, 3, 2, 2],
                  'sqr_ft': [1723, 800, 1410, 1791, 1075, 1716, 3157, 3356, 2851, 3233, 930, 2400, 2279, 2318, 1780,
                             2115, 3533, 1096,
                             2110, 2426, 1998, 3204, 4250, 3300, 4137, 5306, 4525, 7900, 4493, 5957, 2147, 2975, 1555,
                             1627, 1500, 3000,
                             1400, 2695, 1606, 1374, 1605, 1500, 1308, 2200, 3131, 1633, 1294, 3657, 2086, 1251, 2204,
                             3469, 3800, 2494,
                             2486, 2000, 2085, 3100, 4215, 2493, 2335, 2000, 2591, 2175, 2264, 3319, 3406, 2619, 1663,
                             1260]})
y = pd.DataFrame({'price': [395000, 155000, 379000, 429900, 245000, 375000, 549000, 1050000, 659500, 799000, 275000,
                                629000, 625000, 489000, 349000, 579000, 565000, 379900, 625000, 560000, 469000, 785000,
                                1149000, 669900, 850000, 1150000, 1299000, 1799000, 799000, 2100000, 349900, 599900,
                                315000, 495000, 387000, 479900, 475000, 519900, 424900, 399000, 325000, 325000, 458000,
                                387500, 575000, 499500, 399000, 1089000, 460000, 399900, 675000, 600000, 899000, 599000,
                                420000, 499000, 639000, 1225000, 1199000, 430000, 999000, 839000, 700000, 795000,
                                589000, 1395000, 880000, 769000, 449000, 475000]})


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







