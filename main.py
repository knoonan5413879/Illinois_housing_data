import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import shap
import requests
from sklearn.ensemble import RandomForestRegressor

# App title
st.write("""
# Illinois House Price App
Predict house prices and check affordability based on mortgage rates
""")
st.write('---')

# Load the housing data
x = pd.DataFrame({
    'bedrooms': [3, 1, 3, 3, 2, 4, 4, 5, 4, 4, 4, 3, 4, 4, 3, 4, 6, 3, 4, 4, 3, 5, 4, 4, 4, 4, 5, 6, 4, 5,
                 3, 5, 3, 4, 4, 4, 3, 5, 4, 3, 3, 4, 3, 3, 5, 4, 3, 5, 3, 4, 4, 4, 4, 4, 4, 3, 4, 5, 5, 3,
                 4, 4, 3, 3, 3, 5, 4, 5, 3, 3],
    'bathrooms': [2, 1, 2, 2, 1, 3, 3, 4, 4, 3, 2, 3, 3, 3, 2, 3, 4, 2, 4, 3, 2, 4, 6, 4, 4, 6, 5, 7, 4, 7,
                  2, 3, 2, 3, 2, 2, 2, 3, 3, 2, 2, 2, 3, 2, 6, 2, 3, 5, 3, 2, 4, 3, 3, 3, 3, 3, 3, 4, 5, 3,
                  4, 3, 3, 2, 2, 6, 3, 3, 2, 2],
    'sqr_ft': [1723, 800, 1410, 1791, 1075, 1716, 3157, 3356, 2851, 3233, 930, 2400, 2279, 2318, 1780, 2115,
               3533, 1096, 2110, 2426, 1998, 3204, 4250, 3300, 4137, 5306, 4525, 7900, 4493, 5957, 2147,
               2975, 1555, 1627, 1500, 3000, 1400, 2695, 1606, 1374, 1605, 1500, 1308, 2200, 3131, 1633,
               1294, 3657, 2086, 1251, 2204, 3469, 3800, 2494, 2486, 2000, 2085, 3100, 4215, 2493, 2335,
               2000, 2591, 2175, 2264, 3319, 3406, 2619, 1663, 1260]
})

y = pd.DataFrame({
    'price': [395000, 155000, 379000, 429900, 245000, 375000, 549000, 1050000, 659500, 799000, 275000,
              629000, 625000, 489000, 349000, 579000, 565000, 379900, 625000, 560000, 469000, 785000,
              1149000, 669900, 850000, 1150000, 1299000, 1799000, 799000, 2100000, 349900, 599900,
              315000, 495000, 387000, 479900, 475000, 519900, 424900, 399000, 325000, 325000, 458000,
              387500, 575000, 499500, 399000, 1089000, 460000, 399900, 675000, 600000, 899000, 599000,
              420000, 499000, 639000, 1225000, 1199000, 430000, 999000, 839000, 700000, 795000, 589000,
              1395000, 880000, 769000, 449000, 475000]
})
#adds for a 5 percent increase in prices
y = y * 1.05

# Sidebar input
st.sidebar.header("Specify Input Parameters")
def user_input_features():
    bedrooms = st.sidebar.slider('Bedrooms', x.bedrooms.min(), x.bedrooms.max(), int(x.bedrooms.mean()))
    bathrooms = st.sidebar.slider('Bathrooms', x.bathrooms.min(), x.bathrooms.max(), int(x.bathrooms.mean()))
    sqr_ft = st.sidebar.slider('Square Footage', x.sqr_ft.min(), x.sqr_ft.max(), int(x.sqr_ft.mean()))
    data = {'bedrooms': bedrooms, 'bathrooms': bathrooms, 'sqr_ft': sqr_ft}
    return pd.DataFrame(data, index=[0])

df = user_input_features()

# Train the model
model = RandomForestRegressor()
model.fit(x, y)

# Predict price
prediction = model.predict(df)
predicted_price = prediction[0]

# Mortgage calculator function
def calculate_monthly_payment(loan_amount, annual_interest_rate, loan_term_years):
    monthly_rate = annual_interest_rate / 100 / 12
    number_of_payments = loan_term_years * 12
    monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**number_of_payments) / ((1 + monthly_rate)**number_of_payments - 1)
    return monthly_payment

# Tabs
tab1, tab2 = st.tabs(["🏠 House Price Prediction", "💵 Affordability Calculator"])

# --- Tab 1: House Price Prediction ---
with tab1:
    st.header('Specified Input Parameters')
    st.write(df)
    st.write('---')

    st.header('Prediction of Price')
    st.subheader(f"${predicted_price:,.2f}")
    st.write('---')

    # SHAP Feature Importance
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)

    st.header('Feature Importance')
    plt.title('Feature Importance (SHAP Values)')
    shap.summary_plot(shap_values, x)
    st.pyplot(bbox_inches='tight')
    st.write('---')

    plt.title('Feature Importance (SHAP Values - Bar Chart)')
    shap.summary_plot(shap_values, x, plot_type="bar")
    st.pyplot(bbox_inches='tight')
    st.write('---')

# --- Tab 2: Affordability Calculator ---
# --- Tab 2: Affordability Calculator ---
with tab2:
    st.header("Affordability Calculator")

    # Fetch or input interest rate
    interest_rate = st.slider('Select Mortgage Interest Rate (%)', 2.0, 10.0, 6.5)
    loan_term = st.selectbox('Select Loan Term (Years)', [15, 30], index=1)

    # New inputs: property tax and insurance
    st.subheader("Estimated Other Costs:")
    property_tax_rate = st.slider('Annual Property Tax Rate (%)', 1.0, 4.0, 2.25)  # IL avg ~2.25%
    annual_home_insurance = st.slider('Annual Home Insurance Estimate ($)', 500, 5000, 1200)

    # Calculations
    mortgage_payment = calculate_monthly_payment(predicted_price, interest_rate, loan_term)
    monthly_property_tax = (predicted_price * (property_tax_rate / 100)) / 12
    monthly_home_insurance = annual_home_insurance / 12
    total_monthly_payment = mortgage_payment + monthly_property_tax + monthly_home_insurance

    # Display results
    st.subheader('Predicted House Price:')
    st.write(f"${predicted_price:,.2f}")

    st.subheader('Estimated Monthly Mortgage Payment (Principal + Interest Only):')
    st.write(f"${mortgage_payment:,.2f}")

    st.subheader('Estimated Monthly Property Taxes:')
    st.write(f"${monthly_property_tax:,.2f}")

    st.subheader('Estimated Monthly Home Insurance:')
    st.write(f"${monthly_home_insurance:,.2f}")

    st.header('🏡 Estimated Total Monthly Housing Cost:')
    st.success(f"${total_monthly_payment:,.2f} per month")

    st.write("""
    _Note: These estimates do not include HOA fees, maintenance costs, or other potential costs._
    """)
    st.write('---')








