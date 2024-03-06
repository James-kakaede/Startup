import pandas as pd 
import numpy as ny 
import matplotlib.pyplot as plt 
import warnings
import xgboost as xgb
warnings.filterwarnings('ignore')
import streamlit as st 
import joblib
from sklearn.linear_model import LinearRegression

data = pd.read_csv('https://raw.githubusercontent.com/arib168/data/main/50_Startups.csv')
data.to_csv('startup.csv')


st.markdown("<h1 style = 'color: #83C0C1; text-align: center; font-family: helvetica '>STARTUP PROJECT</h1>", unsafe_allow_html =True)
st.markdown("<h4 style = 'margin: -30px; color: #6D2932; text-align: center; font-family: helvetica'>Designed and built by", unsafe_allow_html =True)
st.markdown('<br>', unsafe_allow_html = True)
st.markdown("<h3 style = 'margin: -30px; color: #1E1E1E; text-align: center; font-family: cursive '>kaka Tech Word", unsafe_allow_html =True)

st.markdown("<h4 style = 'margin: -30px; color: #1E1E1E; text-align: center; font-family: cursive '>Giving solution to mankind", unsafe_allow_html =True)
st.markdown('<br>', unsafe_allow_html = True)
st.image('pngwing.com (2).png', width = 150, use_column_width=True)
st.markdown("<p style = 'font-family: recursive' >By analyzing a diverse set of parameters, including Market Expense, Administrative Expense, and Research and Development Spending, our team seeks to develop a robust predictive model that can offer valuable insights into the future financial performance of startups. This initiative not only empowers investors and stakeholders to make data-driven decisions but also provides aspiring entrepreneurs with a comprehensive framework to evaluate the viability of their business models and refine their strategies for long-term success</p>", unsafe_allow_html=True)
st.markdown('<br>', unsafe_allow_html = True)
st.dataframe(data,use_container_width=True )
st.sidebar.image('pngwing.com (8).png', width = 150, use_column_width = True, caption='Welcone to house prediction')
st.sidebar.markdown('<br>', unsafe_allow_html = True)
st.sidebar.write('Feature input', )
rd_spend = st.sidebar.number_input('Research and Development Expenses', data['R&D Spend'].min(), data['R&D Spend'].max()+10000)
#admin = st.sidebar.number_input('Administrative Expense', data['Ast.sidebar.number_input('Marketing Expense')], data['Marketing Spend'].min(), data['Marketing Spend'].max()+10000)
                                                               

admin = st.sidebar.number_input('Administrative Expense', data['Administration'].min(), data['Administration'].max()+10000)
marketing = st.sidebar.number_input('Marketing Expense', data['Marketing Spend'].min(), data['Marketing Spend'].max()+10000)

st.markdown("<br>", unsafe_allow_html= True)
# Administration'].min(), data['Administration'].max()+ 10000)
# mkt_spend = 
st.write('Input Variables')
input_var = pd.DataFrame({'R&D Spend': [rd_spend], 'Administration': [admin], 'Marketing Spend': [marketing]})
st.dataframe(input_var)

model = joblib.load('startUpModel.pkl')

predicter = st.button('Predict ')
if predicter:
    prediction = model.predict(input_var)
    st.success(f'The Predicted value for your company is {prediction}')
    st.balloons()
