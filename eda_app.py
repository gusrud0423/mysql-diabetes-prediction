import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import h5py
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import pickle

def run_eda_app():
    st.subheader('EDA 화면입니다.')

    diabetes_df = pd.read_csv('data/diabetes.csv') 
             
    radio_menu = ['데이터 프레임', '통계치', '상관관계 분석']
    selectbox_radio =  st.radio('선택하세요', radio_menu)

    if selectbox_radio == '데이터 프레임' :
        st.dataframe( diabetes_df )
    elif selectbox_radio == '통계치' :
        st.dataframe( diabetes_df.describe() )
    elif selectbox_radio == '상관관계 분석' :
        st.dataframe( diabetes_df.corr() )


    # # 컬럼 가져와서 선택한것만 보이게 하라 
    columns = diabetes_df.columns
    columns = list(columns)

    corr_columns = diabetes_df.columns
    selected_columns = st.multiselect('컬럼을 선택하시오', columns)
    if len(selected_columns) != 0 :
        st.dataframe( diabetes_df[selected_columns] )   # empty 안나오게
    # #st.dataframe(diabetes_df[selected_columns]) 

    
    # # 상관관계 분석 화면에 보여주도록 만듦
    # # 멀티셀렉트에 컬럼명을 보여주고,
    # # 해당 컬럼들에 대한 상관계수를 보여주세요
    # # 단, 컬럼들은 숫자컬럼들만 멀티셀렉트에 나타나야 됨
    
    # # 컬럼만 가져와서는 그 컬럼이 숫자인지 문자인지 모르니 dtype 해야함
   
    
    selected_corr = st.multiselect('상관계수 컬럼 선택', corr_columns)
    
    if len(selected_corr) > 0 :

        st.dataframe(diabetes_df[selected_corr].corr())
        
    #     # 위에서 선택한 컬럼들을 이용해서, seaborn 에 페어플롯을 그린다
    #     # if 문 안에서 해야 선택한 컬럼들을 가지고 할수있다
        
        fig = sns.pairplot(data =  diabetes_df[selected_corr])
        st.pyplot(fig)


    else :
        st.write('선택한 컬럼이 없습니다')


    #     # 컬럼을 하나만 선택하면, 해당 컬럼의 min 과 max 에 
    #     # 해당하는 사람의 데이터를 화면에 보여주는 기능 개발
    number_columns = diabetes_df.columns[ diabetes_df.dtypes != object ]
    selected_col = st.selectbox( '컬럼 선택', number_columns ) 

    min_data =  diabetes_df[selected_col] == diabetes_df[selected_col].min() 
    st.write('최소값 데이터') 
    st.dataframe( diabetes_df.loc[ min_data ] )

    max_data =  diabetes_df[selected_col] == diabetes_df[selected_col].max() 
    st.write('최대값 데이터') 
    st.dataframe( diabetes_df.loc[ max_data ] )


 
    
    




   

    