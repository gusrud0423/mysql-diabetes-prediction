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
import joblib 

    # 여기서 모델 불러오자 

def run_ml_app() :

    st.subheader('Machine Learning')

    
    # 임신횟수                             # 최소 몇~~ 부터 최대 몇 ~ 까지 범위지정
    Pregnancies =  st.number_input('임신 횟수 입력', min_value=0)

    # 공복혈당                             
    Glucose=  st.number_input('공복혈당 입력', min_value= 0)

    # 혈압
    BloodPressure = st.number_input('혈압 입력', min_value=0)

    # 피부 두께
    SkinThickness = st.number_input('피부두께', min_value=0)

    # 인슐린
    Insulin = st.number_input('인슐린 수치 입력', min_value=0)

    #BMI
    BMI = st.number_input('BMI 수치 입력', min_value=0)

    #Diabetes pedigree function
    Diabetes_pedigree_function = st.number_input('DNA 영향력 입력', min_value=0)

    #Age
    Age = st.number_input('나이 입력', min_value=0)

#2. 예측한다
    # 2-1. 모델 불러오기
    model =  joblib.load('data/best_model.pk1')

    # 2-2. 넘파이 어레이 만든다 
    new_data = np.array( [  Pregnancies, Glucose,  BloodPressure,SkinThickness, Insulin,BMI, Diabetes_pedigree_function, Age  ] )
    
    # 2-3. 피처 스케일링 하자
    new_data = new_data.reshape(1,-1)
    

    # 2-4. 예측한다
    y_pred =  model.predict(new_data)

    # 예측 결과는 스케일링 된 결과이므로 다시 돌려야 한다
    # st.write( y_pred[0] )
    
    button = st.button('결과 보기')
   
    if y_pred == 1 :
        st.write( '당뇨병 입니다' )
    elif y_pred == 0 :
        st.write( '당뇨병 아닙니다')
    

    try :  # 셀렉트해서 화면에 보여주게 
        # 1. 커넉터로부터 커넥션을 받는다.
        connection = mysql.connector.connect(
            host = 'database-1.cyfvtkkh7ho8.us-east-2.rds.amazonaws.com',
            database = 'yhDB',
            user = 'streamlit',
            password = 'yh1234'
        )

        if connection.is_connected() :
            cursor = connection.cursor(dictionary= True) # 딕셔너리형태로 가져와라
            query = """ insert into books ( title, author_fname, author_lname, released_year, stock_quantity, pages )
                        values ( '유명한책', 'Harry', 'Gainman', '2020',345, 288 );"""

            cursor.execute(query)
            results = cursor.fetchall()
            for row in results :
                st.write(row) 


    except Error as e :

                print('디비 관련 에러 발생', e)
            
    finally :
        # 5. 모든 데이터베이스 실행 명령을 전부 끝냈으면,
        #    커서와 커넥션을 모두 닫아준다.
        cursor.close()
        connection.close()
        print("MySQL 커넥션 종료")


    