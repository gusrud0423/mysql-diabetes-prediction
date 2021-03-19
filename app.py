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

from sklearn.ensemble import RandomForestClassifier

from eda_app import run_eda_app
from ml_app import run_ml_app
     # ml_app2.py 파일에서 가져온 run_ml_app 함수 임폴트

def main() :
    st.title('당뇨병 판정 예측')

    # 사이드바 메뉴
    menu = ['Home', 'EDA','ML']  #'ML'
    choice =  st.sidebar.selectbox('메뉴', menu)

    if choice == 'Home' :
        st.write('이 앱은 당뇨병 판정에 대한 데이터에 대한 내용입니다. 사람의 정보를 입력하면 당뇨병인지 아닌지를 예측하는 앱입니다.')
        st.write('왼쪽의 사이드바에서 선택하세요.')
    
    elif choice == 'EDA' :

        run_eda_app()
    elif choice == 'ML' :

        run_ml_app()




if __name__ == '__main__' :
    main()
