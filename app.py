# app.py

import streamlit as st
import joblib
import numpy as np

# 저장된 모델 불러오기
model = joblib.load('mymodel.pkl')

st.title("미세먼지(PM2.5) 예측기")

seoul_gu = ['강남구', '강동구', '강북구', '강서구', '관악구' ,'광진구', '구로구' ,'금천구' ,'노원구' ,'도봉구', '동대문구',
 '동작구', '마포구' ,'서대문구', '서초구', '성동구', '성북구', '송파구' ,'양천구', '영등포구', '용산구' ,'은평구' ,'종로구',
 '중구', '중랑구'] # 원하는 값들 리스트

years = [2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023]
months = [1,2,3,4,5,6,7,8,9,10,11,12]
days = list(range(1, 32))
feature1 = st.selectbox("서울 구 선택", seoul_gu)  # index=2면 기본값 20
feature2 = st.selectbox("년", years , index=1)  # index=2면 기본값 20
feature3= st.selectbox("월", months , index=1)  # index=2면 기본값 20
feature4= st.selectbox("일", days , index=1)  # index=2면 기본값 20
# 필요한 feature 더 추가

# 예측 버튼
if st.button("예측하기"):
    X_input = {
        '구분' : [feature1],
        '년' : [feature2],
        '월' : [feature3],
        '일' : [feature4],
    }
    import pandas as pd
    X_input = pd.DataFrame(X_input)
    X_input['구분'] = X_input['구분'].astype('category')

    import numpy as np
    pred = model.predict(X_input)
    st.write(f"예측된 PM2.5 농도: {np.expm1(pred[0]):.2f} ㎍/m³")
