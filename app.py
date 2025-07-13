import streamlit as st
import pandas as pd
import joblib

# 모델 불러오기
model = joblib.load('seoul_poll_model.pkl')

st.title("서울 미세먼지 예측기")

stations_df = pd.read_csv('seoul_air_checkLocation.csv' ,encoding='euc-kr')

selected_gu = st.selectbox("서울 구 선택", stations_df['측정소 이름'].unique())
# 사용자 입력 받기
year = st.number_input('년도', min_value=2000, max_value=2100, value=2024, step=1)
month = st.number_input('월', min_value=1, max_value=12, value=6, step=1)
day = st.number_input('일', min_value=1, max_value=31, value=23, step=1)
hour = st.number_input('시', min_value=0, max_value=23, value=12, step=1)


if st.button('미세먼지 예측하기'):
    # 입력값을 데이터프레임으로 만들기
    input_data = pd.DataFrame({
        '년도': [year],
        '월': [month],
        '일': [day],
        '시': [hour],
        '측정소 이름': [selected_gu]
    })
    input_df = pd.get_dummies(input_data)

    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    # 예측
    prediction = model.predict(input_df)
    
    st.write(f"예측된 미세먼지 값: {prediction[0]:.2f}")
