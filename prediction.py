import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# load all files
with open('model_lin_reg.pkl', 'rb') as file_1:
  model_lin_reg = pickle.load(file_1)

with open('model_scaler.pkl', 'rb') as file_2:
  model_scaler = pickle.load(file_2)

with open('model_encoder.pkl','rb') as file_3:
  model_encoder = pickle.load(file_3)

with open('list_num_cols.txt', 'r') as file_4:
  list_num_cols = json.load(file_4)

with open('list_cat_cols.txt', 'r') as file_5:
  list_cat_cols = json.load(file_5)

def run():
    with st.form('form_fifa_2022'):
        name = st.text_input('Name: ', value = ' ')
        age = st.number_input('Age: ', min_value = 16, max_value = 60, value = 25)
        height = st.slider('Height: ', min_value = 150, max_value = 200, value = 170)
        weight = st.slider('Weight: ', min_value = 40, max_value = 100, value = 70)
        price = st.number_input('Price: ', min_value = 50, value = 1000000)
        attackingworkrate = st.selectbox('Attacking Work Rate: ', ['Low', 'Medium', 'High'])
        defensiveworkrate = st.selectbox('Defensive Work Rate: ', ['Low', 'Medium', 'High'])
        pace = st.number_input('Pace Total (0-100): ', min_value = 0, max_value = 100, value = 50)
        shooting = st.number_input('Shooting Total (0-100): ', min_value = 0, max_value = 100, value = 50)
        passing = st.number_input('Passing Total (0-100): ', min_value = 0, max_value = 100, value = 50)
        dribbling = st.number_input('Dribbling Total (0-100): ', min_value = 0, max_value = 100, value = 50)
        defending = st.number_input('Defending Total (0-100): ', min_value = 0, max_value = 100, value = 50)
        physicality = st.number_input('Physicality Total (0-100): ', min_value = 0, max_value = 100, value = 50)

        submit = st.form_submit_button('Predict')

    data_inf = {
        'Name': name,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'Price': price,
        'AttackingWorkRate': attackingworkrate,
        'DefensiveWorkRate': defensiveworkrate,
        'PaceTotal': pace,
        'ShootingTotal': shooting,
        'PassingTotal': passing,
        'DribblingTotal': dribbling,
        'DefendingTotal': defending,
        'PhysicalityTotal': physicality
    }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    if submit:
        data_inf_num = data_inf[list_num_cols]
        data_inf_cat = data_inf[list_cat_cols]
        ## Feature Scaling
        data_inf_num_scaled = model_scaler.transform(data_inf_num)

        ## Feature Encoding
        data_inf_cat_encoded = model_encoder.transform(data_inf_cat)

        ## Concate
        data_inf_final = np.concatenate([data_inf_num_scaled, data_inf_cat_encoded], axis=1)
        # predict using linear regression
        y_pred_inf = model_lin_reg.predict(data_inf_final)
        st.write('### Rating: ', str(int(y_pred_inf[0])))


if __name__ == "__main__":
  run()