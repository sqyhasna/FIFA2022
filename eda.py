import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image

def run():
    st.title("FIFA 2022 Player Rating Prediction")
    data = pd.read_csv('https://raw.githubusercontent.com/FTDS-learning-materials/phase-1/refs/heads/v2.3/w1/P1W1D1PM%20-%20Machine%20Learning%20Problem%20Framing.csv')

    # Tambahkan gambar 
    img = Image.open('gambar.jpg')
    st.image(img)

    # Tambahkan dataframe
    st.dataframe(data)

    # Tampilkan grafik barplot
    # chart 1 - barplot
    st.write('### Plot Attacking Work Rate')
    fig = plt.figure(figsize = (10,5))
    sns.countplot(x='AttackingWorkRate', data = data)
    st.pyplot(fig)

    # chart 2 - histogram
    st.write('### Plot Histogram for Price')
    fig = plt.figure(figsize = (10,5))
    # options
    options = st.selectbox("Select Column fir Histogram", {'Age', 'Overall', 'ValueEUR'})
    sns.histplot(data[options], bins = 20, kde = True)
    st.pyplot(fig)

    # chart 3 - plotly express
    st.write('### Scatter plot for ValueEUR & Rating Player')
    fig = px.scatter(data, x = 'ValueEUR', y = 'Overall', hover_data = ['Name', 'Age'])
    st.plotly_chart(fig)

if __name__== "__main__":
    run()