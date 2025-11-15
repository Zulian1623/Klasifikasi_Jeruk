import pandas as pd
import streamlit as st
import pickle

st.set_page_config(
    page_title='Belajar Klasifikasi Jeruk',
    page_icon=":tangerine:"
)

with open("model_jeruk.pkl", "rb") as f:
    model = pickle.load(f)


st.title(':tangerine: Belajar Klasifikasi Jeruk')
st.markdown('Model Machine Learning Untuk Mengklasifikasi Jeruk')

diameter = st.slider("Diameter",3.9,9.0,6.5)
berat = st.slider("Berat",70.0,280.0,150.0)
tebal_kulit = st.slider("Tebal Kulit",0.1,2.0,1.3)
kadar_gula = st.slider("Kadar Gula",5.5,15.0,10.0)
asal_daerah = st.pills("Asal Daerah", ['Kalimantan','Jawa Barat','Jawa Tengah'], default='Kalimantan')
warna = st.pills("Warna", ['hijau','kuning','oranye'], default='hijau')
musim_panen = st.pills("Musim Panen", ['kemarau','hujan'], default='kemarau')

if st.button("Prediksi", type="primary"):
    data_baru = pd.DataFrame([[diameter,berat,tebal_kulit,kadar_gula,asal_daerah,warna,musim_panen]], 
                             columns=['diameter', 'berat', 'tebal_kulit', 'kadar_gula', 'asal_daerah','warna', 'musim_panen'])
    prediksi = model.predict(data_baru)[0]
    persentase = max(model.predict_proba(data_baru)[0])
    st.success(f"Model memprediksi **{prediksi}** dengan tingkat keyakinan **{persentase*100:2f}%**")
    st.balloons()

st.divider()
st.caption("Dibuat dengan :tangerine: oleh **Zulian_DEV**")