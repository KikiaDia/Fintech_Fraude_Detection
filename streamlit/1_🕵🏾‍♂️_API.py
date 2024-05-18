import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import io

st.set_page_config(
  page_title='API',
  layout='wide',
  page_icon='📜'
)

st.sidebar.image('https://www.creativefabrica.com/wp-content/uploads/2021/07/05/Fraud-Detection-icon-Graphics-14301492-1.jpg')

st.title('🧠 Detection de Fraudes : Transactions de Mobile Money 🦾')

tab1, tab2 = st.tabs(["API", "Dashboard"])
df_exists = False
df2_exists = False

with tab1:
  uploaded_file = st.file_uploader("Choose a file")
  if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df_exists = True
        st.data_editor(df, use_container_width=True)
        st.toast("Fichier uploader avec succès", icon='✅')
        output_buffer = io.StringIO()
        df.to_csv(output_buffer, index=False)
        output_buffer.seek(0)
        files = {'file': ("upload.csv", output_buffer)}
        endpoint = 'http://localhost:8000/api/anomaly-detection/file'
        response = requests.post(endpoint, files=files)
        print(response)
        if response.status_code == 200:
          content_io = io.BytesIO(response.content)
          st.success("Affichage du résultat en cours", icon='🦾')
          df2 = pd.read_csv(content_io)
          df2_exists = True
          st.data_editor(df2, use_container_width=True)
    except Exception as e:
      st.toast("N'uploadez que des fichiers csv", icon='❌')

with tab2:
  if df2_exists or df_exists:
    if df2_exists:
      dataset = df2
    elif df_exists:
      dataset = df
    dataset['Timestamp'] = pd.to_datetime(dataset['Timestamp'])
    dataset['Type'] = dataset['Type'].astype(str)
    dataset.dropna(inplace=True)
    
    type_counts = dataset['Type'].value_counts(normalize=True) * 100
    type_counts = type_counts.reset_index()
    type_counts.columns = ['Type', 'Percentage']
    
    cols = st.columns(len(type_counts))
    direction = ["30%", "70%", "20%", "10%"]
    for i, row in type_counts.iterrows():
        cols[i].metric(row['Type'], f"{row['Percentage']:.2f} %", direction[i])
        
    st.text("Exemples de fraudes sur les données")
    st.image('fraud.png')
    
    # Ajouter une colonne date sans l'heure pour grouper par jour
    dataset['Date'] = dataset['Timestamp'].dt.date
    # Créer un pivot table pour compter les occurrences de chaque type de transaction par date
    pivot_data = dataset.pivot_table(index='Date', columns='Type', aggfunc='size', fill_value=0)
    # Réinitialiser l'index pour convertir 'Date' de l'index à une colonne normale
    pivot_data = pivot_data.reset_index()
    # Line plot avec Plotly Express
    fig_line = px.line(pivot_data, x='Date', y=pivot_data.columns[1:], title='Distribution des types de Transaction par Date')
    fig_line.update_layout(xaxis_title='Date', yaxis_title='Count')
    st.plotly_chart(fig_line, use_container_width=True, theme='streamlit')
    # Montant des transactions
    montant_sum = dataset.groupby('Type')['Montant'].sum().reset_index()
    fig_bar = px.bar(montant_sum, x='Type', y='Montant', title='Montant total des transactions par type')
    st.plotly_chart(fig_bar, use_container_width=True)
    if df2_exists:
      outliers_counts = dataset['Outliers'].value_counts()
      labels = ['Suspectes', 'Normal']
      fig_pie = px.pie(names=labels, values=outliers_counts, title='Distribution des transactions suspectes')
      st.plotly_chart(fig_pie, use_container_width=True)

      filtered_dataset = dataset[dataset['Outliers'] == -1]
      montant_sum = filtered_dataset.groupby('Type')['Montant'].sum().reset_index()
      fig_bar = px.bar(montant_sum, x='Type', y='Montant', title='Montant total des transactions suspectes par type')
      st.plotly_chart(fig_bar, use_container_width=True)
  else:
    st.text("Exemples de fraudes sur les données")
    st.image('fraud.png')