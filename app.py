
import streamlit as st
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

@st.cache_resource
def treinar_modelo():
    df = pd.read_csv('https://raw.githubusercontent.com/clmentbisaillon/fake-and-real-news-dataset/master/fake_or_real_news.csv')
    df['label'] = df['label'].map({'FAKE': 1, 'REAL': 0})

    def limpar_texto(texto):
        texto = texto.lower()
        texto = re.sub(r'\W', ' ', texto)
        texto = re.sub(r'\s+', ' ', texto)
        return texto

    df['text'] = df['text'].apply(limpar_texto)
    X = df['text']
    y = df['label']

    vectorizer = TfidfVectorizer(max_df=0.7)
    X_vect = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    return modelo, vectorizer

modelo, vectorizer = treinar_modelo()

st.title("🧠 Detector de Fake News")
st.write("Insira o texto de uma notícia abaixo e descubra se ela parece **falsa** ou **real** com base em um modelo treinado.")

noticia_input = st.text_area("Digite ou cole a notícia aqui:")

if st.button("Analisar"):
    if noticia_input.strip() == "":
        st.warning("Por favor, insira o texto de uma notícia.")
    else:
        def limpar_texto(texto):
            texto = texto.lower()
            texto = re.sub(r'\W', ' ', texto)
            texto = re.sub(r'\s+', ' ', texto)
            return texto

        noticia_limpa = limpar_texto(noticia_input)
        noticia_vect = vectorizer.transform([noticia_limpa])
        resultado = modelo.predict(noticia_vect)[0]

        if resultado == 1:
            st.error("🚨 Esta notícia parece ser **FAKE NEWS**!")
        else:
            st.success("✅ Esta notícia parece ser **REAL**.")
