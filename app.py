import streamlit as st
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import json

# Chargement du modèle entraîné
model = load_model("modele/FOMENA_TSATSOP_VALDES_JOEL_CLASS.h5")

# Chargement de l'index des mots
with open("modele/word_index.json", "r") as f:
    word_index = json.load(f)

maxlen = 200

st.title("Classification d'articles Reuters")
st.write("Entrez un texte et obtenez la catégorie prédite par le modèle.")

text_input = st.text_area("Entrez votre article :")

if st.button("Prédire"):
    if text_input:
        words = text_input.lower().split()
        sequence = [word_index.get(word, 0) + 3 for word in words if word in word_index]
        sequence_padded = pad_sequences([sequence], maxlen=maxlen)

        prediction = model.predict(sequence_padded)
        category_pred = np.argmax(prediction)

        st.success(f"La catégorie prédite est : {category_pred}")
    else:
        st.warning("Veuillez entrer un texte avant de prédire.")