import streamlit as st
import numpy as np
import spacy
import string
import re
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
import pickle

nlp = spacy.load('en_core_web_lg')

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)
    
with open('label_encoder.pkl', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Load the model and tokenizer
model = load_model("final_lstm_model.h5")

def word_cleaner(text):
    
    text = text.strip()
    
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    text = re.sub(r'[^\x00-\x7F]+', '', text)

    doc = nlp(text)

    lemmatized_words = [token.lemma_ for token in doc]
    
    additional_stopwords = set(['hotel', 'resort', 'day', 'use', 'need', 'think', 'night', 'say', 'look', 'beach', 'stay', 'time', 'people', 'place', 'area', 'room', 'come', 'staff', 'tell'])

    stop_words = set(stopwords.words('english')).union(set(spacy.lang.en.stop_words.STOP_WORDS)).union(additional_stopwords) - {'not', 'very'}
    
    lemmatized_words = [word for word in lemmatized_words if word.lower() not in stop_words]

    cleaned_text = ' '.join(lemmatized_words)

    return cleaned_text


# Define max sequence length
max_sequence_length = 100

# Function to preprocess text and make predictions
def predict_sentiment(text):
    # Tokenize and pad the text data
    sequence = tokenizer.texts_to_sequences([text])
    sequence_padded = pad_sequences(sequence, maxlen=max_sequence_length)
    
    # Make predictions
    predicted_probabilities = model.predict(sequence_padded)
    
    # Convert probabilities to class labels
    predicted_labels = np.argmax(predicted_probabilities, axis=1)
    
    predicted_classes = label_encoder.inverse_transform(predicted_labels)
    
    return predicted_classes[0], predicted_probabilities[0]

# Streamlit app
st.title("Hotel Review Classification v2")

# Input text box for user input
text_input = st.text_input("Enter your feedback: ", "")

# Predict button
if st.button("Predict"):
    # Predict sentiment
    predicted_class, predicted_probabilities = predict_sentiment(word_cleaner(text_input))
    
    # Display results
    st.write(f"You have given a {predicted_class} review! Thanks")
    
    if predicted_class == 'Positive':
        st.write(f"Probability of review being {predicted_class} is {predicted_probabilities[1]*100}")
    else:
        st.write(f"Review being {predicted_class} is {predicted_probabilities[0]*100} %")