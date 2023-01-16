# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 10:37:09 2023

@author: DELL
"""



# 1. Library imports
import uvicorn
from fastapi import FastAPI
from Depression import Depresion
import numpy as np
import pickle
import pandas as pd
import nltk
import re
import string
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer




# 2. Create the app object
app = FastAPI() 
classifier=pickle.load(open('NBclassifier.pkl','rb'))
vectorizer = pickle.load(open('CountVectorizer.pkl','rb'))


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# 4. Route with a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000
@app.get('/{name}')
def get_name(name: str):
    return {'Welcome To detecting depression API': f'{name}'}

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted text with the confidence
@app.post('/predict')
def predict_depression(data:Depresion):
    data = data.dict()
    text=data['text']
    text = str(text)
   
    
    #text.lower()
    #clean data
    def convert_to_lower(text):
        return text.lower()
    text = convert_to_lower(text)
    
    
    def remove_numbers(text):
        number_pattern = r'\d+'
        without_number = re.sub(pattern=number_pattern, repl=" ", string=text)
        return without_number
    text = remove_numbers(text)
    
    
    def remove_punctuation(text):
        return text.translate(str.maketrans('', '', string.punctuation))
    text = remove_punctuation(text)
    
    
    def remove_stopwords(text):
        removed = []
        stop_words = list(stopwords.words("english"))
        tokens = word_tokenize(text)
        for i in range(len(tokens)):
            if tokens[i] not in stop_words:
                removed.append(tokens[i])
        return " ".join(removed)
    text = remove_stopwords(text)
    
    
    def remove_extra_white_spaces(text):
        single_char_pattern = r'\s+[a-zA-Z]\s+'
        without_sc = re.sub(pattern=single_char_pattern, repl=" ", string=text)
        return without_sc
    text = remove_extra_white_spaces(text)
    
    
    def lemmatizing(text):
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text)
        for i in range(len(tokens)):
            lemma_word = lemmatizer.lemmatize(tokens[i])
            tokens[i] = lemma_word
        return " ".join(tokens)
    text = lemmatizing(text)
    
    '''
    text = [text]
    vectorizer = CountVectorizer()
    text = vectorizer.fit_transform(text)
    #text = text.toarray()
    
    '''
    
    
   
   # print(classifier.predict())
    prediction = classifier.predict(vectorizer.transform([text]))  #[text]
    if(prediction[0] == 1):
        prediction="This persson is depressed, he needs to see a therapist "
    else:
        prediction="This person is not depressed"
    return {
        'prediction': prediction
    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
  
# open anaconda terminal
#cd C:\Users\DELL\emotions_pred
#conda activate DMML
#uvicorn app:app --reload