
import os
import json
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from modules import ExploratoryDataAnalysis,ModelEvaluation

#%% PATHS
MODEL_PATH = os.path.join(os.getcwd(), 'saved_models', 'model.h5')
OHE_PATH = os.path.join(os.getcwd(), 'saved_models','ohe.pkl')
TOKEN_PATH = os.path.join(os.getcwd(),'saved_models','token.json')

#%% STEP 1: Model Loading

lstm_model = load_model(MODEL_PATH)
lstm_model.summary()
ohe_scaler = pickle.load(open(OHE_PATH,'rb'))
token = json.load(open(TOKEN_PATH,'r'))

#%%
# STEP 2: Impute New Data          
new_review = [input('Review about the movie\n')]

#%% STEP 3: Data Cleaning
eda = ExploratoryDataAnalysis()
new_review = eda.remove_tags(new_review) 
new_review = eda.lower_split(new_review) 

#%% STEP 4:  Data vectorization
tokenizer = tokenizer_from_json(token)
new_review = tokenizer.texts_to_sequences(new_review)
new_review = pad_sequences(new_review)

#%% STEP 5: Model Prediction
me = ModelEvaluation()
me.predict_input(lstm_model,new_review,ohe_scaler)
