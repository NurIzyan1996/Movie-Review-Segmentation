
import os
import pandas as pd
import numpy as np
from modules import ExploratoryDataAnalysis, DataPreprocessing
from modules import ModelCreation,ModelEvaluation

#%% paths
LOG_PATH = os.path.join(os.getcwd(),'log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'saved_models', 'model.h5')
OHE_SAVE_PATH = os.path.join(os.getcwd(), 'saved_models','ohe.pkl')
TOKENIZER_JSON_PATH = os.path.join(os.getcwd(),'saved_models','token.json')
URL = 'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv'

#%%EDA
# STEP 1: Data Loading
df = pd.read_csv(URL)
review = df['review']
sentiment = df['sentiment']

#%%
# STEP 2: Data Cleaning
eda = ExploratoryDataAnalysis()
review = eda.remove_tags(review) # to remove tags
review = eda.lower_split(review) # to convert to lowercase & split

#%%
# STEP 3:  Data vectorization
review = eda.review_tokenize(review, TOKENIZER_JSON_PATH, prnt=True)
review = eda.review_pad_sequences(review)

#%%
#STEP : Preprocessing
# One Hot Encoder for LABEL = sentiment
data_pre = DataPreprocessing()
sentiment = data_pre.one_hot_encoder(sentiment, OHE_SAVE_PATH)

# to calculate the number of total categories
nb_categories = len(np.unique(sentiment))

#%%
#STEP 6: Model Creation
# split train & test data
mc = ModelCreation()
X_train, X_test, y_train, y_test = mc.split_data(review, sentiment)

num_words = 10000
model = mc.lstm_layer(num_words, nb_categories)

# train the model
mc.train_model(LOG_PATH,model,X_train,y_train,X_test,y_test,epochs=3)
#%%
# STEP 7: Model Evaluation
me = ModelEvaluation()
y_true, y_pred = me.predict_model(model,X_test,y_test,nb_categories)
me.report_metrics(y_true,y_pred)

#%%
# STEP 9: Model Deployment
model.save(MODEL_SAVE_PATH)
