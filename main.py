# Importing the libraries
from nltk.corpus import stopwords
import nltk
import streamlit as st
from pycaret.classification import *
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Importing the mall dataset with pandas
df_news = pd.read_csv('./gold-dataset.csv')
print(df_news.info())
df_news = df_news[['News', 'PriceSentiment']].copy()
print(df_news.head())
df_news = df_news.dropna()
df_news['News'] = df_news['News'].str.replace('\W', '', regex=True)
print(df_news['PriceSentiment'].unique())


nltk.download('stopwords')
# stopwords for Indonesian
stop_words = stopwords.words('english')
print(stop_words)

tv = TfidfVectorizer(lowercase=True, stop_words=stop_words, token_pattern="[A-Za-z]+")
tf_idf = tv.fit_transform(df_news['News'])
tf_idf_df = pd.DataFrame(tf_idf.toarray(), columns=tv.get_feature_names())
# tf_idf_df['target_cat'] = df_news.reset_index().PriceSentiment.map({'none':0, 'neutral':1, 'positive':2, 'negative':-1})

from pycaret.classification import *
# setup = setup(data=tf_idf_df, target='target_cat', session_id=123, train_size = 0.7, fold=10, silent=True)

st.title('Sentiment Analysis of Commodity News (Gold)')

model_file_path = './trained_model.pickle'
# map_pickle = open(model_file_path, 'rb')
loaded_model = load_model(model_file_path)
print("ML model loaded ......")
# map_pickle.close()

with st.form('user_inputs'):
    news_title = st.text_input('News Title', key=1)
    st.form_submit_button()

news_title = re.sub('[^a-zA-Z0-9 \n\.]', '', news_title)

text_transformed = tv.transform([news_title])
text_transformed_df = pd.DataFrame(text_transformed.toarray(), columns=tv.get_feature_names())
prediction = predict_model(loaded_model, text_transformed_df)
prediction[['Label', 'Score']]

# st.write('The news sentiment prediction : {} '.format(prediction['Label']))
st.write("Sentiment Matrix => [ none':0, 'neutral':1, 'positive':2, 'negative':-1 ]")
print("prediction executed.")