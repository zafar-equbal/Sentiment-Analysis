import streamlit as st;
import pandas as pd 

#for training and test splitting
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

#for mlops pipelines pyt
from sklearn.pipeline import make_pipeline

#joblib 
import joblib

#How to detect Encoding
import chardet
import time

# Model Name 
MODEL_NAME = './sentiment_model.pkl'


st.set_page_config(page_title='Sentiment Analysis', layout='centered')
st.title('Sentiment Analysis')  

#Manage the state 
if 'data_ready' not in st.session_state:
    st.session_state['data_ready'] = False
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'X' not in st.session_state:
    st.session_state['X'] = None
if 'y' not in st.session_state:
    st.session_state['y'] = None

#Ui for csv uploading 
uploaded_file = st.file_uploader('Upload Your CSV', type=['csv'])
if uploaded_file is not None:
    with uploaded_file as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        print(f' Encoding Type: {result}')
        encoding = result['encoding']
        st.success(f'Detected Encoding : {encoding}')
       
        #IO Methods 
        from io import StringIO, BytesIO

        try:
            decoded = raw_data.decode(encoding)
            df = pd.read_csv(StringIO(decoded))
            st.session_state['df'] = df
            st.success('File Uploaded Succesfully')
            st.dataframe(df.head())
        except Exception as e:
            st.error(f'Error while Encoding {e}')
            st.stop()

#Mapping of the columns :-
if st.session_state['df'] is not None:
    with st.form('column_form'):
        st.markdown('### Map the columns with Headers')
        text_column = st.selectbox('select the text column', st.session_state['df'].columns)
        label_column = st.selectbox('select the label column', st.session_state['df'].columns)
        submit_btn = st.form_submit_button('Continue')

    #Show Training Model 
    if submit_btn:
        st.session_state['X'] = st.session_state['df'][text_column]  #text 
        st.session_state['y'] = st.session_state['df'][label_column].astype('str')  #labels 
        st.session_state['data_ready'] = True

if st.session_state['data_ready']:
    if st.button('Train Model'):
        st.markdown('### Please Wait Training in Progress')

        #Progress Bar 
        progress = st.progress(0)
        for percent_complete in range(1, 101):
            time.sleep(0.005)
            progress.progress(percent_complete)

        #Training of the Model
        X = st.session_state['X']
        y = st.session_state['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = make_pipeline(CountVectorizer(), MultinomialNB())
        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_NAME)

        #Accuracy Score 
        score = model.score(X_test, y_test)
        st.info(f'Test Accuracy Score {score:.2f}')

        #Success 
        st.success('Model Trained and Saved Successfully')


# Show this only if model is trained and data is ready
if st.session_state.get("data_ready"):
    st.markdown("---")
    st.subheader(" Test Your Own Sentence")

    try:
        model = joblib.load(MODEL_NAME)
        user_input = st.text_area(" Enter a sentence to analyze:")

        if st.button("Predict Sentiment"):
            if user_input.strip() == "":
                st.warning(" Please enter some text.")
            else:
                result = model.predict([user_input])[0]

                if result == "positive":
                    st.success(" Sentiment: Positive")
                elif result == "negative":
                    st.error(" Sentiment: Negative")
                elif result == "neutral":
                    st.info(" Sentiment: Neutral")
                else:
                    st.write(f" Sentiment: {result}")

    except Exception as e:
        st.error(f"❌ Could not load the model: {e}")
