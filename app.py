# import streamlit as st
# import pandas as pd
# import joblib
# import re

# # Load model and vectorizer
# model = joblib.load('sentiment_model.pkl')
# vectorizer = joblib.load('tfidf_vectorizer.pkl')

# # Optional: map numeric predictions to label names manually
# label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}  # adjust based on your training labels

# def clean_text(text):
#     text = str(text).lower()
#     text = re.sub(r'[^\w\s]', '', text)   # remove punctuation
#     text = re.sub(r'\d+', '', text)       # remove numbers
#     text = re.sub(r'\s+', ' ', text).strip()  # remove extra spaces
#     return text

# def predict_sentiment(text):
#     cleaned = clean_text(text)
#     vec = vectorizer.transform([cleaned])
#     pred = model.predict(vec)[0]
#     return label_map.get(pred, "Unknown")   # convert numeric to readable label

# # Streamlit UI
# st.title("Sentiment Web Analyzer")
# background_image = 'image.jfif'   # optional
# st.image(background_image, use_column_width=True)

# st.header("Now Scale Your Thoughts")

# # Single text analysis
# with st.expander("Analyze Your Text"):
#     text = st.text_input("Text here:")
#     if text:
#         sentiment = predict_sentiment(text)
#         st.write(f"**Sentiment:** {sentiment}")

# # Excel file analysis
# with st.expander("Analyze Excel files"):
#     st.write("_**Note**_ : Your file must contain a column named 'text' with the text to analyze.")
#     upl = st.file_uploader('Upload file', type=['xlsx', 'xls'])
#     if upl:
#         df = pd.read_excel(upl)
#         if 'text' in df.columns:
#             df['Predicted_Sentiment'] = df['text'].apply(predict_sentiment)
#             st.write(df.head(10))

#             @st.cache_data
#             def convert_df(df):
#                 return df.to_csv(index=False).encode('utf-8')

#             csv = convert_df(df)
#             st.download_button(
#                 label="Download data as CSV",
#                 data=csv,
#                 file_name='sentiment_predictions.csv',
#                 mime='text/csv',
#             )
#         else:
#             st.error("File must contain a column named 'text'.")




import streamlit as st
import pandas as pd
import joblib
import re
import numpy as np

# Load model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# If your model is a sklearn classifier, you can get classes
try:
    original_classes = model.classes_   # e.g., [0,1,2] or ['neg','neu','pos']
    st.write(f"Model classes: {original_classes}")
except:
    st.write("Model doesn't have classes_ attribute")

# Label mapping - ADJUST BASED ON YOUR TRAINING
# Common mappings:
# Option A: numeric 0=Negative, 1=Neutral, 2=Positive
# Option B: numeric 0=Negative, 1=Positive (if binary)
# Option C: string labels like 'neg','neu','pos'
label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}   # CHANGE THIS if needed

# If your model outputs strings directly, use:
# label_map = {'neg':'Negative', 'neu':'Neutral', 'pos':'Positive'}

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)   # remove punctuation
    text = re.sub(r'\d+', '', text)       # remove numbers
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    
    # Get prediction and probability
    pred = model.predict(vec)[0]
    
    # For sklearn classifiers, get probabilities
    try:
        proba = model.predict_proba(vec)[0]
        confidence = max(proba)
        prob_dict = {model.classes_[i]: proba[i] for i in range(len(proba))}
    except:
        confidence = None
        prob_dict = None
    
    # Convert prediction to label
    if pred in label_map:
        sentiment = label_map[pred]
    else:
        # If pred is a string (like 'neg'), try direct mapping
        sentiment = label_map.get(pred, str(pred))
    
    # For debugging, return also confidence and raw prediction
    return sentiment, confidence, prob_dict, pred

# Streamlit UI
st.set_page_config(page_title="Sentiment Analyzer", page_icon="😊")
st.title("Sentiment Web Analyzer")

# Optional background image (handle missing file gracefully)
try:
    st.image('image.jfif', use_column_width=True)
except:
    st.info("Background image not found, but app works fine.")

st.header("Now Scale Your Thoughts")

# Sidebar for debugging options
with st.sidebar:
    st.subheader("Debugging Tools")
    show_debug = st.checkbox("Show raw prediction & confidence")
    st.markdown("---")
    st.write("**Test with known phrases:**")
    test_positive = st.button("Test: 'I love this movie!'")
    test_negative = st.button("Test: 'This is terrible.'")
    test_neutral = st.button("Test: 'The table is made of wood.'")

# Single text analysis
with st.expander("Analyze Your Text"):
    text = st.text_input("Enter your text here:")
    if text:
        sentiment, confidence, prob_dict, raw_pred = predict_sentiment(text)
        st.write(f"**Sentiment:** {sentiment}")
        if show_debug and confidence:
            st.write(f"Confidence: {confidence:.2f}")
            st.write(f"Raw prediction: {raw_pred}")
            if prob_dict:
                st.write("Probabilities:", prob_dict)

# Handle test buttons
if test_positive:
    sentiment, conf, prob, raw = predict_sentiment("I love this movie!")
    st.write(f"**'I love this movie!' → {sentiment}** (conf: {conf:.2f})")
if test_negative:
    sentiment, conf, prob, raw = predict_sentiment("This is terrible.")
    st.write(f"**'This is terrible.' → {sentiment}** (conf: {conf:.2f})")
if test_neutral:
    sentiment, conf, prob, raw = predict_sentiment("The table is made of wood.")
    st.write(f"**'The table is made of wood.' → {sentiment}** (conf: {conf:.2f})")

# Excel file analysis
with st.expander("Analyze Excel files"):
    st.write("_**Note**_ : Your file must contain a column named 'text' with the text to analyze.")
    upl = st.file_uploader('Upload file', type=['xlsx', 'xls'])
    if upl:
        df = pd.read_excel(upl)
        if 'text' in df.columns:
            # Apply prediction and get sentiment only (or include confidence)
            df['Predicted_Sentiment'] = df['text'].apply(lambda x: predict_sentiment(x)[0])
            st.write(df.head(10))

            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')

            csv = convert_df(df)
            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='sentiment_predictions.csv',
                mime='text/csv',
            )
        else:
            st.error("File must contain a column named 'text'.")