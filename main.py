import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# load the IMDB dataset
word_index = imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

#load the pre-trained model
model = load_model('simple_rnn_imdb.h5')

# Preprocessing functions
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Create a function to predict sentiment
def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)
    prediction = model.predict(preprocessed_input)
    sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
    return sentiment, prediction[0][0]

# Header
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.markdown("### ✨ Discover if your movie review is positive or negative using AI!")
st.markdown("---")

# Instructions
with st.expander("ℹ️ How to use this app"):
    st.write("""
    1. 📝 Type or paste your movie review in the text area below
    2. 🔮 Click the **Analyze Sentiment** button
    3. 🎯 Get instant results with confidence scores!
    """)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📄 Enter Your Review")
    user_input = st.text_area(
        "Type your movie review here:",
        height=200,
        placeholder="Example: The film delivers fantastic performances from its entire cast paired with a thrilling plot that keeps you on the edge of your seat, making it a great cinematic experience from start to finish."
    )
    
    st.markdown("")
    predict_button = st.button("🔮 Analyze Sentiment", type="primary", use_container_width=True)

with col2:
    st.markdown("### 💡 Quick Tips")
    st.info("""
    **Write a good review:**
    - 🎭 Mention acting quality
    - 📖 Describe the plot
    - 🎨 Comment on visuals
    - 🎵 Note the soundtrack
    """)

# Prediction section
st.markdown("---")

if predict_button:
    if user_input.strip():
        with st.spinner("🤖 Analyzing your review..."):
            preprocessed_input = preprocess_text(user_input)
            prediction = model.predict(preprocessed_input)
            sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
            confidence = float(prediction[0][0])  # ✅ Convert to Python float
            
        # Results
        st.markdown("### 🎯 Analysis Results")
        
        # Display metrics in columns
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            if sentiment == "Positive":
                st.metric(label="Sentiment", value="😊 Positive", delta="Good vibes!")
            else:
                st.metric(label="Sentiment", value="😞 Negative", delta="Not so great")
        
        with metric_col2:
            st.metric(label="Confidence Score", value=f"{confidence:.1%}")
        
        with metric_col3:
            word_count = len(user_input.split())
            st.metric(label="Word Count", value=f"{word_count} words")
        
        # Visual confidence bar
        st.markdown("#### 📊 Confidence Breakdown")
        
        if sentiment == "Positive":
            st.success(f"🎉 This review is **{confidence:.1%}** positive!")
            st.progress(confidence)  # ✅ Now works with Python float
        else:
            st.error(f"👎 This review is **{(1-confidence):.1%}** negative!")
            st.progress(1 - confidence)  # ✅ Now works with Python float
        
        # Additional insights
        st.markdown("---")
        with st.expander("📈 See detailed analysis"):
            st.write(f"**Raw Prediction Score:** {confidence:.4f}")
            st.write(f"**Threshold:** 0.5 (scores above are positive, below are negative)")
            st.write(f"**Model Certainty:** {'High' if abs(confidence - 0.5) > 0.3 else 'Moderate' if abs(confidence - 0.5) > 0.15 else 'Low'}")
    else:
        st.warning("⚠️ Please enter a movie review to analyze!")
else:
    st.info("👆 Enter your movie review above and click **Analyze Sentiment** to get started!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>🤖 Powered by TensorFlow & Streamlit | 🎬 IMDB Dataset</p>
    </div>
    """,
    unsafe_allow_html=True
)