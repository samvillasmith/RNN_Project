# 🎬 IMDB Sentiment Analyzer

A deep learning-powered sentiment analysis application that predicts whether movie reviews are positive or negative using a Simple RNN (Recurrent Neural Network) model trained on the IMDB dataset.

## ✨ Features

- **Real-time Sentiment Analysis**: Analyze movie reviews instantly with AI
- **Confidence Scoring**: Get detailed confidence metrics for predictions
- **Interactive UI**: Beautiful, user-friendly Streamlit interface with emojis and formatting
- **Visual Feedback**: Progress bars and color-coded results
- **Detailed Analytics**: Word count, confidence breakdown, and model certainty metrics

## 🚀 Demo

The application provides:
- 😊 Positive sentiment detection
- 😞 Negative sentiment detection
- 📊 Confidence score visualization
- 📈 Detailed analysis breakdown

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd RNN
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv rnn_env
source rnn_env/bin/activate  # On Windows: rnn_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download the pre-trained model**

The model file `simple_rnn_imdb.h5` should be in the root directory. If not available, train the model using the provided notebook.

## 📖 Usage

### Running the Web Application

```bash
streamlit run main.py
```

The application will open in your default browser at `http://localhost:8501`

### Training the Model

To train the model from scratch, open and run `simplernn.ipynb`:

```bash
jupyter notebook simplernn.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Model architecture definition
- Training with early stopping
- Model evaluation
- Saving the trained model

## 📁 Project Structure

```
RNN/
├── main.py                 # Streamlit web application
├── simplernn.ipynb        # Model training notebook
├── embedding.ipynb        # Word embedding examples
├── simple_rnn_imdb.h5    # Trained model file
├── requirements.txt       # Python dependencies
├── .gitignore            # Git ignore file
└── README.md             # Project documentation
```

## 🧠 Model Architecture

### Simple RNN Model

```
Model: "sequential"
_________________________________________________________________
Layer (type)                Output Shape              Param #   
=================================================================
embedding (Embedding)       (32, 500, 128)            1,280,000
simple_rnn (SimpleRNN)      (32, 128)                 32,896
dense (Dense)               (32, 1)                   129
=================================================================
Total params: 1,313,025 (5.01 MB)
Trainable params: 1,313,025 (5.01 MB)
```

### Model Details

- **Vocabulary Size**: 10,000 words
- **Embedding Dimension**: 128
- **Sequence Length**: 500 tokens
- **RNN Units**: 128
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Training Dataset**: IMDB Movie Reviews (25,000 training samples)

### Performance

- **Training Accuracy**: ~94%
- **Validation Accuracy**: ~84%
- Early stopping implemented to prevent overfitting

## 🎯 How It Works

1. **Input Processing**:
   - User enters a movie review
   - Text is converted to lowercase
   - Words are tokenized and encoded using IMDB word index
   - Sequence is padded to 500 tokens

2. **Prediction**:
   - Processed sequence passed through embedding layer
   - Simple RNN processes the sequence
   - Dense layer outputs probability (0-1)
   - Score > 0.5 = Positive, Score ≤ 0.5 = Negative

3. **Results Display**:
   - Sentiment classification (Positive/Negative)
   - Confidence percentage
   - Visual progress bar
   - Model certainty level (High/Moderate/Low)

## 🔧 Dependencies

```
numpy
tensorflow
streamlit
```

## 💡 Example Reviews

**Positive**:
```
"This movie was absolutely incredible with stunning visuals, brilliant acting, 
and a captivating storyline that kept me engaged from start to finish."
```

**Negative**:
```
"The film was a complete disappointment with terrible acting, a boring plot, 
and poor direction that made it feel like a waste of time."
```

**Neutral**:
```
"The movie had decent performances and an average plot that was neither 
particularly exciting nor disappointing, just a standard film experience."
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- IMDB Dataset from Keras
- TensorFlow/Keras team for the deep learning framework
- Streamlit for the web application framework

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Made using TensorFlow & Streamlit**