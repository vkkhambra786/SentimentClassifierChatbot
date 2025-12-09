# SentimentClassifierChatbot

A sentiment analysis chatbot that fine-tunes a DistilBERT model on the IMDB dataset to classify text as positive or negative sentiment.

## Features

- Fine-tunes DistilBERT on IMDB movie reviews dataset
- Binary sentiment classification (positive/negative)
- Streamlit web interface for real-time sentiment analysis
- Rule-based chatbot responses
- Model evaluation and metrics

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script to fine-tune DistilBERT on the IMDB dataset:

```bash
python train_sentiment.py
```

For quick testing with smaller dataset (recommended for first run):
- Open `train_sentiment.py`
- Change `USE_SMALL_SUBSET = False` to `USE_SMALL_SUBSET = True`
- Run the training script

Training takes about 30-60 minutes on a GPU, or several hours on CPU.

### Running the Chatbot

After training, launch the Streamlit app:

```bash
streamlit run app_streamlit.py
```

### Evaluating the Model

Test the trained model on the test set:

```bash
python eval_sentiment.py
```

## Project Structure

- `train_sentiment.py` - Model training script
- `eval_sentiment.py` - Model evaluation script
- `app_streamlit.py` - Streamlit chatbot interface
- `requirements.txt` - Python dependencies
- `sentiment_model/` - Trained model directory (created after training)

## Model Details

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Dataset**: IMDB movie reviews
- **Task**: Binary sentiment classification
- **Training**: 3 epochs, batch size 16, learning rate 2e-5

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- Streamlit
- Scikit-learn