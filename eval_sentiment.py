# eval_sentiment.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

MODEL_DIR = "sentiment_model"

def evaluate_pretrained_model(nlp_pipe):
    """Evaluate pre-trained model on IMDB test set"""
    ds = load_dataset("imdb", split="test")
    eval_size = 200
    texts = list(ds["text"])[:eval_size]
    labels = list(ds["label"])[:eval_size]

    preds = []
    for t in texts:
        out = nlp_pipe(t[:1000])[0]
        # Handle different label formats
        label = out["label"].lower()
        if "pos" in label or "label_1" in label:
            preds.append(1)
        elif "neg" in label or "label_0" in label:
            preds.append(0)
        else:
            preds.append(1 if out["score"] >= 0.5 else 0)

    print("Classification report (pre-trained model, subset):")
    print(classification_report(labels, preds, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))

def main():
    print("Loading model from", MODEL_DIR)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        print("Using locally trained model")
    except Exception as e:
        print(f"Local model not found ({e}). Using pre-trained sentiment model instead.")
        # Use pre-trained model for evaluation
        from transformers import pipeline
        nlp_pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")  # type: ignore
        device = 0 if torch.cuda.is_available() else -1
        nlp_pipe.device = device
        # Evaluate with pre-trained model
        evaluate_pretrained_model(nlp_pipe)
        return

    device = 0 if torch.cuda.is_available() else -1
    nlp_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device)  # type: ignore

    ds = load_dataset("imdb", split="test")
    # For quick eval, limit to first 200 items; change or remove for full test
    eval_size = 200
    texts = list(ds["text"])[:eval_size]
    labels = list(ds["label"])[:eval_size]

    preds = []
    for t in texts:
        out = nlp_pipe(t[:1000])[0]  # limit long inputs
        # HuggingFace returns labels like "LABEL_0" or "NEGATIVE"/"POSITIVE" depending on model; handle common cases
        lab = out["label"]
        if isinstance(lab, str):
            lab_lower = lab.lower()
            if "pos" in lab_lower:
                preds.append(1)
            elif "neg" in lab_lower:
                preds.append(0)
            elif lab_lower.startswith("label_"):
                # convert label_1 -> 1
                preds.append(int(lab_lower.split("_")[1]))
            else:
                preds.append(1 if out["score"] >= 0.5 else 0)
        else:
            preds.append(int(lab))

    print("Classification report (subset):")
    print(classification_report(labels, preds, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))

if __name__ == "__main__":
    main()
