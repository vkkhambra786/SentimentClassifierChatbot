# train_sentiment.py
# Fine-tune DistilBERT on IMDB sentiment dataset (binary)
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "sentiment_model"
BATCH_SIZE = 16
EPOCHS = 1  # Reduced for faster CPU training
MAX_LENGTH = 128  # Reduced for faster processing on CPU
USE_SMALL_SUBSET = True  # set True for quick local runs (small subsets)

def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, padding=False, max_length=MAX_LENGTH)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

def main():
    print("Checking if model already exists...")
    if os.path.exists(OUTPUT_DIR) and os.listdir(OUTPUT_DIR):
        response = input(f"Model directory '{OUTPUT_DIR}' already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Training cancelled.")
            return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading IMDB dataset...")
    raw_datasets = load_dataset("imdb")
    # Optionally use smaller subset for quick tests
    if USE_SMALL_SUBSET:
        print("Using small subset for quick testing...")
        # Use type ignore to avoid type checker issues with HuggingFace datasets
        raw_datasets = {
            "train": raw_datasets["train"].shuffle(seed=42).select(range(2000)),  # type: ignore
            "test": raw_datasets["test"].shuffle(seed=42).select(range(1000))  # type: ignore
        }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizing dataset (this may take a few minutes)...")
    tokenized_train = raw_datasets["train"].map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=["text"], num_proc=4)  # type: ignore
    tokenized_test = raw_datasets["test"].map(lambda x: preprocess_function(x, tokenizer), batched=True, remove_columns=["text"], num_proc=4)  # type: ignore

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        push_to_hub=False,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=False,  # Keep False for CPU
        dataloader_num_workers=4,  # Add parallel data loading
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,  # type: ignore
        eval_dataset=tokenized_test,  # type: ignore
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Starting training...")
    trainer.train()
    print("Saving model and tokenizer to", OUTPUT_DIR)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Evaluate on test and print metrics
    print("Evaluating on test set...")
    preds_output = trainer.predict(tokenized_test)  # type: ignore
    logits = preds_output.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = logits.argmax(axis=-1)
    labels = preds_output.label_ids
    if labels is None:
        # If label_ids is None, get labels from the dataset
        labels = list(tokenized_test["label"])  # type: ignore
    # Ensure labels is a numpy array for sklearn compatibility
    labels = np.array(labels)
    preds = np.array(preds)
    from sklearn.metrics import classification_report, confusion_matrix
    print("Classification report (test):")
    print(classification_report(labels, preds, digits=4))
    print("Confusion matrix (test):")
    print(confusion_matrix(labels, preds))

if __name__ == "__main__":
    main()
