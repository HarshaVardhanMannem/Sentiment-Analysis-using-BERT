

# 📘 Fine-Tuning BERT for Sentiment Analysis

This repository contains a Jupyter Notebook for fine-tuning a BERT-based model on the IMDB Movie Reviews dataset for binary sentiment classification (positive/negative). The project demonstrates end-to-end steps including data preprocessing, dataset conversion, model training, evaluation, experiment tracking using Weights & Biases (W&B), and model publishing to Hugging Face Hub.

---

## 🚀 Features

* Loads and processes the IMDB movie review dataset
* Converts data into Hugging Face `datasets` format
* Fine-tunes a pretrained BERT model (`bert-base-uncased`) for sentiment classification
* Uses `transformers` Trainer API with mixed-precision (fp16) for efficient training
* Tracks experiments with Weights & Biases (W&B)
* Saves and exports the trained model locally and to Hugging Face Hub
* Provides an inference pipeline for real-time predictions

---

## 🧠 Model & Libraries Used

* **Model**: `bert-base-uncased` (Hugging Face Transformers)
* **Libraries**:
  * `transformers`
  * `datasets`
  * `evaluate`
  * `pandas`
  * `torch`
  * `wandb`
  * `huggingface_hub`

---

## 🗃️ Dataset

* **Source**: [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
* **Total Size**: 50,000 reviews (perfectly balanced — 25,000 positive, 25,000 negative)
* **Train/Test Split**: 70% train (35,000 reviews) / 30% test (15,000 reviews)
* **Classes**: Binary — Positive (`1`) / Negative (`0`)

---

## 📓 Notebook Walkthrough

The notebook `Fine_Tuning_ BERT for Sentiment_Analysis.ipynb` follows these steps:

### 1. Setup & Login
* Logs into **Weights & Biases** (`wandb`) for experiment tracking.

### 2. Data Loading
* Loads `IMDB-Dataset.csv` using `pandas`.
* Confirms the dataset is balanced: 25,000 positive and 25,000 negative reviews.

### 3. Pretrained Model Sanity Check
* Loads `bert-base-uncased` and its tokenizer from Hugging Face.
* Runs a quick inference test on a sample sentence (`"This movie was fantastic"`) to verify the model loads correctly before fine-tuning.

### 4. Dataset Conversion & Label Encoding
* Converts the Pandas DataFrame into a Hugging Face `Dataset` object.
* Applies a 70/30 train/test split.
* Maps text labels to integers: `negative → 0`, `positive → 1`.

### 5. Tokenization
* Uses `BertTokenizerFast` (vocab size: 30,522) with the following settings:
  * `padding=True`, `truncation=True`
  * `max_length=128` tokens
* Tokenizes the entire dataset in batched mode for efficiency.
* Each sample gains three new fields: `input_ids`, `token_type_ids`, `attention_mask`.

### 6. Metrics Definition
* Uses the Hugging Face `evaluate` library to compute **accuracy** during evaluation.

### 7. Model Initialization
* Loads `bert-base-uncased` with a classification head (`AutoModelForSequenceClassification`, `num_labels=2`).

### 8. Training Configuration
* Configured via `TrainingArguments`:

| Parameter | Value |
|---|---|
| Output directory | `train_dir` |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Train batch size (per device) | 32 |
| Eval batch size (per device) | 64 |
| Evaluation strategy | Per epoch |
| Mixed precision (fp16) | ✅ Enabled |

### 9. Training
* Runs `trainer.train()` using the Hugging Face `Trainer` API.
* Experiment metrics and loss curves are tracked live in W&B.

### 10. Evaluation
* Runs `trainer.evaluate()` on the held-out test set.

### 11. Model Saving & Inference
* Saves the fine-tuned model locally to `bert-sentiment-analysis/`.
* Loads a `text-classification` pipeline and runs sample predictions.

### 12. Publishing to Hugging Face Hub
* Creates a public repository on Hugging Face Hub.
* Uploads the entire fine-tuned model to [`Harsha901/bert-sentiment-analysis-model`](https://huggingface.co/Harsha901/bert-sentiment-analysis-model).
* Demonstrates loading and running inference directly from the Hub.

---

## 📊 Training Results

| Metric | Value |
|---|---|
| Total training steps | 3,282 |
| Training loss | 0.2033 |
| Training runtime | ~1,227 seconds (~20.5 min) |
| Train samples / second | 85.55 |
| Train steps / second | 2.67 |
| Total FLOPs | 6.91 × 10¹⁵ |
| Epochs | 3 |

---

## 🏆 Evaluation Metrics (Test Set — 15,000 reviews)

| Metric | Value |
|---|---|
| **Accuracy** | **89.43%** |
| Evaluation loss | 0.3749 |
| Eval runtime | ~34.6 seconds |
| Eval samples / second | 433.85 |
| Eval steps / second | 6.80 |

---

## 🔍 Sample Inference Results

The fine-tuned model correctly classifies reviews with very high confidence:

| Input Text | Predicted Label | Confidence |
|---|---|---|
| "this movie was horrible, the plot was really boring. acting was okay" | **negative** | 99.93% |
| "the movie is really sucked. there is not plot and acting was bad" | **negative** | 99.93% |
| "what a beautiful movie. great plot. acting was good. will see it again" | **positive** | 99.75% |
| "This movie was absolutely amazing!" | **positive** | 99.58% |

---

## 🤗 Model on Hugging Face Hub

The trained model is publicly available at:
**[Harsha901/bert-sentiment-analysis-model](https://huggingface.co/Harsha901/bert-sentiment-analysis-model)**

You can use it directly:

```python
from transformers import pipeline

classifier = pipeline('text-classification', model='Harsha901/bert-sentiment-analysis-model')
result = classifier("This movie was absolutely amazing!")
print(result)
# [{'label': 'positive', 'score': 0.9957851767539978}]
```

---

## 📦 Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/HarshaVardhanMannem/Sentiment-Analysis-using-BERT.git
   cd Sentiment-Analysis-using-BERT
   ```

2. **Install dependencies**

   ```bash
   pip install transformers datasets evaluate pandas torch wandb huggingface_hub
   ```

3. **Download IMDB Dataset**

   * Ensure `IMDB-Dataset.csv` is in the notebook directory.

4. **Run the notebook**

   * Launch Jupyter Notebook or Jupyter Lab.
   * Open and run `Fine_Tuning_ BERT for Sentiment_Analysis.ipynb`.

---

## 📊 Experiment Tracking

This project uses **Weights & Biases** (`wandb`) for tracking training metrics, losses, and model performance.

To enable W&B:

```python
import wandb
wandb.login()
```

---

## 📌 Future Work

* Hyperparameter tuning (learning rate, batch size, epochs) for better generalization
* Evaluation on other datasets (e.g., Amazon Reviews, Yelp)
* Exporting model to ONNX or deploying via FastAPI / Streamlit

---

## 🙌 Acknowledgments

* Hugging Face for the `transformers`, `datasets`, and `evaluate` libraries
* [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/) for the IMDB dataset
* Weights & Biases for intuitive experiment tracking

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
