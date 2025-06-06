

# 📘 Fine-Tuning BERT for Sentiment Analysis

This repository contains a Jupyter Notebook for fine-tuning a BERT-based model on the IMDB Movie Reviews dataset for binary sentiment classification (positive/negative). The project demonstrates end-to-end steps including data preprocessing, dataset conversion, model training, evaluation, and experiment tracking using Weights & Biases (W\&B).

---

## 🚀 Features

* Loads and processes the IMDB movie review dataset
* Converts data into Hugging Face `datasets` format
* Fine-tunes a pretrained BERT model for sentiment classification
* Uses `transformers` Trainer API for training and evaluation
* Tracks experiments with Weights & Biases (W\&B)
* Exports trained model for future use or deployment

---

## 🧠 Model & Libraries Used

* Model: `bert-base-uncased` (Hugging Face Transformers)
* Libraries:

  * `transformers`
  * `datasets`
  * `pandas`
  * `scikit-learn`
  * `torch`
  * `wandb`

---

## 🗃️ Dataset

* **Source**: [IMDB Movie Reviews Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
* **Size**: 50,000 reviews (25k train / 25k test)
* **Classes**: Binary (Positive = 1, Negative = 0)

---

## 📦 Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/bert-sentiment-analysis.git
   cd bert-sentiment-analysis
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Download IMDB Dataset**

   * Ensure `IMDB-Dataset.csv` is in the notebook directory.

4. **Run the notebook**

   * Launch Jupyter Notebook or Jupyter Lab.
   * Open and run the `Fine_Tuning_BERT_for_Sentiment_Analysis.ipynb`.

---

## 📊 Experiment Tracking

This project uses **Weights & Biases** (`wandb`) for tracking training metrics, losses, and model performance.

To enable W\&B:

```python
import wandb
wandb.login()
```

---

## 🏁 Results

* Achieved high accuracy on binary classification task
* The model generalizes well on unseen data
* W\&B logs include loss curves, confusion matrix, and sample predictions

---

## 📌 Future Work

* Hyperparameter tuning for better generalization
* Evaluation on other datasets (e.g., Amazon Reviews, Yelp)
* Exporting model to ONNX or deploying via FastAPI/Streamlit

---

## 🙌 Acknowledgments

* Hugging Face for the amazing `transformers` and `datasets` libraries
* [Stanford AI Lab](https://ai.stanford.edu/~amaas/data/sentiment/) for the IMDB dataset
* Weights & Biases for intuitive experiment tracking

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
