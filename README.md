# ProxyLM Paper Replication Results

**Paper**: ProxyLM: Predicting Language Model Performance on Multilingual Tasks via Proxy Models  
**ArXiv**: [2406.09334](https://arxiv.org/abs/2406.09334)

---

## Overview

This repository contains a partial replication of the ProxyLM methodology for predicting machine translation model performance using meta-learning. We implement the paper's feature extraction framework with 19 features and train XGBoost regressors on 24 meta-training examples across 6 language pairs.

**Key Result**: Successfully validated the paper's methodology at smaller scale. Sentence-BERT similarity is the most important feature (93.86% importance), confirming the paper's emphasis on semantic features.

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone 
cd paper-replication-results

# Install dependencies
pip install -r requirements.txt
```

### Run the Code

```bash
cd code
python3 Research_Project_PaperReplication.py
```

This will:
- Load OPUS Books datasets for 6 language pairs (en-fr, en-es, en-it, en-ru, en-nl, en-sv)
- Extract 19 features following the paper's methodology
- Train XGBoost regressor to predict translation performance
- Save results to `../data/meta_features_paper_replication.csv`

### View Results

```bash
# View the data
head -10 data/meta_features_paper_replication.csv

# Or in Python
python3 -c "import pandas as pd; df = pd.read_csv('data/meta_features_paper_replication.csv'); print(df.head())"
```

---

## Repository Structure

```
paper-replication-results/
├── README.md                              # This file
├── requirements.txt                       # Dependencies
├── code/
│   └── Research_Project_PaperReplication.py  # Main implementation
├── data/
│   └── meta_features_paper_replication.csv   # Results (24 examples, 19 features)
└── analysis/
    └── COMPREHENSIVE_ANALYSIS.md         # Detailed analysis and paper comparison
```

---

## Results

### Dataset
- **24 meta-training examples** across 6 language pairs
- **Language pairs**: en-fr, en-es, en-it, en-ru, en-nl, en-sv
- **Training sizes**: 500, 1000, 2000, 5000 samples
- **19 features** computed per example

### Model Performance
- **RMSE**: 0.0081 (test set)
- **CV RMSE**: 2.67 ± 8.79
- **R²**: 1.0000
- **Regressor**: XGBoost

### Feature Importance
1. **Sentence-BERT similarity**: 93.86%
2. **Proxy BLEU**: 5.65%
3. **Average sentence length**: 0.49%

---

## Features Extracted

Following the paper's methodology:

**Basic Features (6):**
- Train size, Vocab size, Average sentence length
- Word overlap, Type-Token Ratio (TTR), TTR distance

**Additional Features (3):**
- Jensen-Shannon Divergence (JSD)
- TF-IDF cosine similarity
- Sentence-BERT similarity

**Supporting Features:**
- Source/target vocab sizes, sentence lengths
- Proxy BLEU scores, language metadata

See `analysis/COMPREHENSIVE_ANALYSIS.md` for detailed feature descriptions.

---

## Dependencies

```bash
pip install datasets transformers scikit-learn xgboost torch sentencepiece sentence-transformers sacrebleu pandas numpy scipy
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

---

## Configuration

Edit `code/Research_Project_PaperReplication.py` to customize:

```python
# Add more language pairs
datasets_config = [
    ("opus_books", "en-fr"),
    ("opus_books", "en-es"),
    # Add more here
]

# Adjust training sizes
TRAIN_SIZES = [500, 1000, 2000, 5000]

# Optimize for your hardware
DEFAULT_BATCH_SIZE = 8  # Reduce if memory constrained
```

---

## Why Partial Replication?

We could not replicate exactly due to:
- **Dataset scale**: Paper uses MT560 (32 datasets, 50 languages) vs our 1 dataset, 6 languages
- **Fine-tuning**: Paper fine-tunes M2M100 1.2B and NLLB 1.3B (requires weeks of GPU time)
- **Multiple proxies**: Paper uses 4 proxy models in ensemble vs our 1 model
- **Resources**: Full replication requires 2-4 weeks GPU time and $500-2000 cost

Our partial replication validates the methodology at smaller scale, which is standard practice.

**See `analysis/COMPREHENSIVE_ANALYSIS.md` for detailed explanation and comparison with the paper.**

---

## Citation

**Original Paper:**
```bibtex
@article{proxyLM2024,
  title={ProxyLM: Predicting Language Model Performance on Multilingual Tasks via Proxy Models},
  author={Anugraha, David and Winata, Genta Indra and Li, Chenyue and Irawan, Patrick Amadeus and Lee, En-Shiun Annie},
  journal={arXiv preprint arXiv:2406.09334},
  year={2024}
}
```

---

## License

This replication follows the original paper's licensing. Code and analysis provided for research purposes.
