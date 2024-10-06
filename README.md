# Text Retrieval and Relevance Ranking

## Overview

This project implements various text retrieval and relevance ranking models for analyzing a large dataset of news articles. The assignment is split into two parts:

1. **Part I** focuses on parsing text data, building vocabulary, and ranking documents based on different models such as Bit Vector and TF-IDF.
2. **Part II** introduces more advanced techniques using Word2Vec for document relevance.

The project is implemented in two Python files:
- `textretrieval.py`: Contains implementations for Tasks 1-3 (Text Parsing, Bit Vector Model, TF-IDF Model).
- `Word2Vec-TFDF.py`: Contains implementations for Task 4 (Word2Vec) and Task 5 (extra credit).

---

## Dataset

The dataset used for this project is AG's News Topic Classification Dataset, specifically the `test.csv` file from [this repository](https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv). The dataset contains news articles, each with a title, description, and class. For this project, we focus on the "description" field.

---

## Requirements

The project is built using Python 3, and the following libraries are required:
- Pandas
- NumPy
- NLTK
- Gensim (for Word2Vec)

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

---

## Instructions

### Part I: Text Parsing and Vector Space Models (`textretrieval.py`)

1. **Task 1: Text Data Parsing and Vocabulary Selection (15 points)**
   - Cleans and preprocesses the dataset by removing stop-words, punctuation, numbers, HTML tags, and excess whitespaces.
   - Builds a vocabulary of the top 200 most frequent words from the dataset.

2. **Task 2: Document Relevance with Bit Vector Model (25 points)**
   - Implements a basic Vector Space Model (VSM) using a bit-vector representation.
   - Computes relevance scores for documents based on the query.
   
3. **Task 3: Document Relevance with TF-IDF Model (40 points)**
   - Implements the TF-IDF model using Okapi-BM25 (without document length normalization).
   - Ranks documents based on the relevance to the provided queries.

### Part II: Word2Vec Model (`Word2Vec-TFDF.py`)

4. **Task 4: Document Relevance with Word2Vec (20 points)**
   - Uses Word2Vec to compute word relevance based on pre-trained word embeddings.
   - Scores documents using the average log-likelihood of Word2Vec embeddings.
   
5. **Task 5 (Extra Credit): TF-IDF with Document Length Normalization (15 points)**
   - Extends the TF-IDF model from Task 3 by adding document length normalization.

---

## How to Run

To execute the text parsing and relevance models:

1. **Run `textretrieval.py`** for Tasks 1-3:
   ```bash
   python textretrieval.py
   ```

2. **Run `Word2Vec-TFDF.py`** for Tasks 4-5:
   ```bash
   python Word2Vec-TFDF.py
   ```

Each script will process the dataset and output the top 5 most relevant documents and the bottom 5 least relevant documents based on the provided queries.

---

## Queries Tested

The following queries are used to test the models:

- **Query 1**: "olympic gold athens"
- **Query 2**: "reuters stocks friday"
- **Query 3**: "investment market prices"

The results for each query are printed to the console.

---

## Output

For each model, the output includes:
- Top 5 most relevant documents.
- Bottom 5 least relevant documents.
- Relevance scores for each document.

---

## Contact

For any questions regarding the implementation, feel free to contact Letian Jiang.

---

This README provides an overview of your project, including its structure, implementation details, and how to run the scripts. Let me know if you need further customization!