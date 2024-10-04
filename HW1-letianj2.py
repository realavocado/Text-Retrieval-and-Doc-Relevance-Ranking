# Problem 4
from gensim.models import KeyedVectors
import numpy as np
from scipy.special import expit  # Sigmoid function


class DocumentRelevanceWithWord2Vec:
    def __init__(self, model_path):
        # Load the pre-trained Word2Vec model
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    def compute_similarity(self, query_terms, document_terms):
        # Get embeddings for query terms and document terms if they exist in the model
        query_vecs = [self.model[word] for word in query_terms if word in self.model]
        doc_vecs = [self.model[word] for word in document_terms if word in self.model]

        # If none of the query or document terms are in the model, return 0 relevance
        if not query_vecs or not doc_vecs:
            return 0

        log_likelihood = 0
        for q_vec in query_vecs:
            for d_vec in doc_vecs:
                # Compute dot product and apply sigmoid to get probability
                similarity = expit(np.dot(q_vec, d_vec))
                # Accumulate log of probabilities
                log_likelihood += np.log(similarity)

        return log_likelihood

    def rank_documents(self, query, documents):
        # Preprocess query terms
        query_terms = query.lower().split()
        document_scores = []

        # Iterate over each document and compute relevance score with the query
        for doc in documents:
            document_terms = doc.lower().split()
            score = self.compute_similarity(query_terms, document_terms)
            document_scores.append((doc, score))

        # Sort documents by relevance score in descending order
        document_scores.sort(key=lambda x: x[1], reverse=True)
        return document_scores


# Example usage
model_path = 'GoogleNews-vectors-negative300.bin'  # Replace with the actual path to the model
relevance_model = DocumentRelevanceWithWord2Vec(model_path)

query = "olympic gold athens"
documents = [
    "athens is a historic city that hosted the olympic games",
    "the stock market fluctuated on friday due to tech stocks",
    "investors are watching the gold prices in the market"
]

ranked_docs = relevance_model.rank_documents(query, documents)
print("Top-ranked documents:")
for doc, score in ranked_docs[:5]:  # Show top 5 results
    print(f"Document: {doc} - Score: {score}")


############################################################################################################

# Problem 5
import nltk
import numpy as np
import math
from nltk.corpus import stopwords

class TextRetrieval:
    def __init__(self):
        nltk.download('stopwords')
        self.vocab = np.zeros(200)
        self.dataset = None
        self.IDF = np.zeros(200)
        self.stop_words = set(stopwords.words('english'))

    def compute_IDF(self, M, collection):
        for i, word in enumerate(self.vocab):
            doc_freq = sum(1 for doc in collection if word in doc.split())
            if doc_freq > 0:
                self.IDF[i] = math.log((M + 1) / doc_freq)
            else:
                self.IDF[i] = 0

    def calculate_avdl(self):
        total_length = sum(len(doc.split()) for doc in self.dataset[2])
        avdl = total_length / len(self.dataset)
        return avdl

    def text2TFIDF(self, text, applyBM25_and_IDF=False, b=0.65, k=3.5):
        tfidfVector = np.zeros(len(self.vocab))
        text_words = text.split()
        doc_length = len(text_words)  # Length of the current document
        avdl = self.calculate_avdl()  # Calculate average document length

        for i, word in enumerate(self.vocab):
            if word in text_words:
                term_freq = text_words.count(word)
                tfidfVector[i] = term_freq  # Basic term frequency
                if applyBM25_and_IDF:
                    # BM25 with document length normalization
                    normalizer = 1 - b + b * (doc_length / avdl)
                    tfidfVector[i] = self.IDF[i] * ((k + 1) * term_freq) / (term_freq + k * normalizer)

        return tfidfVector

# Testing code
if __name__ == "__main__":
    retrieval = TextRetrieval()

    # Sample dataset and vocabulary
    retrieval.dataset = [["title1", "category1", "olympic gold medal"],
                        ["title2", "category2", "stocks market prices rise"],
                        ["title3", "category3", "investment in gold market"],
                        ["title4", "category4", "market friday stocks"],
                        ["title5", "category5", "gold medal in athens"]]

    # Set a sample vocabulary and compute IDF for it
    retrieval.vocab = np.array(["olympic", "gold", "athens", "stocks", "market", "friday", "investment", "prices", "medal"])
    retrieval.compute_IDF(len(retrieval.dataset), [doc[2] for doc in retrieval.dataset])

    # Query test with document length normalization and BM25 enabled
    query = "olympic gold athens"
    print("TF-IDF with BM25 and length normalization for query:", query)
    for doc in retrieval.dataset:
        score = retrieval.text2TFIDF(doc[2], applyBM25_and_IDF=True)
        print(f"Document: {doc[2]}, Score: {np.dot(score, retrieval.text2TFIDF(query, applyBM25_and_IDF=True))}")
