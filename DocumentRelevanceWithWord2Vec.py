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
