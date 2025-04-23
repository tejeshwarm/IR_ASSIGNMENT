import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
df = pd.read_csv("Crop_recommendation.csv")

# Preprocessing
scaler = MinMaxScaler()
numerical_columns = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
label_encoder = LabelEncoder()
df["label_encoded"] = label_encoder.fit_transform(df["label"])
df["features"] = df[numerical_columns].astype(str).agg(" ".join, axis=1)

# Get User Input
N = float(input("Enter Nitrogen value: "))
P = float(input("Enter Phosphorus value: "))
K = float(input("Enter Potassium value: "))
temperature = float(input("Enter Temperature (°C): "))
humidity = float(input("Enter Humidity (%): "))
pH = float(input("Enter pH value: "))
rainfall = float(input("Enter Rainfall (mm): "))

# Normalize Input
test_input = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
test_input_scaled = scaler.transform(test_input)
test_features_str = test_input_scaled.flatten().astype(str).tolist()

# Word2Vec Model Implementation
def word2vec_recommendation():
    tokenized_corpus = [doc.split() for doc in df["features"]]
    model = Word2Vec(tokenized_corpus, vector_size=5, window=5, min_count=1, workers=4)
    
    input_vector = np.mean([model.wv[word] for word in test_features_str if word in model.wv], axis=0)
    document_vectors = np.array([np.mean([model.wv[word] for word in doc if word in model.wv], axis=0) for doc in tokenized_corpus])
    
    similarity_scores = cosine_similarity([input_vector], document_vectors).flatten()
    top_indices = similarity_scores.argsort()[-5:][::-1]
    return df.iloc[top_indices]["label"].values

# Get Recommendations
print("Word2Vec Recommendations:", word2vec_recommendation())
