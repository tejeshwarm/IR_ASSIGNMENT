import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
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
test_features_str = " ".join(map(str, test_input_scaled.flatten()))

# TF-IDF Model Implementation
def tfidf_recommendation():
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["features"])
    test_tfidf = vectorizer.transform([test_features_str])
    similarity_scores = cosine_similarity(test_tfidf, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-5:][::-1]
    return df.iloc[top_indices]["label"].values

# Get Recommendations
print("TF-IDF Recommendations:", tfidf_recommendation())
