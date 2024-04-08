import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
from datetime import datetime

# Load the data
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    data = pd.read_csv("Mall_Customers.csv")  # Default data if no file is uploaded

# Display the loaded data
st.write("Data Sample:")
st.write(data.head())

# Select features for clustering
selected_features = ["Annual Income (k$)", "Age", "Spending Score (1-100)"]
X = data[selected_features].values

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the KMeans model
num_clusters = 5  # Fixed number of clusters
kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
predicted_cluster = kmeans.fit_predict(X_scaled)

# Take input from the user for features
user_input = []
for feature in selected_features:
    value = st.number_input(f"Enter value for {feature}:", value=0.0)
    user_input.append(value)

# Show the "Calculate" button after taking input for all features
if len(user_input) == len(selected_features):
    if st.button('Calculate'):
        # Predict the cluster for the user data
        user_data = scaler.transform([user_input])
        predicted_cluster_user = kmeans.predict(user_data)

        # Display details of the cluster the user belongs to
        cluster_number = predicted_cluster_user[0]
        st.write(f"You belong to Cluster {cluster_number}")

        # Calculate mean, median, mode, etc. for the cluster
        cluster_data = data[predicted_cluster == cluster_number]
        cluster_stats = cluster_data.describe()
        st.write("Cluster Statistics:")
        st.write(cluster_stats)

        # Save user input and timestamp to JSON
        user_data_json = {
            "user_input": user_input,
            "predicted_cluster": int(cluster_number),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open('user_data.json', 'a') as file:
            file.write(json.dumps(user_data_json) + '\n')
else:
    st.warning("Please provide input for all selected features.")
