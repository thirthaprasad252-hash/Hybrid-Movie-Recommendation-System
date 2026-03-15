# Hybrid-Movie-Recommendation-System
A production-grade movie recommender system that combines Content-Based 
and Collaborative Filtering techniques into a tunable hybrid model, 
built with Streamlit.

## Features
- Content-Based Filtering using TF-IDF on genre tags + Cosine Similarity
- Collaborative Filtering using Truncated SVD (Matrix Factorization)
- Bayesian Popularity Ranking for cold-start users with no history
- Hybrid Fusion with a live-tunable alpha blending parameter
- RMSE evaluation on an 80/20 train-test split
- EDA dashboard with genre distribution, rating stats, and top movies

## Tech Stack
Python · Streamlit · Pandas · NumPy · Scikit-learn

## Run Locally
pip install -r requirements.txt
streamlit run app.py
