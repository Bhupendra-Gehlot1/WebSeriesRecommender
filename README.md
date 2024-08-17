# TV Series Recommendation System

A machine learning-based recommendation system for TV series using content-based filtering.

## Project Overview

This project implements a TV series recommendation system using Python and scikit-learn. It processes a dataset of TV series, extracts relevant features, and uses a Nearest Neighbors algorithm to recommend similar shows based on content similarity.

## Features

- Data cleaning and preprocessing
- Text vectorization using CountVectorizer
- Nearest Neighbors algorithm for content-based recommendations
- Handling of missing data and text normalization

## Technologies Used

- Python
- pandas
- numpy
- scikit-learn
- re (Regular Expressions)

## Dataset

The project uses a TV series dataset (likely from Kaggle, based on the file structure), which includes information such as series names, descriptions, genres, and production details.

## How It Works

1. Data is loaded and cleaned, removing duplicates and unnecessary columns.
2. Text data is processed and normalized.
3. A CountVectorizer is used to convert text descriptions into numerical features.
4. A Nearest Neighbors model is trained on these features.
5. The system can then recommend similar TV series based on content similarity.

## Usage

To get recommendations for a TV series:

```python
series_name = 'Stranger Things'
distances, indices = model.kneighbors(X=train_df.loc[series_name,:].values.reshape(1,-1),
                                      n_neighbors=6)
