# Genre Classification using KNN

This project focuses on classifying audio files into different genres using the K-Nearest Neighbors (KNN) algorithm.

## Introduction

- This project's aim was to utilize KNN to predict the genre of an audio input based on its features extracted using Mel Frequency Cepstral Coefficients (MFCC). It also incorporates a Streamlit user interface for easy interaction.

## Dataset

We are utilizing the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) available on Kaggle for training and testing our genre classification model.

## Code execution

```python
git clone https://github.com/jubinjacob03/genre-classification-recommendation_Spotify.git
```

```python
cd your-repository
```
---
> [!IMPORTANT]
> Download the GTZAN dataset and extract the <strong><em>Data</em></strong> folder in root directory, i.e. root directory should contain <strong><em>Data</em></strong> folder.
---

```python
streamlit run app.py
```
---
> [!WARNING]
> This model is trained to work on only <strong><em>plain</em></strong> audio files as an input. When using <strong><em>vocals</em></strong>, the model might fail
