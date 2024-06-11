# Genre Classification using KNN

This project focuses on classifying audio files into different genres using the K-Nearest Neighbors (KNN) algorithm.

## Introduction

This project's aim was to utilize KNN to predict the genre of an audio input based on its features extracted using <strong><em>Mel Frequency Cepstral Coefficients</em></strong> (MFCC). It also incorporates a <strong><em>Streamlit</em></strong> user interface for easy interaction. To enhance user experience, the project integrates with the Spotify API to recommend playlists based on the predicted genre. By leveraging Spotify's vast music library, users can discover curated playlists tailored to their genre preferences. The authentication process is streamlined using <strong><em>SpotifyClientCredentials</em></strong>, ensuring seamless access to Spotify's features.

## Dataset

- We are utilizing the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) available on Kaggle for training and testing our genre classification model.

## Code execution
  
```python
# cloning the repository
git clone https://github.com/jubinjacob03/genre-classification-recommendation_Spotify.git
```

```python
# navigating to the directory of cloned repository
cd your-repository
```
---
> [!IMPORTANT]
> Download the GTZAN dataset and extract the <strong><em>Data</em></strong> folder into root directory, i.e. root directory should contain <strong><em>Data</em></strong> folder.
---
```bash
# installing all the required packages
pip install python_speech_features scipy numpy scikit-learn matplotlib seaborn streamlit spotipy
```
> [!CAUTION]
> Before deploying the application, ensure to add your Spotify API credentials ( <strong><em>client ID and client secret</em></strong> ) in <code>line 18 & 19</code> of <strong>app.py</strong>. Failing to do so will result in authentication errors when accessing the Spotify API. Refer to the Spotify API documentation for instructions on obtaining and managing your credentials.
---

```python
streamlit run app.py
```
---
> [!WARNING]
> This model is trained to work on only <strong><em>plain</em></strong> audio files as an input. When using <strong><em>vocals</em></strong>, the model might fail


## Main Packages Used

- [Python Speech Features](https://python-speech-features.readthedocs.io/en/latest/): Library for extracting features from speech signals, including MFCC.
- [Scipy](https://docs.scipy.org/doc/scipy/): Scientific computing library for working with audio files and mathematical operations.
- [Numpy](https://numpy.org/doc/): Fundamental package for numerical computing in Python.
- [Scikit-learn](https://scikit-learn.org/stable/): Machine learning library for implementing KNN classifier and evaluation metrics.
- [Matplotlib](https://matplotlib.org/stable/contents.html): Visualization library for creating plots and charts.
- [Seaborn](https://seaborn.pydata.org/): Statistical data visualization library based on Matplotlib.
- [Streamlit](https://docs.streamlit.io/en/stable/): Interactive web application framework for building ML and data science projects.

## Spotify API Documentation

- [Spotify Developer Documentation](https://developer.spotify.com/documentation/): Official documentation for the Spotify API, providing comprehensive guides, reference, and examples for integrating Spotify features into applications.
- [Spotipy Documentation](https://spotipy.readthedocs.io/en/2.18.0/): Spotipy is a Python library for accessing the Spotify Web API. Its documentation offers guidance on authentication, querying Spotify's endpoints, and handling responses.

