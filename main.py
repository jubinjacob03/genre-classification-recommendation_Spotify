from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy as np
import os
import pickle
import random
import operator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
from collections import defaultdict
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os

client_id = os.environ.get('CLIENT_ID')
client_secret = os.environ.get('CLIENT_SECRET')
redirect_uri = os.environ.get('REDIRECT_URI')


# Streamlit UI
st.title("Audio Detection & Classification")


# Option 2: Select or Drop Audio
uploaded_file = st.file_uploader("Select or Drop Audio", type=["wav"])
if uploaded_file:
    st.audio(uploaded_file, format="audio/wav", start_time=0)
    
    # Option to predict genre for the new audio file
    if st.button("Predict Genre"):
        st.write("Engine is starting...")


    # Function to plot confusion matrix
        def plot_confusion_matrix(y_true, y_pred, classes):
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            st.pyplot(fig)


    # function to get the distance between feature vecotrs and find neighbors
        def getNeighbors(trainingSet, instance, k):
            distances = []
            for x in range(len(trainingSet)):
                dist = distance(trainingSet[x], instance, k) + distance(instance, trainingSet[x], k)
                distances.append((trainingSet[x][2], dist))
            distances.sort(key=operator.itemgetter(1))
            neighbors = []
            for x in range(k):
                neighbors.append(distances[x][0])
                
            return neighbors


    # identify the class of the instance
        def nearestClass(neighbors):
            classVote = {}
            for x in range(len(neighbors)):
                response = neighbors[x]
                if response in classVote:
                    classVote[response] += 1
                else:
                    classVote[response] = 1

            sorter = sorted(classVote.items(), key = operator.itemgetter(1), reverse=True)
            return sorter[0][0]
            

    # function to evaluate the model
        def getAccuracy(testSet, predictions):
            correct = 0
            for x in range(len(testSet)):
                if testSet[x][-1] == predictions[x]:
                    correct += 1
            return (1.0 * correct) / len(testSet)


    # directory that holds the wav files
        directory = "Data/genres_original/"
        results = defaultdict(int)
        i = 1

        for folder in os.listdir(directory):
            results[i] = folder
            i += 1
        # st.write(results)
    
    # binary file where we will collect all the features extracted using mfcc (Mel Frequency Cepstral Coefficients)
        f = open("mydataset.dat", 'wb')
        i = 0
        for folder in os.listdir(directory):
            i += 1
            if i == 11:
                break
            for file in os.listdir(directory+folder):        
                try:
                    (rate, sig) = wav.read(directory+folder+"/"+file)
                    mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
                    covariance = np.cov(np.matrix.transpose(mfcc_feat))
                    mean_matrix = mfcc_feat.mean(0)
                    feature = (mean_matrix, covariance, i)
                    pickle.dump(feature, f)
                except Exception as e:
                    print('Got an exception: ', e, ' in folder: ', folder, ' filename: ', file)        
        f.close()
    
    
    # Split the dataset into training and testing sets respectively
        dataset = []

    # Function to load dataset
        def load_dataset(filename, split, tr_set, te_set):
            with open(filename, 'rb') as f:
                while True:
                    try:
                        dataset.append(pickle.load(f))
                    except EOFError:
                        f.close()
                        break
            for x in range(len(dataset)):
                if random.random() < split:
                    tr_set.append(dataset[x])
                else:
                    te_set.append(dataset[x])
    
        trainingSet = []
        testSet = []
        load_dataset('my.dat', 0.66, trainingSet, testSet)
    
    
        def distance(instance1 , instance2 , k ):
            distance =0 
            mm1 = instance1[0] 
            cm1 = instance1[1]
            mm2 = instance2[0]
            cm2 = instance2[1]
            distance = np.trace(np.dot(np.linalg.inv(cm2), cm1)) 
            distance+=(np.dot(np.dot((mm2-mm1).transpose() , np.linalg.inv(cm2)) , mm2-mm1 )) 
            distance+= np.log(np.linalg.det(cm2)) - np.log(np.linalg.det(cm1))
            distance-= k
            return distance
    
    
    # making predictions using KNN
        leng = len(testSet)
        predictions = []
        for x in range(leng):
            predictions.append(nearestClass(getNeighbors(trainingSet, testSet[x], 5)))
        accuracy1 = getAccuracy(testSet, predictions)
        st.write(accuracy1)

    # Function to evaluate the model and show results
        def evaluate_model(test_set, predictions, classes, results):
            accuracy = getAccuracy(test_set, predictions)
            st.write(f"Accuracy: {accuracy}")
        
            y_true = [instance[-1] for instance in test_set]
            plot_confusion_matrix(y_true, predictions, classes)
            classification_rep = classification_report(y_true, predictions, target_names=classes)

        # Display each line of the classification report separately
            st.text("Classification Report:")
            for line in classification_rep.split('\n'):
                st.text(line)

        evaluate_model(testSet, predictions, classes=list(results.values()), results=results)


    # Function to plot audio waves
        def plot_audio_wave(signal, rate, title="Audio Wave"):
            fig, ax= plt.subplots(figsize=(10, 4))
            time = np.arange(0, len(signal)) / rate
            ax.plot(time, signal)
            ax.set_title(title)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            st.pyplot(fig)
            
            
        plot_audio_wave(sig, rate, title="Test Audio Wave")
        st.write("Genre for the uploaded file")

        (rate, sig) = wav.read(uploaded_file)
        mfcc_feat = mfcc(sig, rate, winlen=0.020, appendEnergy=False)
        covariance = np.cov(np.matrix.transpose(mfcc_feat))
        mean_matrix = mfcc_feat.mean(0)
        feature = (mean_matrix, covariance, i)



        pred = nearestClass(getNeighbors(dataset, feature, 5))
        st.write(results[pred])
        predicted_genre = results[pred]
        
        # Authenticate with Spotify API
        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

        # Recommend playlists based on the specified genre
        recommended_playlists = []
        playlists = sp.category_playlists(category_id=predicted_genre, country='US', limit=3)
        recommended_playlists.extend(playlists['playlists']['items'])
        

        # Display logo and text in the same row
        col1, col2 = st.columns([1, 3])  

        # Column 1: Logo
        spotify_logo_path = "logo.png"
        col1.image(spotify_logo_path, width=80)

        # Column 2: Text
        col2.subheader(f"Recommended Playlists for Genre : {predicted_genre}")


        for playlist in recommended_playlists:
            col1, col2 = st.columns(2)
            with col1:
                st.header(playlist['name'])
            with col2:
                st.subheader("Playlist Details :")
                st.write(f"Playlist ID : {playlist['id']}")
            playlist_url = f"https://open.spotify.com/playlist/{playlist['id']}"
            st.markdown(f"[Open Playlist on Spotify]({playlist_url})")
        st.markdown("---")

