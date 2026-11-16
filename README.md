🎧 EmoSense AI – Speech Emotion Recognition

📌 Project Overview

EmoSense AI is a machine learning-based Speech Emotion Recognition (SER) application that analyses audio recordings and predicts the emotion expressed in speech.

The application uses audio signal processing and a Support Vector Machine (SVM) classifier to identify emotions from voice recordings.

🎯 Project Objective

The main objective is to develop an intelligent application that recognises emotional patterns in speech and presents the prediction through an interactive dashboard.

😊 Emotion Categories

The model supports eight emotion categories:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

🛠️ Technologies Used

- Python
- Google Colab
- Librosa
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib

⚙️ Methodology

1. Audio data collection using the RAVDESS Speech dataset.
2. Audio preprocessing and feature extraction.
3. Extraction of MFCC, pitch, energy, spectral centroid and zero-crossing rate features.
4. Training an SVM classifier.
5. Evaluating the model using a separate test set.
6. Integrating the trained model into a Streamlit application.

📊 Model Performance

- Model: Support Vector Machine (SVM)
- Extracted features: 128
- Test accuracy: 40.42%
- Test macro F1-score: 0.394

The model supports eight emotion classes. Performance varies across emotion categories, and further improvements are needed for reliable predictions on natural speech.

💻 Application Features

- Upload audio recordings.
- Analyse speech recordings.
- Predict the most likely emotion.
- Display the top three emotion predictions.
- Visualise emotion scores.
- Display audio waveform and MFCC features.

🚀 How to Run the Project

1. Clone the repository

git clone YOUR_GITHUB_REPOSITORY_URL

2. Open the project folder

cd EmoSense-AI

3. Install dependencies

pip install -r requirements.txt

4. Run the application

python -m streamlit run app.py

📁 Project Structure

EmoSense-AI/
│
├── app.py
├── emotion_svm_model_v2.pkl
├── emotion_labels.json
├── requirements.txt
└── README.md

🔮 Future Improvements

- Improve emotion classification accuracy.
- Explore advanced deep learning models.
- Improve recognition of natural speech.
- Expand testing with diverse speakers and recording conditions.

⚠️ Limitations

Speech emotion predictions are estimates based on acoustic patterns. They do not establish a person's actual emotional state. Performance may vary with recording quality, speaking style and speaker differences.

🙏 Acknowledgement

Dataset: Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).

Project developed as part of the CodeAlpha Internship – Speech Emotion Recognition Task.

---

Developed with Python and Machine Learning.
