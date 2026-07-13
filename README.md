# Emotion Recognition from Speech 🎙️

Task 2 — CodeAlpha Machine Learning Internship

## 📌 Objective
Recognize human emotions — such as happy, angry, sad, calm, fearful, disgust, surprised, and neutral — directly from speech audio using deep learning and speech signal processing techniques.

## 🧠 Approach
Raw audio can't be fed into a neural network directly, so the pipeline first converts speech into meaningful numerical features, then learns patterns from those features:

1. **Preprocessing** — load `.wav` files and standardize sample rate
2. **Feature Extraction** — compute MFCCs (Mel-Frequency Cepstral Coefficients), which capture the timbral/spectral texture of speech the way the human ear perceives it
3. **Modeling** — a hybrid **CNN + LSTM** architecture:
   - Conv1D layers scan the MFCC sequence for local spectral patterns
   - LSTM layer models how those patterns evolve over time (tone, pacing, emphasis)
4. **Classification** — dense output layer predicts the emotion label

## 📂 Dataset
This project uses **[RAVDESS](https://zenodo.org/record/1188976)** (Ryerson Audio-Visual Database of Emotional Speech and Song).

- Download and extract the dataset
- Place all `Actor_01`, `Actor_02`, ... folders inside a `RAVDESS/` directory in the project root
- Emotion label is encoded in the filename (3rd field): `01`=neutral, `02`=calm, `03`=happy, `04`=sad, `05`=angry, `06`=fearful, `07`=disgust, `08`=surprised

The same pipeline also works with **TESS** or **EMO-DB** with minor changes to the label-parsing logic.

## 🛠️ Tech Stack
- **Python**
- **librosa** — audio loading & MFCC extraction
- **TensorFlow / Keras** — CNN + LSTM model
- **scikit-learn** — train/test split, label encoding, evaluation metrics
- **matplotlib / seaborn** — visualization

## 🚀 How to Run
1. Download the RAVDESS dataset and place it in `RAVDESS/` at the project root
2. Open `Task2_Emotion_Recognition_Speech.ipynb` in Jupyter
3. Run all cells top to bottom — features are extracted, the model trains, and results are plotted automatically

## 📊 Output
- MFCC visualization for a sample audio clip
- Training/validation accuracy and loss curves
- Classification report (precision, recall, F1-score per emotion)
- Confusion matrix heatmap
- Trained model saved as `speech_emotion_model.h5`

## 📁 Project Structure
```
├── RAVDESS/                              # dataset (not included, download separately)
├── Task2_Emotion_Recognition_Speech.ipynb
├── speech_emotion_model.h5               # generated after training
└── README.md
```

## ✍️ Author
**Areesha** — BSCS, University of Agriculture Faisalabad
CodeAlpha Machine Learning Internship, 2026
