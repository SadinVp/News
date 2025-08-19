# 📰 Fake News Detection Using NLP

A machine learning project that detects whether a news article is **Real** or **Fake** using **Natural Language Processing (NLP)** and **Deep Learning (LSTM)**.  
Achieved **90%+ accuracy** on real-world datasets.

---

## 🚀 Features
- Detects fake vs. real news using **TF-IDF + LSTM**  
- Preprocessing with **NLTK (stopwords, cleaning, tokenization)**  
- Deep learning model built with **TensorFlow/Keras**  
- Web interface built with **Flask + HTML/CSS**  
- Interactive text input & prediction display  

---

## 🛠️ Tech Stack
- **Python 3.8+**  
- **NLTK** (text preprocessing)  
- **Scikit-learn** (TF-IDF, evaluation)  
- **TensorFlow/Keras** (LSTM model)  
- **Flask** (backend web app)  
- **HTML/CSS** (frontend UI)  

---

## 📂 Project Structure
Fake-News-Detection/
│── dataset/                # Dataset (train.csv, test.csv)
│── models/                 # Trained model + tokenizer
│── static/                 # CSS files
│── templates/              # HTML frontend
│── preprocess.py           # Preprocessing functions
│── train.py                # Training script
│── app.py                  # Flask web app
│── requirements.txt        # Dependencies
│── README.md               # Project documentation

---

## 📊 Dataset
You can use the **Kaggle Fake News Dataset**:  
👉 https://www.kaggle.com/c/fake-news/data  

The dataset contains:  
- **Text**: News article text  
- **Label**: `1 = Fake`, `0 = Real`  

Place the dataset in the `dataset/` folder before training.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
git clone https://github.com/SadinVP/News
cd Fake-News-Detection

### 2️⃣ Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

### 3️⃣ Install dependencies
pip install -r requirements.txt

### 4️⃣ Train the model
python train.py

This will generate:
- models/fake_news_model.keras  
- models/tokenizer.pkl

### 5️⃣ Run Flask app
python app.py

Open in browser: 👉 http://127.0.0.1:5000

---

## 🎯 Usage
1. Enter any news article/text in the text box.  
2. Click **Check**.  
3. The system will display **Fake News** 🟥 or **Real News** 🟩.  

---

## 📸 Screenshots


---

## 📈 Model Performance
- **Model**: LSTM with Embedding layer  
- **Accuracy**: ~90%  
- **Metrics**: Precision, Recall, F1-score  

---

## 📌 Future Improvements
- Use **Bidirectional LSTMs / GRU** for better accuracy  
- Add **BERT/Transformer-based models**  
- Deploy on **Heroku / Render / AWS**  

---

## 👨‍💻 Author
- Mohammed Sadin VP
- 📧 Email:msadinvp@gmail.com
- 🌐 GitHub: https://github.com/SadinVp  

---

⚡ *This project is built for educational purposes to demonstrate NLP + Deep Learning applied to Fake News Detection.*  
