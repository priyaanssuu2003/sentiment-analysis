# 🌟 Sentiment Analysis Studio

A production-grade Sentiment Analysis web application built with Python, Pandas, Scikit-learn, and Streamlit. This app classifies text into **Positive**, **Negative**, or **Neutral** sentiments, providing confidence scores and intuitive visualizations.

## Live Demo 
https://sentiment-analysis-cbpmuzj85m4jvlp8ypwpo7.streamlit.app/

## 🚀 Features
- **Single Text Analysis**: Instantly analyze sentences or paragraphs with an interactive UI.
- **Bulk CSV Upload**: Process hundreds of rows simultaneously and download the results.
- **Modern UI/UX**: Custom CSS styling, dark theme, interactive sentiment cards, and animations.
- **Machine Learning**: Powered by Scikit-learn (TF-IDF Vectorizer + Logistic Regression).
- **Interactive Visualizations**: Plotly charts for bulk data sentiment distribution.

## 🛠️ Tech Stack
- **Python 3.8+**
- **Streamlit** - Frontend framework
- **Scikit-learn** - Machine learning modeling
- **Pandas** - Data manipulation
- **Plotly** - Data visualization

## 📦 Installation & Setup

1. **Clone the repository** (or download the files)
   ```bash
   git clone <your-repo-url>
   cd Sentiment-Analysis
   ```

2. **Create a virtual environment** (Optional but recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the Model**
   Before running the app, you need to train and save the machine learning model.
   ```bash
   python train_model.py
   ```
   *Note: The current script uses a small synthetic dataset for demonstration. You can easily replace it with a real dataset like IMDB reviews or Twitter sentiment data.*

5. **Run the Streamlit App**
   ```bash
   streamlit run app.py
   ```

## 🧠 What I Learned
Through building this portfolio project, I successfully demonstrated the end-to-end data science lifecycle:
- **Data In**: Handling raw text inputs and bulk CSV data ingestion.
- **Model**: Building a text preprocessing pipeline and training an NLP model with Scikit-learn, then serializing it for production use.
- **Interface Out**: Designing a modern, responsive web application that presents AI predictions in a highly visual, user-friendly format.

## 📸 Screenshots
<img width="1920" height="1080" alt="Screenshot 2026-08-28 175251" src="https://github.com/user-attachments/assets/0ebe5dcb-f572-4138-a24b-df17bc92f33b" />
<img width="1920" height="1080" alt="Screenshot 2026-08-28 175339" src="https://github.com/user-attachments/assets/b3fa8162-7240-4512-a276-583b8e04282e" />
<img width="1920" height="1080" alt="Screenshot 2026-08-28 175408" src="https://github.com/user-attachments/assets/281c80df-4a0a-42d7-8404-5d580d181f2b" />
<img width="1920" height="1080" alt="Screenshot 2026-08-28 175456" src="https://github.com/user-attachments/assets/457b6fda-7060-4940-aa61-e076c573fe1d" />
<img width="1920" height="1080" alt="Screenshot 2026-08-28 175522" src="https://github.com/user-attachments/assets/f4ae6efb-7124-4de6-84fb-96e80bf0669a" />





