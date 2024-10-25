from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
import re
import nltk
from nltk.corpus import stopwords

# NLTK 불용어 다운로드
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
# 텍스트 정제 함수
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # 특수문자 제거
    text = re.sub(r'\s+', ' ', text)  # 다중 공백 제거
    text = text.lower()  # 소문자 변환
    text = ' '.join(word for word in text.split() if word not in stop_words)  # 불용어 제거
    return text
