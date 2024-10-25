# pip install pandas sklearn datasets pickle seaborn matplotlib
# sklearn
# 파이썬에서 제공되는 머신러닝 라이브러리이다. 
# 다양한 머신러닝 알고리즘과 도구를 제공한다. 
# 머신러닝 모델을 만들고, 데이터를 전처리하며, 모델을 평가하는데 필요한 다양한 기능을 포함하고 있다.
import pandas as pd
from sklearn.model_selection import train_test_split # 데이터를 훈련과 테스트 세트로 나눈다.
from sklearn.feature_extraction.text import CountVectorizer # 텍스트 데이터를 벡터로 변환하는 도구
from sklearn.linear_model import LogisticRegression # 로지스틱 회귀 모델 구현
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # 모델 성능 평가를 위한 자료 제공 지표
from datasets import load_dataset # 객체를 직렬화 하고 저장하는데 사용되는 모듈
import pickle # 직렬화 객체를 저장하는데 사용되는 모듈 학습된 모델이나 데이터 구조를 파일로 저장하고 나중에 다시 로드하는 경우 사용
import os
import seaborn as sns # 데이터 시각화를 위한 라이브러리 Maptplotlib를 기반으로 데이터를 이쁘게 시각화
import matplotlib.pyplot as plt # 그래프를 그리는데 사용되는 라이브러리
import clean_text as ct

plt.rc('font', family='Malgun Gothic')  # Windows
# plt.rc('font', family='AppleGothic')  # MacOS


#--- 데이터 준비 단계 

# 데이터셋을 다운로드하고 로컬에 저장하는 함수
def download_and_save_dataset(local_path="./model/dataset/email_spam_dataset.csv"):
    folder_path = os.path.dirname(local_path) 
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # 폴더가 없으면 생성

    if not os.path.exists(local_path):
        # 데이터셋 다운로드
        dataset = load_dataset("NotShrirang/email-spam-filter", split='train')
        
        # Hugging Face 데이터셋을 pandas 데이터프레임으로 변환 후 저장
        df = pd.DataFrame(dataset)
        df.to_csv(local_path, index=False)
        print(f"데이터셋을 {local_path}에 저장했습니다.")
    else:
        print(f"로컬에 이미 데이터셋이 있습니다: {local_path}")

# 로컬에 데이터셋이 없으면 다운로드
download_and_save_dataset()


# 로컬에서 데이터셋 로드
def load_local_dataset(local_path="./model/dataset/email_spam_dataset.csv"):
    if os.path.exists(local_path):
        df = pd.read_csv(local_path)
        return df
    else:
        raise FileNotFoundError(f"{local_path} 파일이 존재하지 않습니다.")

# 로컬 데이터셋 로드
df = load_local_dataset()

# 데이터셋 확인
print(df.head())

# 데이터 준비
texts = df["text"] # 이메일 텍스트
labels = df["label_num"] # 스팸 여부


#--- 전처리 단계

# CountVectorizer
# 텍스트 데이터를 Bag of Words(단어 가방) 기법을 사용하여 벡터로 변환하는 도구이다.
# Bag of Words는 텍스트 문서에서 단어의 등장 횧수만을 고려하여 텍스트를 수치 데이터로 변환하는 기법이다.
# 문맥이나 순서는 무시하며, 각 단어의 빈도수만을 가지고 수치화한다.

# 변환 과정 
# 토큰화 (텍스트를 단어 단위로 분리) -> 단어 집합 (텍스트 전체에서 등장한 단어 목록을 만듬) -> 단어 빈도수 계산 -> 벡터 생성
 
# 벡터
# 벡터는 수학적으로 여러 숫자의 배열로, 머신러닝에서는 특성(feature)을 나타내는 값들을 하나의 열(row)로 표현한 것이다. 
# 텍스트 데이터를 벡터로 변환한다는 것은, 텍스트를 일종의 숫자 배열로 변경하여 머신러닝 모델이 이를 처리할 수 있도록 만든다는 것이다.
# 이는 컴퓨터는 텍스트를 직접 이해하지 못하기 때문에 숫자형 태의 데이터로 변경하는 것이다. 
# 텍스트와 같은 비정형 데이터(즉, 구조화되지 않은 데이터)를 다루려면 이를 숫자로 변환해야한다. 
# 텍스트 데이터를 벡터로 변환 (Bag of Words)

# 텍스트 데이터를 벡터로 변환하는데 사용되는 라이브러리
vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(texts) # 처음 불용어 제거 전

#-- 불용어 제거 후 사용
clean_texts = df["text"].apply(ct.clean_text) # 텍스트 전처리
# fit : 텍스트 데이터에서 단어 목록 생성
# transform : 문서를 단어의 등장 횟수로 표현하는 벡터로 변환
X = vectorizer.fit_transform(clean_texts) # 택스트 데이터를 벡터화 

# 훈련 데이터와 테스트 데이터로 분리
# x : 벡터화된 텍스트 데이터
# labels : 각 문서에 대한 레이블(스팸 여부)
# test_size : 전체 데이터에서 테스트 데이터 비율
# X_train, y_train : 훈련 데이터와 레이블
# X_test, y_test : 테스트 데이터와 레이블
# random_state= 값 : 균일한 테스트를 위해 동일한 데이터 분할을 목적으로 사용한다. 값의 인덱스를 기준으로 난수를 발생시킴.
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)


#--- 학습 단계
# 로지스틱 회귀 모델 학습
# 로지스틱 회귀 모델을 구현하는데 사용되는 것으로 max_iter= 1000은 모델이 학습 시 최대 반복 횟수를 1000으로 설정한다는 것이다.
# 로지스틱 회귀는 최적의 파라미터를 찾기 위해 반복적으로 학습하는데, 이 값을 설정함으로써 모델이 수렴할 때까지 최대 1000번 반복하게 된다. 
# 이 숫자가 너무 작으면 모델이 최적화되지 않을 수 있다.
model = LogisticRegression(max_iter=1000)
# fit 메서드는 모델을 훈련 데이터에 학습하는 과정이다.
# 학습이 완료되면, 모델은 새로운 데이터에 대한 예측을 수행할 준비가 된다.
model.fit(X_train, y_train)


# 모델 파일 저장 (Pickle 사용)
# 지정한 파일을 열고 작업이 끝나면 자동으로 닫아준다. wb는 파일을 쓰기 모드로 열고 바이너리 형식으로 저장
# model :학습된 로지스틱 회귀 모델을 저장하고 이 모델을 사용하여 다음 예측을 수행할 수 있도록 하기 위함이다.
with open("./model/LogisticRegression/spam_classifier_model.pkl", "wb") as model_file: 
    pickle.dump(model, model_file) # 학습된 모델을 지정한 파일에 저장한다. pickle은 파이썬 객체로 파일을 저장하고 나중에 다시 모드할 수 있도록 하는 모듈이다.
    print("모델이 './model/LogisticRegression/spam_classifier_model.pkl' 파일로 저장되었습니다.")
    
# CountVectorizer도 함께 저장 (나중에 로드할 때 일관성 유지)
# 텍스트 데이터를 벡터로 변환하는데 사용된다.
# 모델을 저장한 후 나중에 새로운 데이터에 대한 예측을 수행할 때, 
# 동일한 방식으로 텍스트를 벡터화해야 하는데 이 파일을 활용하여 훈련 데이터와 같은 방법으로 새로운 텍스트를 변환할 수 있게 된다. 
with open("./model/LogisticRegression/vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)
    print("벡터라이저가 'vectorizer.pkl' 파일로 저장되었습니다.")

# 테스트 데이터로 예측
y_pred = model.predict(X_test)

# 모델 성능 평가 (정확도)
# 실제 레이블과 모델의 예측 레이블을 비교항 정확도를 계산
accuracy = accuracy_score(y_test, y_pred)
print(f"모델의 테스트 데이터 정확도: {accuracy * 100:.2f}%")




#--- 성능 평가 단계
# 모델의 성능을 평가하고 결과를 시각화하는 단계

# 1. 혼동 행렬 출력
# 혼동 행렬
# 분류 모델의 성능을 평가하기 위한 도구로, 실제 클래스와 예측 클래스 간의 관계를 나타내는 표이다. 
# 일반적으로 이진 분류 문제에서 사용된다.
# 진짜 긍정(True Positive, TP) : 실제로 긍정 클래스인 샘플을 모델이 긍정으로 올바르게 예측한 수.
# 진짜 부정(True Positive, TN) : 실제로 부정 클래스인 샘플을 모델이 부정으로 올바르게 예측한 수.
# 거짓 긍정(False Positive, FP) : 실제로 부정 클래스인 샘플을 모델이 긍정으로 잘못 예측한 수.
# 거짓 부정(False Negative, FN) : 실제로 긍정 클래스인 샘플을 모델이 부정으로 잘못 예측한 수.
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Spam', 'Spam'], yticklabels=['Not Spam', 'Spam'])
plt.title('혼동 행렬')
plt.ylabel('실제값')
plt.xlabel('예측값')
plt.show()

# 2. 분류 보고서 출력
# Precision(정밀도) : 모델이 긍정으로 예측한 것 중 실제로 긍정인 비율
# recall(재현율) : 실제 긍정인 것 중에서 모델이 긍정으로 올바르게 예측한 비율
# F1-score : 정밀도와 재현율의 조화 평균 
# suppor(지원 수 ) : 각 클래스의 정의 (안 스팸과 스팸)
# Accuracy(정확도) : 전체 샘플 중에서 올바르게 예측한 비율
# Macro avg (매크로 평균) : 클래스의 수를 동일하게 하여 각 클래스의 성능 지표를 계산한 것
# Weighted avg (가중 평균) : 실제 데이터의 분포를 반영하여 성능을 평가한 것

print("\n분류 보고서:")
print(classification_report(y_test, y_pred, target_names=['Not Spam', 'Spam']))

# 3. 모델이 스팸 분류에 사용한 단어별 가중치 확인
# vectorize에서 사용된 단어 목록들을 가져온다.
feature_names = vectorizer.get_feature_names_out()

# 학습된 로지스틱 회귀 모델에서 각 단어에 대한 가중치(계수)를 가져온다.
coefficients = model.coef_[0]
coef_df = pd.DataFrame({'Word': feature_names, 'Coefficient': coefficients})

# 스팸과 관련된 단어 (가중치가 높은 순으로 상위 10개)
top_spam_words = coef_df.sort_values(by='Coefficient', ascending=False).head(10)
print("\n스팸과 관련된 단어 상위 10개:")
print(top_spam_words)

# 정상 이메일과 관련된 단어 (가중치가 낮은 순으로 상위 10개)
top_non_spam_words = coef_df.sort_values(by='Coefficient').head(10)
print("\n정상 이메일과 관련된 단어 상위 10개:")
print(top_non_spam_words)