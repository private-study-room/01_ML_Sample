
import pickle

# 저장된 모델 및 벡터라이저 파일을 로드
def load_model_and_vectorizer(model_path="./model/LogisticRegression/spam_classifier_model.pkl", vectorizer_path="./model/LogisticRegression/vectorizer.pkl"):
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
        print(f"모델이 '{model_path}'에서 로드되었습니다.")
    
    with open(vectorizer_path, "rb") as vec_file:
        vectorizer = pickle.load(vec_file)
        print(f"벡터라이저가 '{vectorizer_path}'에서 로드되었습니다.")
    
    return model, vectorizer

# 모델 및 벡터라이저 로드
loaded_model, loaded_vectorizer = load_model_and_vectorizer()

# 새로운 이메일 예측
new_emails = ["Win a free car today", "Please send me the project report"]
new_emails.extend([
    "Please come to our store and buy the product",
    "Advertisement Buy your products at our store"
])

# new_emails 값을 벡터로 변환
new_X = loaded_vectorizer.transform(new_emails)
# 텍스트 예측 수행
predictions = loaded_model.predict(new_X)

# 예측 결과 출력
for email, label in zip(new_emails, predictions):
    print(f"'{email}'이 이메일은 {'Spam' if label == 1 else 'Not Spam'}으로 구분됩니다.")
