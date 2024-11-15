import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# 1. 데이터 로드 및 전처리
image_folder = './model_data/train'

# 사용자 이미지 데이터 로드 및 전처리 함수
def load_and_preprocess_images(image_folder, target_size=(28, 28)):
    images = []
    for file_name in os.listdir(image_folder):
        img_path = os.path.join(image_folder, file_name)
        img = Image.open(img_path).convert("L")  # 흑백으로 변환
        img = img.resize(target_size)  # (28, 28) 크기로 조정
        img_array = np.array(img) / 255.0  # 0~1로 정규화
        images.append(img_array)
    images = np.array(images)
    return np.expand_dims(images, axis=-1)  # (num_images, 28, 28, 1) 형태로 변환

# 사용자 이미지 로드
train_images = load_and_preprocess_images(image_folder)

# 2. Autoencoder 모델 정의
encoding_dim = 64  # 특성 벡터의 차원
input_shape = (28, 28, 1)


# 인코더 정의
encoder_input = layers.Input(shape=input_shape)
x = layers.Flatten()(encoder_input)
x = layers.Dense(128, activation='relu')(x)
encoder_output = layers.Dense(encoding_dim, activation='relu')(x)

# 디코더 정의
x = layers.Dense(128, activation='relu')(encoder_output)
x = layers.Dense(784, activation='sigmoid')(x)
decoder_output = layers.Reshape((28, 28, 1))(x)

# Autoencoder 모델 생성
autoencoder = models.Model(encoder_input, decoder_output)
encoder = models.Model(encoder_input, encoder_output)  # 인코더 부분만 분리

# 3. Autoencoder 학습
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(train_images, train_images, epochs=10, batch_size=256, shuffle=True)

# 4. 인코더를 사용하여 훈련 데이터에서 특징 벡터 추출
train_features = encoder.predict(train_images)
train_features = train_features.reshape(-1, encoding_dim)  # (60000, 64)

# 5. KMeans를 사용하여 군집화
num_clusters = 10  # 군집 개수 설정
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
train_labels_pred = kmeans.fit_predict(train_features)

# 6. 시각화: 각 군집에서 일부 샘플 이미지를 표시
def plot_cluster_examples(images, labels, cluster_num, num_examples=10):
    cluster_indices = np.where(labels == cluster_num)[0]
    plt.figure(figsize=(10, 1))
    for i, idx in enumerate(cluster_indices[:num_examples]):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(images[idx].squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()

# 예시: 각 군집에서 대표 이미지 10개씩 출력
for cluster in range(num_clusters):
    print(f"Cluster {cluster} examples:")
    plot_cluster_examples(train_images, train_labels_pred, cluster)
