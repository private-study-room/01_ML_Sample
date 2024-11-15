import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 로드 및 전처리
fashion_mnist = tf.keras.datasets.fashion_mnist # 텐서플로에 포함된 데이터셋을 불러온다.
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data() # 데이터를 훈련용과 테스트용으로 구분
# 데이터는 훈련 데이터(6만개)와 테스트 데이터(1만개)로 이미 구분이 되어있다.


# 이미지 정규화 및 채널 차원 추가
train_images = train_images / 255.0 
# 이미지 픽셀 값은 0부터 255 사이의 정수 값으로 되어 있다. 
# 이 값을 train_images = train_images/255.0을 통해 0과 1사이의 범위로 정규화를 한다.
# 이를 통해 학습 속도가 빨라지고 수렴 과정이 안정화된다.
# 그러 수렴이란?


train_images = np.expand_dims(train_images, axis=-1)  # (60000, 28, 28, 1)
# 초기 형태 6만개의 이미지 28x28 크기 픽셀의 이미지를 CNN에 입력하기 위해서는 4차원의 형식이 필요하기 때문에 axis=-1 값을 입력하여 차원을 추가한다.
# 여기서 -1의 값은 해당 자원을 자동으로 계산하라는 의미로 배열의 크기를 알 수 없거나 특정 차원에 대해 자동으로 크기를 맞추고 싶을때 -1을 사용한다.
# 이를 통해 (배치크기, 높이 너비, 채널 수)를 cnn에 맞는 형식으로 제공하게 된다.
test_images = test_images / 255.0
test_images = np.expand_dims(test_images, axis=-1)

# 배치 사이즈
# 모델이 한 번의 가중치 업데이트를 위해 처리하는 데이터 샘플의 개수를 의미한다.
# 훈련 데이터를 여러 개의 작은 배치로 나누어 각 배치마다 순전파와 역전파 과정을 거쳐 모델의 가중치를 업데이트
# 배치 사이즈가 작을 수록 모델 가중치가 더 자주 업데이트 된다.
# 배치 사이즈가 32인 경우 32개의 샘플마다 가중치를 업데이트 하게된다.



# 2. Autoencoder 모델 정의
encoding_dim = 64  
# 출력 차원, 즉 압축된 특징 벡터의 크기를 의미한다.
# 28*28 크기의 이미지를 64차원의 벡터로 변환하기 위해서 값을 할당

input_shape = (28, 28, 1)
# Autoencoder 모델에 입력되는 데이터의 형태이다.
# 28*28 크기의 흑백 이미지를 의미하며, 1은 채널 수 이다.
# 흑백 이미지는 채널이 1개이고, 컬러 이미지는 3이다.


# 인코더 정의
encoder_input = layers.Input(shape=input_shape)
# 이미지를 입력받을 형식을 정의하는 레이어이다.

x = layers.Flatten()(encoder_input) # 이미지를 1차원 벡터로 변환
# 이미지를 1차원 백터로 펼친다. 이미지를 (784) 형태의 1차원 벡터로 변환
# 이를 통해 모델이 이미지를 처리하기 쉬운 형태로 만든다.

x = layers.Dense(128, activation='relu')(x) # 128 차원으로 축소하고 relu 활성화 함수를 적용한다.
# ReLU 활성화 함수 (activation='relu')를 적용하여 비선형 변환을 적용하고, 모델이 이미지의 복잡한 패턴을 학습할 수 있게한다.

encoder_output = layers.Dense(encoding_dim, activation='relu')(x)  
# 128차원을 최종적으로 encoding_di(64) 크기의 저차원 벡터로 축소한다. 
# Relu 활성화 함수가 적용되어 비선형 변환이 이루어진다.
# 이를 통해 잠재 표현(letent Representation)이 되며 이 잠재 표현은 메모리를 절약하며 데이터의 주요 특성을 학습하여 
# kmeans와 같은 군집화 작업에 유용하게 사용된다.

# 압축을 하는 이유
# 고차원 데이터를 저차원으로 압축하면서 중요한 정보는 보존하고 중요하지 않은 정보를 날림으로써 효율적으로 데이터의 주요 특징을 추출하기 위함이다.

# 디코더 정의
# 64차원으로 압축한 데이터를 28*28로 복원하는 역할을 한다.
x = layers.Dense(128, activation='relu')(encoder_output)
# 인코더의 출력인 64차원의 벡터를 128차원으로 확장한다.
# 인코더의 축소 과정을 반대로 하여 이미지를 복원하기 위해 차원을 늘려주는 역할을 한다.
# 활성홤수에 relu를 적용하여 비선형 변환을 적용하고 모델이 복잡한 패턴을 학습할 수 있도록 한다.

x = layers.Dense(784, activation='sigmoid')(x)
# 128차원의 출력 벡터를 784차원으로 확장하여 원본 이미지의 모든 픽셀 값을 복원할 수 있도록 한다.
# 활성함수로 sigmoid 함수를 사용하며 출력 값을 0 ~ 1 사이로 조정한다. 
# 이미지의 픽셀값을 정규화된 범위로 재구성하기 위해 필요하다.

decoder_output = layers.Reshape((28, 28, 1))(x)
# 784차원 벡터를 (28,28,1) 형식으로 변환하여 원래 이미지 형식과 동일하게 만든다.
# reshape 레이어는 벡터를 2차원 이미지로 재구성하고 1채널을 추가하여 최종 출력 형태를 (28,28,1)로 만든다.



# Autoencoder 모델 생성
autoencoder = models.Model(encoder_input, decoder_output)
# modol을 생성하고 28*28 크기의 이미지가 인코더를 거쳐 64차원으로 압축되고 디코더를 통해 원본(28,28,1)로 복원된다.

encoder = models.Model(encoder_input, encoder_output)  # 인코더 부분만 분리
# autoencoder 모델에서 인코더 부분만 별도로 분리한 모델이다.
# 인코더만 따로 분리하여 이미지를 64차원의 특징 벡터로 변환할 수 있도록 하기 위해서 사용된다.


# 3. Autoencoder 학습
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# Adam 옵티마이저를 사용하여 학습 속도를 최적화한다.
# loss="binary_crossentropy" 손실 함수를 사용하여 입력 이미지와 출력 이미지 간의 차이를 최소화한다.
# 이는 이미지의 각 픽셀 값을 0과 1사이로 정규화하기 때문에 적절한 손실 함수이다.

autoencoder.fit(train_images, train_images, epochs=10, batch_size=256, shuffle=True)
# fit 함수로 모델을 학습시킨다.
# train_images를 입력과 출력으로 사용하여, 모델이 이미지 데이터를 압축하고 복원하는 방법을 학습
# epochs=10 전체 데이터셋을 10번 반복 학습
# batch_size 한 번 가중치 업데이트 마다 256개의 샘플을 사용한다.
# shuffle 각 에포크가 끝날 때마다 데이터를 무작위로 섞어서 학습 효율을 높인다.

# 4. 인코더를 사용하여 훈련 데이터에서 특징 벡터 추출
train_features = encoder.predict(train_images)
# 훈련 이미지에서 인코더를 사용하여 64차원 특징벡터를 추출한다.
# 여기서 train_images는 초기 학습 데이터를 답고 있지만 모델을 거치고 나면 특징 데이터를 포함하고 있는 상태가된다.

train_features = train_features.reshape(-1, encoding_dim)  # (60000, 64)
# 64차원 특징 벡터는 원본 이미지의 중요한 특징을 담고 있어 이후 군집화와 (Kmeans)와 같은 비지도 학습에 활용한다.
# 이미지의 주요 정보를 64차원 벡터로 표현하고 있는 형태로 군집화나 유사도 계산에 사용하기 적합한 구조이다.



# 5. KMeans를 사용하여 군집화
num_clusters = 10  # 군집 개수 설정
# 군집의 개수를 10으로 설정하여 kmeans가 train_features 데이터를 10개의 그룹으로 나누도록 설정하기 위함이다.
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
# 10개의 군집을 만들도록 KMeans 알고리즘을 초기화한다.
# random_state는 모델이 재현성을 보장하기 위해 랜덤 값을 추출하고 고정한다.

train_labels_pred = kmeans.fit_predict(train_features)
# kmeans 군집화 모델을 학습하고, 각 샘플이 속하는 군집 레이블을 예측하여 train_lables_pred에 저장한다.
# train_features는 (60000, 64) 형태의 배열이므로, KMeans 알고리즘은 
# 각 64차원 특징 벡터를 기반으로 데이터를 10개의 군집으로 나누고, 데이터가 속한 군집의 레이블을 반환한다.



# 6. 시각화: 각 군집에서 일부 샘플 이미지를 표시
def plot_cluster_examples(images, labels, cluster_num, num_examples=10):
    cluster_indices = np.where(labels == cluster_num)[0] # labels에서 cluster_num과 일치하는 인덱스만 추출하여 저장
    plt.figure(figsize=(10, 1)) # 이미지 사이즈 설정
    for i, idx in enumerate(cluster_indices[:num_examples]): # 특정 군집에 속하는 첫 10개 이미지의 인덱스만 선택
        # enumerate 함수는 리스트나 배열을 순회할 때 각 항목의 인덱스와 값을 동시에 반환하는 함수

        plt.subplot(1, num_examples, i + 1) # 이미지를 가로로 나열해서 시각화
        plt.imshow(images[idx].squeeze(), cmap='gray') #차원을 제거하고 이미지를 흑백으로 표시
        plt.axis('off') # 플롯의 축을 제거하고 이미지에 집중하기 위함
    plt.show()

# 예시: 각 군집에서 대표 이미지 10개씩 출력
for cluster in range(num_clusters): # 0 ~ num_clusters-1 까지 반복
    print(f"Cluster {cluster} examples:")
    plot_cluster_examples(train_images, train_labels_pred, cluster)
