import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.utils import to_categorical

# 데이터셋 로드
lines = pd.read_csv('./kor.txt', names=['src', 'tar', 'lic'], sep='\t')
lines = lines.loc[:, 'src':'tar']
lines = lines[0:10000]  # 샘플 60,000개로 제한

# 문자 집합 생성
src_vocab = sorted(set(char for line in lines.src for char in line))  # 인코더 문자 집합
tar_vocab = sorted(set(char for line in lines.tar for char in line))  # 디코더 문자 집합

# 인덱스 매핑 생성
src_to_index = {char: i + 1 for i, char in enumerate(src_vocab)}
tar_to_index = {char: i + 1 for i, char in enumerate(tar_vocab)}
index_to_tar = {i: char for char, i in tar_to_index.items()}

# 데이터 전처리
max_src_len = max(lines.src.apply(len))
max_tar_len = max(lines.tar.apply(len))

# 인코더 입력 데이터 준비
encoder_input_data = np.zeros((len(lines), max_src_len, len(src_vocab) + 1), dtype='float32')
for i, line in enumerate(lines.src):
    for t, char in enumerate(line):
        encoder_input_data[i, t, src_to_index[char]] = 1.0

# 디코더 입력 데이터 준비
decoder_input_data = np.zeros((len(lines), max_tar_len, len(tar_vocab) + 1), dtype='float32')
for i, line in enumerate(lines.tar):
    for t, char in enumerate(line):
        decoder_input_data[i, t, tar_to_index[char]] = 1.0

# 디코더 타겟 데이터 준비 (shifted version of decoder input)
decoder_target_data = np.zeros((len(lines), max_tar_len, len(tar_vocab) + 1), dtype='float32')
for i, line in enumerate(lines.tar):
    for t, char in enumerate(line[1:]):  # 첫 번째 문자 제외
        decoder_target_data[i, t, tar_to_index[char]] = 1.0

# 인코더 설정
input_vocab_size = len(src_vocab) + 1  # 패딩 추가
output_vocab_size = len(tar_vocab) + 1  # 패딩 추가
encoder_inputs = Input(shape=(None, input_vocab_size))
encoder_lstm = LSTM(units=128, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]

# 디코더 입력 정의
decoder_inputs = Input(shape=(None, output_vocab_size))
decoder_lstm = LSTM(units=128, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

# Dense 레이어로 출력을 처리
decoder_dense = Dense(output_vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 인코더와 디코더 연결해 모델 정의
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
# 모델 컴파일
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# 모델 학습
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=32, epochs=50, validation_split=0.2)


# Google Drive 마운트
# drive.mount('/content/drive')

# 모델 저장 (Google Drive에)
model.save('./seq2seq.h5')
print("모델이 Google Drive에 저장되었습니다.")
