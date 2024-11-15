import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense

# 데이터셋 로드
lines = pd.read_csv('./kor.txt', names=['src', 'tar', 'lic'], sep='\t')
lines = lines.loc[:, 'src':'tar']

# 시작 및 종료 심볼 추가
lines['tar'] = '<sos> ' + lines['tar'] + ' <eos>'

# 문자 집합 생성
src_vocab = sorted(set(char for line in lines.src for char in line))  # 인코더 문자 집합
tar_vocab = sorted(set(char for line in lines.tar for char in line))  # 디코더 문자 집합

# 인덱스 매핑 생성
src_to_index = {char: i + 1 for i, char in enumerate(src_vocab)}
tar_to_index = {char: i + 1 for i, char in enumerate(tar_vocab)}
index_to_tar = {i: char for char, i in tar_to_index.items()}

# 데이터 전처리
max_src_len = max(lines.src.apply(len))  # 인코더 입력 최대 길이
max_tar_len = max(lines.tar.apply(len))  # 디코더 입력 최대 길이

# 인코더 입력 데이터 준비
encoder_input_data = np.zeros((len(lines), max_src_len, len(src_vocab) + 1), dtype='float32')
for i, line in enumerate(lines.src):
    for t, char in enumerate(line):
        if char in src_to_index:  # 인덱스에 존재하는지 확인
            encoder_input_data[i, t, src_to_index[char]] = 1.0

# 디코더 입력 데이터 준비
decoder_input_data = np.zeros((len(lines), max_tar_len, len(tar_vocab) + 1), dtype='float32')
for i, line in enumerate(lines.tar):
    for t, char in enumerate(line.split()):
        if char in tar_to_index:
            decoder_input_data[i, t, tar_to_index[char]] = 1.0

# 디코더 타겟 데이터 준비 (shifted version of decoder input)
decoder_target_data = np.zeros((len(lines), max_tar_len, len(tar_vocab) + 1), dtype='float32')
for i, line in enumerate(lines.tar):
    for t, char in enumerate(line.split()[1:]):  # 첫 번째 문자 제외
        if char in tar_to_index:
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

# 모델 저장
model.save('./seq2seq.h5')
print("모델이 저장되었습니다.")

# --- 추론 모델 정의 ---

# 인코더 모델 정의
encoder_model = Model(encoder_inputs, encoder_states)

# 디코더 모델 정의
decoder_state_input_h = Input(shape=(128,))
decoder_state_input_c = Input(shape=(128,))
decoder_inputs_single = Input(shape=(None, output_vocab_size))
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs_single, initial_state=[decoder_state_input_h, decoder_state_input_c]
)
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model(
    [decoder_inputs_single, decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs, state_h, state_c]
)

# 번역 함수 정의
def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq)

    # 디코더의 첫 입력으로 <sos> 제공
    target_seq = np.zeros((1, 1, output_vocab_size))
    target_seq[0, 0, tar_to_index['<sos>']] = 1.0  # <sos> 심볼로 설정

    decoded_sentence = ""
    stop_condition = False
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_to_tar.get(sampled_token_index, '')  # 안전하게 인덱스를 가져옴
        decoded_sentence += " " + sampled_char

        if sampled_char == '<eos>' or len(decoded_sentence) > max_tar_len:
            stop_condition = True

        target_seq = np.zeros((1, 1, output_vocab_size))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]

    return decoded_sentence.strip('<eos>').strip()  # <eos> 심볼 제거

# 번역할 문장 예시
input_sentence = "Hello"  # 번역할 영어 문장

# 인코딩
encoder_input_seq = np.zeros((1, max_src_len, input_vocab_size))
for t, char in enumerate(input_sentence):
    if char in src_to_index:
        encoder_input_seq[0, t, src_to_index[char]] = 1.0

# 번역 수행
translated_sentence = decode_sequence(encoder_input_seq)
print("번역 결과:", translated_sentence)
