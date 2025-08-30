import json
import pandas as pd

# json 파일 불러오기
df = pd.read_json("일상생활및구어체_한영_train_set.json")  # 파일 경로 맞춰주세요

print(df['data'])
# 'data' 컬럼 안에 dict가 들어있다고 가정
sentences = []
for d in df['data']:
    # 'data_set' 안에 한글 문장 추출
    sentences.append(d['ko_original'].replace('>',''))
    sentences.append(d['mt'].replace('>',''))

# 확인
print(sentences[:5])

#SentencePiece 학습용 파일로 저장
with open("enko_corpus.txt", "w", encoding="utf-8") as f:
    for s in sentences:
        f.write(s + "\n")
