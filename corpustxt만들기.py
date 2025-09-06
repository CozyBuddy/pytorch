import json
import pandas as pd

#json 파일 불러오기
df0 = pd.read_json("일상생활및구어체_한영_train_set.json")  # 파일 경로 맞춰주세요
df = pd.read_excel('korean_english_malmungchi/1_구어체(1).xlsx')
df2 = pd.read_excel('korean_english_malmungchi/1_구어체(2).xlsx')
df3 = pd.read_excel('korean_english_malmungchi/2_대화체.xlsx')
df4 = pd.read_excel('korean_english_malmungchi/3_문어체_뉴스(1)_200226.xlsx')
df5 = pd.read_excel('korean_english_malmungchi/3_문어체_뉴스(2).xlsx')
df6 = pd.read_excel('korean_english_malmungchi/3_문어체_뉴스(3).xlsx')
df7 = pd.read_excel('korean_english_malmungchi/3_문어체_뉴스(4).xlsx')
df8 = pd.read_excel('korean_english_malmungchi/4_문어체_한국문화.xlsx')
df9 = pd.read_excel('korean_english_malmungchi/5_문어체_조례.xlsx')
df10 = pd.read_excel('korean_english_malmungchi/6_문어체_지자체웹사이트.xlsx')

df2 = pd.concat([df,df2,df3,df4,df5,df6,df7,df8,df9,df10])

df0['원문'] = df0['data'].apply(pd.Series)['ko_original']
df0['번역문'] = df0['data'].apply(pd.Series)['mt']

df = pd.concat([df0,df2])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
# 'data' 컬럼 안에 dict가 들어있다고 가정
sentences = []
sentences2 = []
# for _ , d in df.iterrows():
#     # 'data_set' 안에 한글 문장 추출
#     sentences.append(d['원문'].str.replace('>',''))
#     sentences2.append(d['번역문'].str.replace('>',''))
sentences = df['원문'].str.replace('>','').tolist()
sentences2 = df['번역문'].str.replace('>','').tolist()

# 확인
print(df.shape)


#SentencePiece 학습용 파일로 저장
with open("ko_corpus.txt", "w", encoding="utf-8") as f:
    for s in sentences:
        f.write(s + "\n")

with open("en_corpus.txt", "w", encoding="utf-8") as f:
    for s in sentences2:
        f.write(s + "\n")


print(df)
