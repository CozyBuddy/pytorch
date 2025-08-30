from datasets import load_dataset

dataset = load_dataset('daekeun-ml/naver-news-summarization-ko')

data = dataset

print(data)

print(data['train']['document'][0])

ko_text = "".join(data['train']['document'])
ko_chars = sorted(list(set((ko_text))))
ko_vocab_size = len(ko_chars)

print('총 글자수' ,  ko_vocab_size)