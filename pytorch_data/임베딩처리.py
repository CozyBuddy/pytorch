from sklearn.feature_extraction.text import TfidfVectorizer

# tfidf_vectorizer = TfidfVectorizer(
#     input='content',
#     encoding='utf-8',
#     lowercase=True,
#     stop_words=None,
#     ngram_range=(1,1),
#     max_df=1.0,
#     min_df=1,
#     vocabulary=None,
#     smooth_idf=True
# )

# corpus = ['That movice is famous movie','I like that actor' , "I don't like that actor"]

# #tfidf_vectorizer = TfidfVectorizer()
# tfidf_vectorizer.fit(corpus)
# tfidf_matrix = tfidf_vectorizer.transform(corpus)

# print(tfidf_matrix.toarray())
# print(tfidf_vectorizer.vocabulary_)


from torch import nn

class VanillaSkipgram(nn.Module):
    def __init__(self, vocab_size,  embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim = embedding_dim
        )

        self.linear = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size
        )

    def forward(self , input_ids):
        embeddings= self.embedding(input_ids)
        output = self.linear(embeddings)
        return output
    

import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt

corpus = Korpora.load('nsmc')
corpus = pd.DataFrame(corpus.test)

tokenizer = Okt()
tokens = [tokenizer.morphs(review) for review in corpus.text]

print(tokens[:3])

from collections import Counter

def build_vocab(corpus, n_vocab , special_tokens):
    counter = Counter()
    for tokens in corpus:
        counter.update(tokens)
    vocab = special_tokens
    for token , count in counter.most_common(n_vocab):
        vocab.append(token)

    return vocab

vocab = build_vocab(corpus = tokens , n_vocab=5000 , special_tokens=["<unk>"])
token_to_id = {token : idx for idx , token in enumerate(vocab)}
id_to_token = {idx : token for idx,token in enumerate(vocab)}

print(vocab[:10])
print(len(vocab))

def get_word_pairs(tokens, window_size):
    pairs = []
    for sentence in tokens:
        sentence_length = len(sentence)
        for idx,  center_word in enumerate(sentence):
            window_start = max(0,idx - window_size)
            window_end = min(sentence_length , idx + window_size + 1)
            center_word = sentence[idx]
            context_words = sentence[window_start : idx] + sentence[idx+1:window_end]
            for context_word in context_words:
                pairs.append([center_word, context_word])

    return pairs

word_pairs = get_word_pairs(tokens, window_size=2)
print(word_pairs[:5])


def get_index_pairs(word_pairs , token_to_id):
    pairs = []
    unk_index = token_to_id["<unk>"]
    for word_pair in word_pairs:
        center_word ,context_word = word_pair
        center_index = token_to_id.get(center_word,unk_index)
        context_index = token_to_id.get(context_word, unk_index)
        pairs.append([center_index , context_index])

    return pairs

index_pairs = get_index_pairs(word_pairs ,token_to_id)

print(index_pairs[:5])

import torch
from torch.utils.data import TensorDataset , DataLoader

index_pairs = torch.tensor(index_pairs)
center_indexes = index_pairs[: , 0]
context_indexes = index_pairs[: ,1]

dataset = TensorDataset(center_indexes , context_indexes)
dataloader = DataLoader(dataset ,batch_size=32 , shuffle=True)

from torch import optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'
word2vec = VanillaSkipgram(vocab_size=len(token_to_id) ,embedding_dim=128).to(device)
criterion = nn.CrossEntropyLoss().to(device)
print(device)
optimizer = optim.SGD(word2vec.parameters() , lr=0.1)


for epoch in range(10):
    cost = 0.0
    for input_ids , target_ids in dataloader:
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        logits = word2vec(input_ids)
        loss = criterion(logits,target_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost+= loss

    cost = cost / len(dataloader)
    print(f'Epoch : {epoch+1:4d} , Cost : {cost:.3f}')
