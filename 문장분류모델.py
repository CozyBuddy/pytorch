# import torch
# from torch import nn

# input_size = 128
# output_size = 256
# num_layers=3
# bidirectional= True
# proj_size= 64

# model = nn.LSTM(
#     input_size=input_size,
#     hidden_size=output_size,
#     num_layers=num_layers,
#     batch_first=True,
#     bidirectional=bidirectional,
#     proj_size=proj_size
  
# )

# batch_size=4
# sequence_len=6

# inputs = torch.randn(batch_size,sequence_len,input_size)
# h_0 = torch.randn(
#     num_layers * (int(bidirectional) +1) ,
#     batch_size,
#     proj_size if proj_size >0 else output_size
# )
# c_0 = torch.randn(num_layers * (int(bidirectional)+1) , batch_size , output_size)

# print(inputs.shape)
# print(h_0.shape)
# print(c_0.shape)
# outputs , (h_n , c_n) = model(inputs,( h_0 , c_0))

# print(outputs.shape, h_n.shape , c_n.shape)

from torch import nn

class SentenceClassifier(nn.Module):
    def __init__(self, n_vocab, hidden_dim , embedding_dim , n_layers , dropout=0.1 , bidirectional=True):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=n_vocab,
            embedding_dim = embedding_dim,
            padding_idx=0
        )

        self.model = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers= n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=True
        )

        if bidirectional:
            self.classifier = nn.Linear(hidden_dim*2,7)
        else:
            self.classifier = nn.Linear(hidden_dim,7)

        self.dropout = nn.Dropout(dropout)


    def forward(self,inputs):
        embedding = self.embedding(inputs)
        output, (h_n, c_n) = self.model(embedding)
        if self.model.bidirectional:
            last_hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        else:
            last_hidden = h_n[-1,:,:]
        logits = self.classifier(self.dropout(last_hidden))


        return logits

import pandas as pd

df = pd.read_excel('koren_emotion.xlsx')


df = df[['Sentence' ,'Emotion']]

def change_emotion(e):
    if(e=='공포'):
        return 0
    elif(e=='놀람'):
        return 1
    elif(e=='분노'):
        return 2
    elif(e=='슬픔'):
        return 3
    elif(e=='중립'):
        return 4
    elif(e=='행복'):
        return 5
    elif(e=='혐오'):
        return 6


df['Emotion'] = df['Emotion'].apply(change_emotion)

ntrain = int(len(df)*0.95)
from sklearn.utils import shuffle

df = shuffle(df, random_state=42)

train = df['Sentence'].iloc[:ntrain]
test = df['Sentence'].iloc[ntrain:]

train_label = df['Emotion'].iloc[:ntrain]
test_label = df['Emotion'].iloc[ntrain:]


print(df.head().to_markdown())

from konlpy.tag import Okt
from collections import Counter

def build_vocab(corpus,n_vocab , special_tokens):
    counter = Counter()
    for tokens in corpus:
        counter.update(tokens)
    vocab = special_tokens

    for token , count in counter.most_common(n_vocab):
        vocab.append(token)
    return vocab

tokenizer = Okt()
train_tokens = [tokenizer.morphs(review) for review in train]
test_tokens = [ tokenizer.morphs(review) for review in test]

vocab = build_vocab(corpus=train_tokens , n_vocab=20000 , special_tokens=['<pad>','<unk>'])

import pickle

# vocab 저장
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)


token_to_id = {token : idx for idx , token in enumerate(vocab)}
id_to_token = { idx : token for idx ,token in enumerate(vocab)}

print(vocab[:10])
print(len(vocab))

import numpy as np

def pad_sequences(sequences , max_length , pad_value):
    result = list()
    for sequence in sequences:
        sequence = sequence[:max_length]
        pad_length = max_length - len(sequence)
        padded_sequence = sequence + [pad_value] * pad_length
        result.append(padded_sequence)
    return np.asarray(result)

unk_id = token_to_id['<unk>']
train_ids = [
    [token_to_id.get(token,unk_id) for token in review] for review in train_tokens
]
test_ids = [
    [token_to_id.get(token,unk_id) for token in review] for review in test_tokens
]

max_length =32

pad_id = token_to_id['<pad>']
train_ids = pad_sequences(train_ids, max_length , pad_id)
test_ids = pad_sequences(test_ids , max_length , pad_id)


import torch
from torch.utils.data import TensorDataset, DataLoader

train_ids = torch.tensor(train_ids)
test_ids = torch.tensor(test_ids)

train_labels = torch.tensor(train_label.values , dtype=torch.long)
test_labels = torch.tensor(test_label.values , dtype=torch.long)

train_dataset = TensorDataset(train_ids , train_labels)
test_dataset = TensorDataset(test_ids , test_labels)

train_loader = DataLoader(train_dataset , batch_size=32 , shuffle=True)
test_loader = DataLoader(test_dataset,  batch_size=32 , shuffle=False)

from torch import optim

n_vocab = len(token_to_id)
hidden_dim  = 64
embedding_dim = 128
n_layers= 3

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

classifier = SentenceClassifier(
    n_vocab=n_vocab , hidden_dim=hidden_dim , embedding_dim = embedding_dim , n_layers = n_layers 
).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(classifier.parameters() , lr=0.001)

def train(model , datasets , criterion , optimizer, device, interval):
    model.train()
    losses = list()

    for step , (input_ids, labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device).long()

        logits = model(input_ids)
        loss = criterion(logits, labels)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%interval == 0 :
            print(f'Train loss {step} : { np.mean(losses)}')


def test(model , datasets, criterion , device):
    model.eval()
    losses = list()
    corrects = list()

    for step , (input_ids,labels) in enumerate(datasets):
        input_ids = input_ids.to(device)
        labels = labels.to(device).long()

        logits = model(input_ids)
        loss = criterion(logits , labels)
        losses.append(loss.item())
        yhat = torch.argmax(logits , dim=1)

        corrects.extend(
            torch.eq(yhat , labels).cpu().tolist()
        )


    print(f'Val Loss : { np.mean(losses)} , Val Accuracy : {np.mean(corrects)}')


epochs = 50
interval = 500

for epoch in range(epochs):
    train(classifier , train_loader, criterion ,optimizer, device ,interval)
    test(classifier , test_loader , criterion , device)

torch.save(classifier.state_dict(), "Sentencemodel.pth")




