

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

# import pandas as pd

# df = pd.read_excel('koren_emotion.xlsx')


# df = df[['Sentence' ,'Emotion']]

# def change_emotion(e):
#     if(e=='공포'):
#         return 0
#     elif(e=='놀람'):
#         return 1
#     elif(e=='분노'):
#         return 2
#     elif(e=='슬픔'):
#         return 3
#     elif(e=='중립'):
#         return 4
#     elif(e=='행복'):
#         return 5
#     elif(e=='혐오'):
#         return 6


# df['Emotion'] = df['Emotion'].apply(change_emotion)

# ntrain = int(len(df)*0.95)
# from sklearn.utils import shuffle

# df = shuffle(df, random_state=42)

# train = df['Sentence'].iloc[:ntrain]
# test = df['Sentence'].iloc[ntrain:]

# train_label = df['Emotion'].iloc[:ntrain]
# test_label = df['Emotion'].iloc[ntrain:]


# print(df.head().to_markdown())

from konlpy.tag import Okt
# from collections import Counter

# def build_vocab(corpus,n_vocab , special_tokens):
#     counter = Counter()
#     for tokens in corpus:
#         counter.update(tokens)
#     vocab = special_tokens

#     for token , count in counter.most_common(n_vocab):
#         vocab.append(token)
#     return vocab

tokenizer = Okt()
# train_tokens = [tokenizer.morphs(review) for review in train]
# test_tokens = [ tokenizer.morphs(review) for review in test]

import pickle

# vocab 불러오기
vocab = []
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
token_to_id = {token : idx for idx , token in enumerate(vocab)}
id_to_token = { idx : token for idx ,token in enumerate(vocab)}

# print(vocab[:10])
# print(len(vocab))

# import numpy as np

# def pad_sequences(sequences , max_length , pad_value):
#     result = list()
#     for sequence in sequences:
#         sequence = sequence[:max_length]
#         pad_length = max_length - len(sequence)
#         padded_sequence = sequence + [pad_value] * pad_length
#         result.append(padded_sequence)
#     return np.asarray(result)

# unk_id = token_to_id['<unk>']
# train_ids = [
#     [token_to_id.get(token,unk_id) for token in review] for review in train_tokens
# ]
# test_ids = [
#     [token_to_id.get(token,unk_id) for token in review] for review in test_tokens
# ]

# max_length =32

# pad_id = token_to_id['<pad>']
# train_ids = pad_sequences(train_ids, max_length , pad_id)
# test_ids = pad_sequences(test_ids , max_length , pad_id)


# import torch
# from torch.utils.data import TensorDataset, DataLoader

# train_ids = torch.tensor(train_ids)
# test_ids = torch.tensor(test_ids)

# train_labels = torch.tensor(train_label.values , dtype=torch.long)
# test_labels = torch.tensor(test_label.values , dtype=torch.long)

# train_dataset = TensorDataset(train_ids , train_labels)
# test_dataset = TensorDataset(test_ids , test_labels)

# train_loader = DataLoader(train_dataset , batch_size=32 , shuffle=True)
# test_loader = DataLoader(test_dataset,  batch_size=32 , shuffle=False)

# from torch import optim

n_vocab = len(token_to_id)
hidden_dim  = 64
embedding_dim = 128
n_layers= 3

# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# classifier = SentenceClassifier(
#     n_vocab=n_vocab , hidden_dim=hidden_dim , embedding_dim = embedding_dim , n_layers = n_layers 
# ).to(device)

# criterion = nn.CrossEntropyLoss().to(device)
# optimizer = optim.Adam(classifier.parameters() , lr=0.001)

# def train(model , datasets , criterion , optimizer, device, interval):
#     model.train()
#     losses = list()

#     for step , (input_ids, labels) in enumerate(datasets):
#         input_ids = input_ids.to(device)
#         labels = labels.to(device).long()

#         logits = model(input_ids)
#         loss = criterion(logits, labels)
#         losses.append(loss.item())

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         if step%interval == 0 :
#             print(f'Train loss {step} : { np.mean(losses)}')


# def test(model , datasets, criterion , device):
#     model.eval()
#     losses = list()
#     corrects = list()

#     for step , (input_ids,labels) in enumerate(datasets):
#         input_ids = input_ids.to(device)
#         labels = labels.to(device).long()

#         logits = model(input_ids)
#         loss = criterion(logits , labels)
#         losses.append(loss.item())
#         yhat = torch.argmax(logits , dim=1)

#         corrects.extend(
#             torch.eq(yhat , labels).cpu().tolist()
#         )


#     print(f'Val Loss : { np.mean(losses)} , Val Accuracy : {np.mean(corrects)}')



# Load the saved state_dict
import torch
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
loaded_model = SentenceClassifier(
    n_vocab=n_vocab,
    hidden_dim=hidden_dim,
    embedding_dim=embedding_dim,
    n_layers=n_layers
).to(device)

loaded_model.load_state_dict(torch.load("Sentencemodel.pth"))
loaded_model.eval() # Set the model to evaluation mode

# Assuming you have the same tokenizer, token_to_id, and max_length from training
from konlpy.tag import Okt
tokenizer = Okt()
token_to_id = {token: idx for idx, token in enumerate(vocab)}
unk_id = token_to_id['<unk>']
pad_id = token_to_id['<pad>']
max_length = 32

def predict_emotion(sentence, model, tokenizer, token_to_id, unk_id, pad_id, max_length, device):
    # Tokenize the sentence
    tokens = tokenizer.morphs(sentence)
    
    # Convert tokens to IDs
    ids = [token_to_id.get(token, unk_id) for token in tokens]
    
    # Pad the sequence
    if len(ids) > max_length:
        ids = ids[:max_length]
    else:
        pad_length = max_length - len(ids)
        ids = ids + [pad_id] * pad_length
    
    # Convert to a PyTorch tensor
    input_tensor = torch.tensor(ids).unsqueeze(0).to(device) # Add a batch dimension

    # Use the model to get logits
    with torch.no_grad():
        logits = model(input_tensor)
        
    # Get the predicted class index
    predicted_class_index = torch.argmax(logits, dim=1).item()
    
    return predicted_class_index

# Map emotion indices back to their names
emotion_map = {
    0: '공포', 1: '놀람', 2: '분노', 3: '슬픔', 4: '중립', 5: '행복', 6: '혐오'
}

# Example sentences
test_sentence_1 = "진짜 짜증난다, 어떻게 이럴 수가 있지?"
test_sentence_2 = "너무 행복해! 오늘 최고의 날이야."
test_sentence_3 = "무서워서 아무것도 할 수가 없어."
test_sentence_4 = "아침에 눈을 떴을때"
# Run predictions
predicted_index_1 = predict_emotion(test_sentence_1, loaded_model, tokenizer, token_to_id, unk_id, pad_id, max_length, device)
predicted_emotion_1 = emotion_map.get(predicted_index_1, "알 수 없음")

predicted_index_2 = predict_emotion(test_sentence_2, loaded_model, tokenizer, token_to_id, unk_id, pad_id, max_length, device)
predicted_emotion_2 = emotion_map.get(predicted_index_2, "알 수 없음")

predicted_index_3 = predict_emotion(test_sentence_3, loaded_model, tokenizer, token_to_id, unk_id, pad_id, max_length, device)
predicted_emotion_3 = emotion_map.get(predicted_index_3, "알 수 없음")

predicted_index_4 = predict_emotion(test_sentence_4, loaded_model, tokenizer, token_to_id, unk_id, pad_id, max_length, device)
predicted_emotion_4 = emotion_map.get(predicted_index_4, "알 수 없음")

print(f"'{test_sentence_1}' -> 예측 감정: {predicted_emotion_1}")
print(f"'{test_sentence_2}' -> 예측 감정: {predicted_emotion_2}")
print(f"'{test_sentence_3}' -> 예측 감정: {predicted_emotion_3}")
print(f"'{test_sentence_4}' -> 예측 감정: {predicted_emotion_4}")
