
import sentencepiece as spm
import pandas as pd


sp = spm.SentencePieceProcessor()
sp.load("spm_ko.model")




sp2 = spm.SentencePieceProcessor()
sp2.load("spm_en.model")


SRC_LANGUAGE = 'ko'
TGT_LANGUAGE = 'en'
UNK_IDX , BOS_IDX , EOS_IDX = sp.unk_id()  , sp.bos_id(), sp.eos_id()
PAD_IDX = sp.piece_to_id('<pad>') 


import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self,d_model , max_len , dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model , 2) * (-math.log(10000) / d_model)
        )

        pe = torch.zeros(max_len , 1 ,d_model)
        pe[: , 0 , 0::2] = torch.sin(position*div_term)
        pe[:,0,1::2] = torch.cos(position*div_term)
        self.register_buffer('pe',pe)

    def forward(self,x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size , emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size , emb_size)
        self.emb_size = emb_size

    def forward(self , tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    


class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers , num_decoder_layers , emb_size , max_len , nhead, src_vocab_size, tgt_vocab_size, dim_feedforward, dropout=0.1):
        super().__init__()
        self.src_tok_emb = TokenEmbedding(src_vocab_size,  emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size , emb_size)
        self.positional_encoding = PositionalEncoding(
            d_model = emb_size, max_len =max_len ,dropout=dropout
        )
        self.transformer = nn.Transformer(
            d_model = emb_size,
            nhead= nhead,
            num_encoder_layers = num_encoder_layers,
            num_decoder_layers = num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.generator = nn.Linear(emb_size , tgt_vocab_size )


    def forward(self, src, trg ,src_mask ,tgt_mask , src_padding_mask , tgt_padding_mask , memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask = src_mask,
            tgt_mask =tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

        return self.generator(outs)

    def encode(self,src, src_mask):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)) , src_mask
        )
    
    def decode(self, tgt, memory, tgt_mask):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)) , memory , tgt_mask
        )

from torch import optim

BATCH_SIZE = 128
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(DEVICE)
print(sp.get_piece_size())

print(sp.unk_id() , sp.pad_id() , sp.bos_id(), sp.eos_id())
model = Seq2SeqTransformer(
    num_encoder_layers=3,
    num_decoder_layers=3,
    emb_size=512,
    max_len=512,
    nhead=8,
    src_vocab_size=sp.get_piece_size(),
    tgt_vocab_size=sp2.get_piece_size(),
    dim_feedforward=2048
).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX).to(DEVICE)
optimizer = optim.Adam(model.parameters())


from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def encode_safe(sp, text):
    tokens = sp.encode(text, out_type=int, add_bos=True, add_eos=True)

    #tokens = [t if t < sp.get_piece_size() else sp.unk_id() for t in tokens]
    return torch.tensor(tokens , dtype=torch.long)

text_transform = {
    SRC_LANGUAGE: lambda x: encode_safe(sp, x),
    TGT_LANGUAGE: lambda x: encode_safe(sp2, x),
}


    
def collator(batch):
    src_batch,  tgt_batch = [] , []
    for src_sample, tgt_sample in batch :
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip('\n')))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip('\n')))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

    return src_batch , tgt_batch


def generate_square_subsequent_mask(s):
    mask = (torch.triu(torch.ones((s,s) ,dtype=torch.bool , device=DEVICE)) == 1).transpose(0,1)
    mask = (
        mask.float().masked_fill(mask ==0 , float('-inf')).masked_fill(mask==1 , float(0.0))
    )
    return mask

def create_mask(src,tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len =tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len ,  src_seq_len) , device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0,1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0,1)

    return src_mask , tgt_mask , src_padding_mask , tgt_padding_mask



def greedy_decode(model , source_tensor , source_mask , max_len , start_symbol):
    source_tensor = source_tensor.to(DEVICE)
    source_mask = source_mask.to(DEVICE)

    memory = model.encode(source_tensor , source_mask)
  
    ys = torch.ones(1,1).fill_(start_symbol).type(torch.long).to(DEVICE)

    for i in range(max_len -1):
        memory =memory.to(DEVICE)
        target_mask = generate_square_subsequent_mask(ys.size(0))
        target_mask = target_mask.to(DEVICE)
        out = model.decode(ys ,memory , target_mask)
       
        out = out.transpose(0,1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat(
            [ys , torch.ones(1,1).type_as(source_tensor.data).fill_(next_word)] , dim=0
        )

        if next_word == EOS_IDX:
            break

    return ys


# def translate(model , source_sentence):
#     model.eval()
#     source_tensor = text_transform[SRC_LANGUAGE](source_sentence).view(-1,1)
#     num_tokens = source_tensor.shape[0]

#     src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
#     tgt_tokens = greedy_decode(
#         model , source_tensor , src_mask ,max_len=num_tokens +5 , start_symbol=BOS_IDX
#     ).flatten()

#     output = vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))[1:-1]
#     return " ".join(output)

def translate(model , source_sentence):
    model.eval()
    source_tensor = text_transform[SRC_LANGUAGE](source_sentence).view(-1,1)
    num_tokens = source_tensor.shape[0]

    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool).to(DEVICE)
 
    tgt_tokens = greedy_decode(
        model , source_tensor , src_mask ,max_len=num_tokens + 20 , start_symbol=BOS_IDX
    ).flatten()
   
    # SentencePiece로 디코딩
    output = sp2.decode_ids([int(x) for x in tgt_tokens.cpu().numpy()])
    return output

model.load_state_dict(torch.load("translatekoen.pth", map_location=DEVICE))
model.eval()
output_oov = translate(model , '''이제일어나서 각자의 길을 가자''')
# torch.save(model.state_dict(), "translatekoen.pth")
print('oove' ,output_oov)

