from Korpora import Korpora

#corpus = Korpora.load('korean_petitions')
# dataset = corpus.train
# petition = dataset[0]

# print('청원 시작일' , petition.begin)
# print('청원 종료일' , petition.end)
# print('청원 동의 수' , petition.num_agree)
# print('청원 범주 ' , petition.category)
# print('청원 제목' , petition.title)
# print('청원 본문' ,  petition.text[:30])

# corpus = Korpora.load('korean_petitions')
# petitions = corpus.get_all_texts()
# with open('corpus.txt' , 'w' , encoding='utf-8') as f :
#     for petition in petitions:
#         f.write(petition + '\n')


# from sentencepiece import SentencePieceTrainer

# SentencePieceTrainer.Train(
#     '--input=corpus.txt\
#         --model_prefix=petition_bpe\
#             --vocab_size=8000 model_type=bpe'
# )

from sentencepiece import SentencePieceProcessor
tokenizer = SentencePieceProcessor()
tokenizer.load('petition_bpe.model')

sentence = '안녕하세요. 토크나이저가 잘 학습되었군요!'
sentences = ['이렇게 입력값을 리스트로 받아서' , '쉽게 토크나이저를 사용할 수 있습니다.']

tokenized_sentence = tokenizer.encode_as_pieces(sentence)
tokenized_sentences = tokenizer.encode_as_pieces(sentences)

print(tokenized_sentence)
print(tokenized_sentences)

encoded_sentence = tokenizer.encode_as_ids(sentence)
encoded_sentences = tokenizer.encode_as_ids(sentences)

print(encoded_sentence)
print(encoded_sentences)

decode_ids = tokenizer.decode_ids(encoded_sentences)
decode_pieces = tokenizer.decode_pieces(encoded_sentences)

print(decode_ids)
print(decode_pieces)

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import Sequence, NFD ,Lowercase
from tokenizers.pre_tokenizers import Whitespace

# tokenizer = Tokenizer(WordPiece())
# tokenizer.normalizer = Sequence([NFD() , Lowercase()])
# tokenizer.pre_tokenizer = Whitespace()

# tokenizer.train(['corpus.txt'])
# tokenizer.save('petition_wordpiece.json')

from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder

tokenizer = Tokenizer.from_file('petition_wordpiece.json')
tokenizer.decoder = WordPieceDecoder()

sentence = '안녕하세요. 토크나이저가 잘 학습되었군요.'
sentences = ['이렇게 입력값을 리스트로 받아서 ' , '쉽게 토크나이저를 사용할 수 있습니다.']

encoded_sentence = tokenizer.encode(sentence)
encoded_sentences = tokenizer.encode_batch(sentences)

print(encoded_sentence.tokens)
print([enc.tokens for enc in encoded_sentences])


print(encoded_sentence.ids)
print([enc.ids for enc in encoded_sentences])


print(tokenizer.decode(encoded_sentence.ids))
#print([enc.ids for enc in encoded_sentences])


