
from gensim.models import Word2Vec
import gensim
import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt

# corpus = Korpora.load('nsmc')
# corpus = pd.DataFrame(corpus.test)

# tokenizer = Okt()
# tokens = [tokenizer.morphs(review) for review in corpus.text]


# word2vec = Word2Vec(
#     senteces=None,
#     corpus_file=None,
#     vector_size=100,
#     alpha=0.025,
#     window=5,
#     min_count=5,
#     worker=3,
#     sg=0,
#     hs=0,
#     cbow_mean=1,
#     negative=5,
#     ns_exponent=0.75,
#     max_final_vocab=None,
#     epochs=5,
#     batch_words=10000
# )

# word2vec = Word2Vec(
#     sentences=tokens,
#     vector_size=128,
#     window=5,
#     min_count=1,
#     sg=1,
#     epochs=3,
#     max_final_vocab=10000
# )

# word2vec.save('word2vec.model')
# word2vec = Word2Vec.load('word2vec.model')

# word = '연기'

# print(word2vec.wv[word])
# print(word2vec.wv.most_similar(word, topn=5))
# print(word2vec.wv.similarity(word, "연기력"))

fastText = gensim.models.FastText(
    sentences=None,
    corpus_file=None,
    vector_size=100,
    alpha=0.025,
    window=5,
    min_count=5,
    workers=3,
    sg=0,
    hs=0,
    cbow_mean=1,
    negative=5,
    ns_exponent=0.75,
    max_final_vocab=None,
    epochs=5,
    batch_words=10000,
    min_n=3,
    max_n=6
)

from Korpora import Korpora

corpus = Korpora.load('kornli')
corpus_texts = corpus.get_all_texts() + corpus.get_all_pairs()
tokens = [sentence.split() for sentence in corpus_texts]

print(tokens[:3])

from gensim.models import FastText

fastText = FastText(
    sentences=tokens,
    vector_size=128,
    window=5,
    min_count=5,
    sg=1,
    max_final_vocab=20000,
    epochs=3,
    min_n=2,
    max_n=6
)

fastText.save('fastText.model')
fastText = FastText.load('fastText.model')

oov_token = '사랑해요'
oov_vector = fastText.wv[oov_token]

print(oov_token in fastText.wv.index_to_key)
print(fastText.wv.most_similar(oov_vector,topn=5))


