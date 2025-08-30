import nltk

#nltk.download()
text = nltk.word_tokenize('Is it possible distinguishing cats and dogs')
print(text)

#nltk.download('averaged_perceptron_tagger')

print(nltk.pos_tag(text))

#nltk.download('punkt')
string1 = 'my favorite subject is math'
string2 = 'my favorite subject is math , english , economic and computer science'

# print(nltk.word_tokenize(string1))
# print(nltk.word_tokenize(string2))

# from konlpy.tag import Komoran

# komoran = Komoran()

# print(komoran.morphs('딥러닝이 쉽나요? 어렵나요?'))

from konlpy.tag import Komoran

komoran = Komoran()
print(komoran.morphs('딥러닝이 쉽나요? 어렵나요?'))
print(komoran.pos('소파 위에 있는 것이 고양이인가요? 강아지인가요?'))

#okt = Okt()
#print(okt.morphs('딥러닝이 쉽나요? 어렵나요?'))

from nltk import sent_tokenize
text_sample = 'Natural Language Processing , or NLP , is the process of extracting the meaning , or intent , behind human language. ' \
' In the field of Conversational artificial intelligence(AI) , NLP allows machines and applications to understand the intent of human language' \
'inputs , and then generate appropriate responses ,resulting in a natural conversation flow'

tokenized_sentences = sent_tokenize(text_sample)

print(tokenized_sentences)

from nltk import word_tokenize
sentence = 'This book is for deep learning learners'
words = word_tokenize(sentence)

print(words)
