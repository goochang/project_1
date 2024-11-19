from gensim.models import Word2Vec

# 샘플 한국어 문장 데이터
sentences = [
    "나는 오늘 책을 읽었다",
    "고양이가 야옹하고 울었다",
    "인공지능은 정말 흥미로운 주제다",
    "한국어 임베딩을 학습하는 중이다"
]

# Python 기본 split() 사용해 간단하게 토큰화
tokenized_sentences = [sentence.split() for sentence in sentences]

# Word2Vec 모델 학습
word2vec_model = Word2Vec(sentences=tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4)

# 단어 '고양이'와 유사한 단어 찾기
similar_words = word2vec_model.wv.most_similar("고양이가")
print(similar_words)