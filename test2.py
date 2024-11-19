from transformers import BertTokenizer, BertModel
import torch

# KLUE-BERT 모델과 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
model = BertModel.from_pretrained('klue/bert-base')

# 입력 문장
sentence = "한국어 임베딩을 학습하고 있습니다."

# 토큰화 및 텐서 변환
inputs = tokenizer(sentence, return_tensors='pt')

# 임베딩 생성
with torch.no_grad():
    outputs = model(**inputs)

# 임베딩 벡터 추출 (평균값으로 계산)
embedding = outputs.last_hidden_state.mean(dim=1)
print(embedding)