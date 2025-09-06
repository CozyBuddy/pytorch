import json

# 1. JSON에 쓸 데이터 (딕셔너리)
config = {
    "architectures": ["AutoModelForSeq2SeqLM"],
    "model_type": "seq2seq",
    "vocab_size": 9000,
    "d_model": 512,
    "encoder_layers": 3,
    "decoder_layers": 3,
    "encoder_attention_heads": 8,
    "decoder_attention_heads": 8,
    "encoder_ffn_dim": 2048,
    "decoder_ffn_dim": 2048,
    "max_position_embeddings": 512,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "pad_token_id": 3
}

# 2. 파일로 저장
with open("config.json", "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=4)  # indent=4 → 보기 좋게 들여쓰기
