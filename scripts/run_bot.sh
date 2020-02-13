TOKEN='your_telegram_bot_token'

CHECKPOINT=../data/GP2-pretrain-step-300000.pkl
MODEL=./configs/345M_yttm/
VOCAB=./configs/345M_yttm/vocab_50000.bpe

python3 telegram_bot.py \
    --model_name_or_path $MODEL \
    --load_checkpoint $CHECKPOINT \
    --tokenizer-path $VOCAB \
    --telegram_token $TOKEN \
    --top_k 0
