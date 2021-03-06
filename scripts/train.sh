CUDA_VISIBLE_DEVICES=0,1  python3 -m torch.distributed.launch --nproc_per_node=2 ./LSP_train_yttm.py \
    --tokenizer-path ./configs/345M_yttm/vocab_50000.bpe \
    --config-path ./configs/345M_yttm/config.json \
    --train_input_file ../data/train.512len.db \
    --eval_input_file ../data/valid.tsv \
    --init_checkpoint ../data/out.pth \
    --train_batch_size 64 \
    --eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --valid_step 2000 \
    --output_dir ../data/dialog \
