CUDA_VISIBLE_DEVICES=1 python3 convert_megatron_to_dialogpt.py \
    --megatron_checkpoint_path ../data/model_optim_rng.pt \
    --dialogpt_output_path ../data/out.pth \
    --config-path ./configs/345M_yttm/config.json

