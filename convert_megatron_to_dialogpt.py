import argparse
import torch

from lsp_model import GPT2LMHeadModel, GPT2Config
from gpt2_training.train_utils import load_model


def convert_to_dialogpt(args):
    config = GPT2Config.from_json_file(args.config_path)
    model = load_model(GPT2LMHeadModel(config), None, args, verbose=True)

    model_state_dict = torch.load(args.megatron_checkpoint_path)

    model_state_dict = fix_state_dict_namespace(model_state_dict['model'])
    model_state_dict = fix_model_shapes(model_state_dict)

    start_model = model
    if (hasattr(model, "transformer")
        and all(not s.startswith('transformer.')
                for s in model_state_dict.keys())):
        logger.info('loading transfomer only')
        start_model = model.transformer
    start_model.load_state_dict(model_state_dict)

    torch.save(start_model.state_dict(), args.dialogpt_output_path)


def fix_state_dict_namespace(model_state_dict):
    replacements = [
        ('word_embeddings.weight', 'transformer.wte.weight'),
        ('position_embeddings.weight', 'transformer.wpe.weight'),
        ('transformer.final_layernorm.weight', 'transformer.ln_f.weight'),
        ('transformer.final_layernorm.bias', 'transformer.ln_f.bias'),
    ]

    n_Layers = 24
    for i in range(n_Layers):
        replacements.append(('transformer.layers.{}.input_layernorm.weight'.format(i), 'transformer.h.{}.ln_1.weight'.format(i)))
        replacements.append(('transformer.layers.{}.input_layernorm.bias'.format(i), 'transformer.h.{}.ln_1.bias'.format(i)))
        replacements.append(('transformer.layers.{}.post_attention_layernorm.weight'.format(i), 'transformer.h.{}.ln_2.weight'.format(i)))
        replacements.append(('transformer.layers.{}.post_attention_layernorm.bias'.format(i), 'transformer.h.{}.ln_2.bias'.format(i)))
        replacements.append(('transformer.layers.{}.attention.query_key_value.weight'.format(i), 'transformer.h.{}.attn.c_attn.weight'.format(i)))
        replacements.append(('transformer.layers.{}.attention.query_key_value.bias'.format(i), 'transformer.h.{}.attn.c_attn.bias'.format(i)))
        replacements.append(('transformer.layers.{}.attention.dense.weight'.format(i), 'transformer.h.{}.attn.c_proj.weight'.format(i)))
        replacements.append(('transformer.layers.{}.attention.dense.bias'.format(i), 'transformer.h.{}.attn.c_proj.bias'.format(i)))
        replacements.append(('transformer.layers.{}.mlp.dense_h_to_4h.weight'.format(i), 'transformer.h.{}.mlp.c_fc.weight'.format(i)))
        replacements.append(('transformer.layers.{}.mlp.dense_h_to_4h.bias'.format(i), 'transformer.h.{}.mlp.c_fc.bias'.format(i)))
        replacements.append(('transformer.layers.{}.mlp.dense_4h_to_h.weight'.format(i), 'transformer.h.{}.mlp.c_proj.weight'.format(i)))
        replacements.append(('transformer.layers.{}.mlp.dense_4h_to_h.bias'.format(i), 'transformer.h.{}.mlp.c_proj.bias'.format(i)))


    for old_key, new_key in replacements:
        model_state_dict[new_key] = model_state_dict.pop(old_key)

    return model_state_dict


def fix_model_shapes(model_state_dict):
    n_Layers = 24

    def transpose2d(model, name):
        model[name] = model.pop(name).transpose(0, 1)
        return model

    (vocab_size, n_emb) = model_state_dict['transformer.wte.weight'].shape
    for i in range(n_Layers):
        model_state_dict = transpose2d(model_state_dict, 'transformer.h.{}.attn.c_attn.weight'.format(i))
        model_state_dict = transpose2d(model_state_dict, 'transformer.h.{}.mlp.c_fc.weight'.format(i))
        model_state_dict = transpose2d(model_state_dict, 'transformer.h.{}.mlp.c_proj.weight'.format(i))

        model_state_dict['transformer.h.{}.attn.bias'.format(i)] = torch.tril(torch.ones(n_emb, n_emb)).view(1, 1, n_emb, n_emb)

    model_state_dict['lm_head.decoder.weight'] = model_state_dict['transformer.wte.weight']

    return model_state_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--megatron_checkpoint_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "The megatron checkpoint path.")
    parser.add_argument("--dialogpt_output_path",
                        default = None,
                        type = str,
                        required = True,
                        help = "Path to the output PyTorch model.")
    parser.add_argument("--config-path",
                        default = "",
                        type = str,
                        help = "An optional config json file corresponding to the model. \n"
                            "This specifies the model architecture.")
    args = parser.parse_args()
    args.n_gpu = 1
    args.device = 'cuda'
    args.fp16 = True
    convert_to_dialogpt(args)
