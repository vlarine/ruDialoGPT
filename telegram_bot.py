#  Copyright (c) polakowo
#  Licensed under the MIT license.

# !pip install python-telegram-bot --upgrade
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import ChatAction, ParseMode
from functools import wraps
import argparse
import logging
import requests
from urllib.parse import urlencode
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import random
import re

import os, sys
import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import socket

from transformers_dev import GPT2LMHeadModel, GPT2Config
from lsp_model import RubertaTokenizer
from gpt2_training.train_utils import get_eval_list_same_length, load_model, boolean_string, fix_state_dict_namespace

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# https://github.com/python-telegram-bot/python-telegram-bot/wiki/Code-snippets

EOS_ID = 50000

def cut_seq_to_eos(sentence, remove_id=[-1]):
    sent=[]
    for s in sentence:
        if s in remove_id:
            continue
        if s != EOS_ID:
            sent.append(s)
        else:
            break
    return sent


### FROM HUGGING FACE REPO
def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits

def generate_next_token(model, input_ids, position_ids=None, token_type_ids=None, prev=None, temperature=1, top_k=0, top_p=0, past=None):
    with torch.no_grad():
        if not past:
            hidden_states, past = model.transformer(prev, position_ids, token_type_ids, past=past)
        else:
            hidden_states, past = model.transformer(prev, past=past)
        logits = model.lm_head(hidden_states)
        logits = logits[0, -1, :] / temperature
        logits = top_filtering(logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(logits.unsqueeze(0), dim=-1)
        prev = torch.multinomial(probs, num_samples=1)
        return prev, probs[0][prev], past

def generate_sequence(model, input_ids, position_ids=None, token_type_ids=None, temperature=1, top_k=0, top_p=0, length=20, past=None, device='cuda'):
    output = input_ids.new_zeros([input_ids.size(0),0])
    prev = input_ids
    for i in range(length):
        prev, probs, past = generate_next_token(model, input_ids, position_ids, token_type_ids, prev, temperature, top_k, top_p, past)
        output = torch.cat((output, prev), dim=1)
    return output

def start_command(update, context):
    context.chat_data['turns'] = []
    update.message.reply_text("Just start texting me. If I'm getting annoying, type \"Bye\"")

def requests_retry_session(
    retries=3,
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    session=None,
):
    session = session or requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    return session

def self_decorator(self, func):
    """Passes bot object to func command."""
    # TODO: Any other ways to pass variables to handlers?
    def command_func(update, context, *args, **kwargs):
        return func(self, update, context, *args, **kwargs)
    return command_func

def send_action(action):
    """Sends `action` while processing func command."""
    def decorator(func):
        @wraps(func)
        def command_func(self, update, context, *args, **kwargs):
            context.bot.send_chat_action(chat_id=update.effective_message.chat_id, action=action)
            return func(self, update, context, *args, **kwargs)
        return command_func
    return decorator

send_typing_action = send_action(ChatAction.TYPING)

@send_typing_action
def message(self, update, context):
    # Parse parameters
    #max_turns_history = self.config.getint('decoder', 'max_turns_history')
    max_turns_history = self.args.max_history
    if 'turns' not in context.chat_data:
        context.chat_data['turns'] = []
    turns = context.chat_data['turns']

    user_message = update.message.text
    if user_message.lower() in {'bye', 'пока'}:
        # Restart chat
        context.chat_data['turns'] = []
        update.message.reply_text("Пока.")
        return None
    elif user_message.lower() in {'', '/start'}:
        context.chat_data['turns'] = []
        update.message.reply_text('Привет, я ruDialogoBot.')
        return None

    if max_turns_history == 0:
        # If you still get different responses then set seed
        context.chat_data['turns'] = []
    # A single turn is a group of user messages and bot responses right after
    turn = {
        'user_messages': [],
        'bot_messages': []
    }
    turns.append(turn)
    turn['user_messages'].append(user_message)
    logger.info(f"{update.effective_message.chat_id} - User >>> {user_message}")
    logger.info(f"Turns {turns}")
    # Merge turns into a single history (don't forget EOS token)
    context_tokens = []
    from_index = max(len(turns)-max_turns_history-1, 0) if max_turns_history >= 0 else 0
    for turn in turns[from_index:]:
        # Each turn begings with user messages
        for message in turn['user_messages']:
            context_tokens += self.tokenizer.encode(message) + [EOS_ID]
        for message in turn['bot_messages']:
            context_tokens += self.tokenizer.encode(message) + [EOS_ID]

    context_tokens = torch.tensor(context_tokens, device=self.args.device, dtype=torch.long).unsqueeze(0)
    position_ids = torch.arange(0, context_tokens.size(-1), dtype=torch.long, device=context_tokens.device)

    out = generate_sequence(self.model, context_tokens, position_ids=position_ids,
                            length=self.args.generation_length, temperature=self.args.temperature,
                            top_k=self.args.top_k, top_p=self.args.top_p)

    out = out.tolist()
    bot_message = self.tokenizer.decode(cut_seq_to_eos(out[0]))

    turn['bot_messages'].append(bot_message)
    logger.info(f"{update.effective_message.chat_id} - Bot >>> {bot_message}")
    update.message.reply_text(bot_message)


def error(update, context):
    logger.warning(context.error)

class TelegramBot:
    def __init__(self, model, tokenizer, args):
        logger.info("Initializing the bot...")

        # Set global variables
        self.model = model
        self.tokenizer = tokenizer
        self.args = args

        REQUEST_KWARGS={}
        if self.args.proxy_url:
            REQUEST_KWARGS['proxy_url'] = self.args.proxy_url
            if self.args.proxy_user:
                REQUEST_KWARGS['urllib3_proxy_kwargs'] = {
                    'username': self.args.proxy_user,
                    'password': self.args.proxy_password,}

        self.updater = Updater(self.args.telegram_token, use_context=True, request_kwargs=REQUEST_KWARGS)
        dp = self.updater.dispatcher

        # on different commands - answer in Telegram
        # conversation with bot
        dp.add_handler(MessageHandler(Filters.text, self_decorator(self, message)))

        # chatbot settings
        dp.add_handler(CommandHandler('start', start_command))

        # log all errors
        dp.add_error_handler(error)

    def run_chat(self):
        logger.info("Running the chatbot...")

        # Start the Bot
        #self.updater.start_polling()
        self.updater.start_polling(timeout=500)

        # Run the bot until you press Ctrl-C or the process receives SIGINT,
        # SIGTERM or SIGABRT. This should be used most of the time, since
        # start_polling() is non-blocking and will stop the bot gracefully.
        self.updater.idle()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--telegram_token', type=str, help='Telegram token')
    parser.add_argument('--model_name_or_path', type=str, default='', help='pretrained model name or path to local checkpoint')
    parser.add_argument('--tokenizer-path', type=str, help='Path to vocabulary')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_checkpoint", '-c', type=str, default='')
    parser.add_argument("--fp16", type=boolean_string, default=False)
    parser.add_argument("--max_seq_length", type=int, default=128)

    parser.add_argument("--generation_length", type=int, default=20)
    parser.add_argument("--max_history", type=int, default=3)

    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument('--proxy_url', type=str, help='Proxy URL')
    parser.add_argument('--proxy_user', type=str, help='Proxy user')
    parser.add_argument('--proxy_password', type=str, help='Proxy password')

    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device, args.n_gpu = device, n_gpu

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #### load the GPT-2 model 
    model_config = GPT2Config.from_json_file(os.path.join(args.model_name_or_path, 'config.json'))
    tokenizer = RubertaTokenizer(vocab_file=args.tokenizer_path)
    model = load_model(GPT2LMHeadModel(model_config), args.load_checkpoint, args, verbose=True)
    model.to(device)
    model.eval()

    # Run Telegram bot
    bot = TelegramBot(model, tokenizer, args)
    bot.run_chat()

if __name__ == '__main__':
    main()
