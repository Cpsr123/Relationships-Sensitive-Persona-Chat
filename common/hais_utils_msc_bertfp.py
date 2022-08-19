from copy import deepcopy
from torch.utils.data import Dataset
import torch

import codecs
import itertools
import json
import random

from tqdm import tqdm
import multiprocessing as mp
import time

class Reddit(Dataset):
    '''
    PyTorch Dataset for Reddit, don't need to change this
    '''

    def __init__(self, author_dialog):
        self.authors = []
        self.author_dialog = []
        for author, author_dialog in author_dialog.items():
            self.authors.append(author)
            self.author_dialog.append(author_dialog)
        #self.author_dialog = self.author_dialog[:100]
        #self.authors = self.authors[:100]

    def __getitem__(self, idx):
        return self.author_dialog[idx]

    def __len__(self):
        return len(self.author_dialog)


def load_reddit(json_data_path, authors=None, max_slide_num=10, ctx_loop_count=5, res_loop_count=5, train_pattern=0, dev_pattern=4, test_pattern=4, response_list=None):

    train_set, valid_set, test_set = None, None, None

    if json_data_path['train']:
        train = get_json_data(authors["train"], json_data_path['train'], max_slide_num=max_slide_num, ctx_loop_count=ctx_loop_count, res_loop_count=res_loop_count, pattern=train_pattern, responses=response_list)
        train_set = Reddit(train)

    if json_data_path['dev']:
        dev = get_json_data(authors["dev"], json_data_path['dev'], max_slide_num=max_slide_num, ctx_loop_count=ctx_loop_count, res_loop_count=res_loop_count, pattern=dev_pattern, responses=response_list)
        valid_set = Reddit(dev)

    if json_data_path['test']:
        test = get_json_data(authors["test"], json_data_path['test'], max_slide_num=max_slide_num, ctx_loop_count=ctx_loop_count, res_loop_count=res_loop_count, pattern=test_pattern, responses=response_list)
        test_set = Reddit(test)

    return train_set, valid_set, test_set


def trunc_tokens(tokens, max_length, trunc_mode):
    tokens = [toks for toks in tokens if toks]
    while len(list(itertools.chain.from_iterable(tokens))) > max_length:
        if trunc_mode == 'front':
            del tokens[0][0]
            # del tokens[0]
        elif trunc_mode == 'back':
            del tokens[-1][-1]
            # del tokens[-1]
        else:
            raise Exception("`trunc_mode` must be 'front' or 'back'.")
        tokens = [toks for toks in tokens if toks]
    return tokens


def assign_ids(tokens):
    input_tokens, input_token_utter_ids, input_segment_ids = [], [], []
    _id = 0
    input_tokens.append("[CLS]")
    input_token_utter_ids.append(1)
    input_segment_ids.append(0)
    for toks in tokens:
        _id += 1
        for tok in toks:
            input_tokens.append(tok)
            input_token_utter_ids.append(_id)
            input_segment_ids.append(0)
        input_tokens.append("[EOS]")
        input_token_utter_ids.append(-_id)
        input_segment_ids.append(0)
    input_tokens.append("[SEP]")
    input_segment_ids.append(0)
    input_token_utter_ids.append(_id)
    return {"tokens": input_tokens, "segment_ids": input_segment_ids, "token_utter_ids": input_token_utter_ids}


def padding(input_dict, max_length):
    while len(input_dict["tokens"]) < max_length:
        input_dict["tokens"].append(0)
        input_dict["attention_mask"].append(0)
        input_dict["segment_ids"].append(0)
        if "lm_label_ids" in input_dict:
            input_dict["lm_label_ids"].append(-1)
    while len(input_dict["token_utter_ids"]) < max_length:
        input_dict["token_utter_ids"].append(0)
    if "author_ids" in input_dict.keys():
        while len(input_dict["author_ids"]) < max_length:
            input_dict["author_ids"].append(0)
    return input_dict

def clean_text(text):

    if not text:
        return ''

    if text.strip() == '':
        return ''

    if text and not text[-1] in ['.', '!', '?']:
        text += '. '
    text = text.replace('\\n', '')
    text = text.replace('&gt;', '>')
    text = text.replace('&lt;', '<')
    text = text.replace('&ge;', '>=')
    text = text.replace('&le;', '>=')
    text = text.replace('&ne;', '!=')
    text = text.replace('&eq;', '=')
    text = text.replace('&amp;', '&')
    text = text.replace('#x200B;', ' ')
    text = text.strip()
    return text

def random_word(tokens, tokenizer):

    output_label = []

    for i, token in enumerate(tokens):
        if token=='[eos]':
            output_label.append(-1)
            continue
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
                logger.warning("Cannot find token '{}' in vocab. Using [UNK] insetad".format(token))
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return output_label

class DTorch:
    def tensor(self, x):
        return x
    def stack(self, x):
        return torch.tensor(x)


def create_src_author_ids(input_dict, author_ids):
    author_idx = 0
    src_author_ids = []
    for token in input_dict["tokens"]:
        if token in ["[CLS]","[SEP]"]:
            src_author_ids.append(0)
        elif token == "[EOS]":
            author_idx += 1
            src_author_ids.append(0)
        else:
            src_author_ids.append(author_ids[author_idx])
    return src_author_ids


def create_res_author_ids(input_dict, author_ids):
    res_author_ids = []
    for token in input_dict["tokens"]:
        if token in ["[EOS]", "[CLS]", "[SEP]"]:
            res_author_ids.append(0)
        else:
            res_author_ids.append(author_ids)
    return res_author_ids


def convert_item(item, lead_len, history_length, max_num_context, src_length, res_length, sem_length, max_src_length, max_res_length, max_sem_length, sem_type, tokenizer, negative_response=None, use_mlm=False, pattern=0, all_author_list=None):
    torch = DTorch()
    semantic_tokens, source_tokens, response_tokens = [], [], []
    semantic_attention_mask, source_attention_mask, response_attention_mask = [], [], []
    semantic_segment_ids, source_segment_ids, response_segment_ids = [], [], []
    semantic_token_utter_ids, source_token_utter_ids, response_token_utter_ids = [], [], []
    source_lm_label_ids, response_lm_label_ids, semantic_lm_label_ids = [], [], []
    source_author_ids, response_author_ids = [], []

    # data padding
    for j in range(history_length):
        # data create
        if j == history_length -1 and negative_response:
            m = j - lead_len
            # print('m-nega:', m)
            # tokenize
            if pattern !=4:
                res_tokens = [tokenizer.tokenize(clean_text(negative_response["Response"]))]
                res_author_ids = all_author_list.index(negative_response["response_author"])
            else:
                res_tokens = [tokenizer.tokenize(clean_text(negative_response))]
                res_author_ids = random.randint(1, len(all_author_list))
            if sem_type == 0:
                sem_tokens = [tokenizer.tokenize(clean_text(item[m]['title']) + clean_text(item[m]['selftext']))]
            else:
                sem_tokens = [tokenizer.tokenize(clean_text(item[m]['title']))]
            src_tokens = [tokenizer.tokenize(clean_text(item[m][key])) for key in ['context/{}'.format(i) for i in reversed(range(max_num_context))] if key in item[m]] + [tokenizer.tokenize(clean_text(item[m]['Context']))]
            # src_tokens = [tokenizer.tokenize(item[i]['Context'])]
            src_author_ids = [all_author_list.index(item[m][key]) for key in ["author/{}".format(i) for i in reversed(range(max_num_context))] if key in item[m]] + [all_author_list.index(item[m]["context_author"])]

        elif j < lead_len:
            # print('brank')
            res_tokens = [tokenizer.tokenize('')]
            sem_tokens = [tokenizer.tokenize('')]
            src_tokens = [tokenizer.tokenize('')]
            src_author_ids , res_author_ids = None, None
        else:
            m = j - lead_len
            # print('m-posi:', m)
            # tokenize
            res_tokens = [tokenizer.tokenize(clean_text(item[m]['Response']))]
            res_author_ids = all_author_list.index(item[m]["response_author"])

            if sem_type == 0:
                sem_tokens = [tokenizer.tokenize(clean_text(item[m]['title']) + clean_text(item[m]['selftext']))]
            else:
                sem_tokens = [tokenizer.tokenize(clean_text(item[m]['title']))]
            src_tokens = [tokenizer.tokenize(clean_text(item[m][key])) for key in ['context/{}'.format(i) for i in reversed(range(max_num_context))] if key in item[m]] + [tokenizer.tokenize(clean_text(item[m]['Context']))]
            
            src_author_ids = [all_author_list.index(item[m][key]) for key in ["author/{}".format(i) for i in reversed(range(max_num_context))] if key in item[m]] + [all_author_list.index(item[m]["context_author"])] 
            # src_tokens = [tokenizer.tokenize(item[i]['Context'])]

        # truncate
        sem_tokens = sem_tokens if len(list(itertools.chain.from_iterable(sem_tokens))) < max_sem_length else trunc_tokens(sem_tokens, max_sem_length, 'back')
        src_tokens = src_tokens if len(list(itertools.chain.from_iterable(src_tokens))) < max_src_length else trunc_tokens(src_tokens, max_src_length, 'front')
        res_tokens = res_tokens if len(list(itertools.chain.from_iterable(res_tokens))) < max_res_length else trunc_tokens(res_tokens, max_res_length, 'back')
        
        # assign segment_ids, attention_mask, token_utter_ids
        sem_dict = assign_ids(sem_tokens)
        src_dict = assign_ids(src_tokens)
        res_dict = assign_ids(res_tokens)
        
        
        if pattern == 4 or pattern == 5 and j == history_length - 1:
            res_author_ids = random.randint(1, len(all_author_list))
        
        #if False:
        if j == history_length - 1 and use_mlm:
            no_tag_tokens = []
            tag_info = []
            for list_id, val in enumerate(src_dict["tokens"]):
                if val in ["[CLS]", "[EOS]", "[SEP]"]:
                    tag_info.append([list_id, val])
                else:
                    no_tag_tokens.append(val)
            lm_label_ids = random_word(no_tag_tokens, tokenizer)
            for val in tag_info:
                no_tag_tokens.insert(val[0], val[1])
                lm_label_ids.insert(val[0], -1)
            src_dict["tokens"] = no_tag_tokens
            src_dict["lm_label_ids"] = lm_label_ids
    
            no_tag_tokens = []
            tag_info = []
            for list_id, val in enumerate(res_dict["tokens"]):
                if val in ["[CLS]", "[EOS]", "[SEP]"]:
                    tag_info.append([list_id, val])
                else:
                    no_tag_tokens.append(val)
            lm_label_ids = random_word(no_tag_tokens, tokenizer)
            for val in tag_info:
                no_tag_tokens.insert(val[0], val[1])
                lm_label_ids.insert(val[0], -1)
            res_dict["tokens"] = no_tag_tokens
            res_dict["lm_label_ids"] = lm_label_ids

            sem_dict["lm_label_ids"] = [-1] * len(sem_dict["tokens"])
        else:
            sem_dict["lm_label_ids"] = [-1] * len(sem_dict["tokens"])
            src_dict["lm_label_ids"] = [-1] * len(src_dict["tokens"])
            res_dict["lm_label_ids"] = [-1] * len(res_dict["tokens"])

        
        src_dict["author_ids"] = create_src_author_ids(src_dict, src_author_ids)
        res_dict["author_ids"] = create_res_author_ids(res_dict, res_author_ids)

        # convert tokens to ids
        sem_dict["tokens"] = tokenizer.convert_tokens_to_ids(sem_dict["tokens"])
        src_dict["tokens"] = tokenizer.convert_tokens_to_ids(src_dict["tokens"])
        res_dict["tokens"] = tokenizer.convert_tokens_to_ids(res_dict["tokens"])
        # make attention_mask
        sem_dict["attention_mask"] = [1] * len(sem_dict["tokens"])
        src_dict["attention_mask"] = [1] * len(src_dict["tokens"])
        res_dict["attention_mask"] = [1] * len(res_dict["tokens"])
        
        # padding
        sem_dict = padding(sem_dict, sem_length)
        src_dict = padding(src_dict, src_length)
        res_dict = padding(res_dict, res_length)

        source_author_ids.append(src_dict["author_ids"])
        response_author_ids.append(res_dict["author_ids"])
        
        semantic_tokens.append(sem_dict["tokens"])
        source_tokens.append(src_dict["tokens"])
        response_tokens.append(res_dict["tokens"])

        semantic_attention_mask.append(sem_dict["attention_mask"])
        source_attention_mask.append(src_dict["attention_mask"])
        response_attention_mask.append(res_dict["attention_mask"])
        
        semantic_segment_ids.append(sem_dict["segment_ids"])
        source_segment_ids.append(src_dict["segment_ids"])
        response_segment_ids.append(res_dict["segment_ids"])
        
        semantic_token_utter_ids.append(sem_dict["token_utter_ids"])
        source_token_utter_ids.append(src_dict["token_utter_ids"])
        response_token_utter_ids.append(res_dict["token_utter_ids"])

        semantic_lm_label_ids.append(sem_dict["lm_label_ids"])
        source_lm_label_ids.append(src_dict["lm_label_ids"])
        response_lm_label_ids.append(res_dict["lm_label_ids"])

    aaa = time.time()
    # convert list to tensor
    
    source_author_ids = torch.tensor(source_author_ids)
    response_author_ids = torch.tensor(response_author_ids)

    semantic_tokens = torch.tensor(semantic_tokens)  # [len(item), sem_length]
    source_tokens = torch.tensor(source_tokens)  # [len(item), src_length]
    response_tokens = torch.tensor(response_tokens)  # [len(item), res_length]
    
    # print('source_tokens:', source_tokens.shape)
    
    # convert list to tensor
    semantic_attention_mask = torch.tensor(semantic_attention_mask)  # [len(item), sem_length]
    source_attention_mask = torch.tensor(source_attention_mask)  # [len(item), src_length]
    response_attention_mask = torch.tensor(response_attention_mask)  # [len(item), res_length]
    
    # convert list to tensor
    semantic_segment_ids = torch.tensor(semantic_segment_ids)  # [len(item), sem_length]
    source_segment_ids = torch.tensor(source_segment_ids)  # [len(item), src_length]
    response_segment_ids = torch.tensor(response_segment_ids)  # [len(item), res_length]
    
    # convert list to tensor
    semantic_token_utter_ids = torch.tensor(semantic_token_utter_ids)  # [len(item), sem_length]
    source_token_utter_ids = torch.tensor(source_token_utter_ids)  # [len(item), src_length]
    response_token_utter_ids = torch.tensor(response_token_utter_ids)  # [len(item), res_length]

    semantic_lm_label_ids = torch.tensor(semantic_lm_label_ids)
    source_lm_label_ids = torch.tensor(source_lm_label_ids)
    response_lm_label_ids = torch.tensor(response_lm_label_ids)

    return source_tokens, response_tokens, semantic_tokens, \
        source_attention_mask, response_attention_mask, semantic_attention_mask, \
        source_segment_ids, response_segment_ids, semantic_segment_ids, \
        source_token_utter_ids, response_token_utter_ids, semantic_token_utter_ids,\
        source_lm_label_ids, response_lm_label_ids, semantic_lm_label_ids, \
        source_author_ids, response_author_ids, time.time()-aaa




def convert_res_his(item, lead_len, history_length, max_num_context, src_length, res_length, sem_length, max_src_length, max_res_length, max_sem_length, sem_type, tokenizer, pattern=0, all_author_list=None):
    torch = DTorch()
    semantic_tokens, source_tokens, response_tokens = [], [], []
    semantic_attention_mask, source_attention_mask, response_attention_mask = [], [], []
    semantic_segment_ids, source_segment_ids, response_segment_ids = [], [], []
    semantic_token_utter_ids, source_token_utter_ids, response_token_utter_ids = [], [], []
    source_author_ids, response_author_ids = [], []
    
    # data padding
    for j in range(history_length - 1):
        # data create
        if j < lead_len:
            # print('brank')
            res_tokens = [tokenizer.tokenize('')]
            sem_tokens = [tokenizer.tokenize('')]
            src_tokens = [tokenizer.tokenize('')]
            src_author_ids, res_author_ids = None, None
        else:
            m = j - lead_len
            # print('m-posi:', m)
            # tokenize
            res_tokens = [tokenizer.tokenize(clean_text(item[m]['Context']))]
            res_author_ids = all_author_list.index(item[m]["context_author"])
            if sem_type == 0:
                sem_tokens = [tokenizer.tokenize(clean_text(item[m]['title']) + clean_text(item[m]['selftext']))]
            else:
                sem_tokens = [tokenizer.tokenize(clean_text(item[m]['title']))]
            src_tokens = [tokenizer.tokenize(clean_text(item[m][key])) for key in ['context/{}'.format(i) for i in reversed(range(max_num_context))] if key in item[m]]
            # src_tokens = [tokenizer.tokenize(item[i]['Context'])]
            src_author_ids = [all_author_list.index(item[m][key]) for key in ["author/{}".format(i) for i in reversed(range(max_num_context))] if key in item[m]] 

        # truncate
        sem_tokens = sem_tokens if len(list(itertools.chain.from_iterable(sem_tokens))) < max_sem_length else trunc_tokens(sem_tokens, max_sem_length, 'back')
        src_tokens = src_tokens if len(list(itertools.chain.from_iterable(src_tokens))) < max_src_length else trunc_tokens(src_tokens, max_src_length, 'front')
        res_tokens = res_tokens if len(list(itertools.chain.from_iterable(res_tokens))) < max_res_length else trunc_tokens(res_tokens, max_res_length, 'back')

        # src_tokens = src_tokens + res_tokens[-1:]

        if pattern == 4 or pattern == 5 and j == history_length - 1:
            res_author_ids = random.randint(1, len(all_author_list))
        
        # assign segment_ids, attention_mask, token_utter_ids
        sem_dict = assign_ids(sem_tokens)
        src_dict = assign_ids(src_tokens)
        res_dict = assign_ids(res_tokens)
        
        src_dict["author_ids"] = create_src_author_ids(src_dict, src_author_ids)
        res_dict["author_ids"] = create_res_author_ids(res_dict, res_author_ids)

        # convert tokens to ids
        sem_dict["tokens"] = tokenizer.convert_tokens_to_ids(sem_dict["tokens"])
        src_dict["tokens"] = tokenizer.convert_tokens_to_ids(src_dict["tokens"])
        res_dict["tokens"] = tokenizer.convert_tokens_to_ids(res_dict["tokens"])
        
        # make attention_mask
        sem_dict["attention_mask"] = [1] * len(sem_dict["tokens"])
        src_dict["attention_mask"] = [1] * len(src_dict["tokens"])
        res_dict["attention_mask"] = [1] * len(res_dict["tokens"])
        
        # padding
        sem_dict = padding(sem_dict, sem_length)
        src_dict = padding(src_dict, src_length)
        res_dict = padding(res_dict, res_length)

        
        source_author_ids.append(src_dict["author_ids"])
        response_author_ids.append(res_dict["author_ids"])

        semantic_tokens.append(sem_dict["tokens"])
        source_tokens.append(src_dict["tokens"])
        response_tokens.append(res_dict["tokens"])
        
        semantic_attention_mask.append(sem_dict["attention_mask"])
        source_attention_mask.append(src_dict["attention_mask"])
        response_attention_mask.append(res_dict["attention_mask"])
        
        semantic_segment_ids.append(sem_dict["segment_ids"])
        source_segment_ids.append(src_dict["segment_ids"])
        response_segment_ids.append(res_dict["segment_ids"])
        
        semantic_token_utter_ids.append(sem_dict["token_utter_ids"])
        source_token_utter_ids.append(src_dict["token_utter_ids"])
        response_token_utter_ids.append(res_dict["token_utter_ids"])

    aaa = time.time()
    # convert list to tensor

    source_author_ids = torch.tensor(source_author_ids)
    response_author_ids = torch.tensor(response_author_ids)

    semantic_tokens = torch.tensor(semantic_tokens)  # [len(item), sem_length]
    source_tokens = torch.tensor(source_tokens)  # [len(item), src_length]
    response_tokens = torch.tensor(response_tokens)  # [len(item), res_length]

    # print('source_tokens:', source_tokens.shape)
    
    # convert list to tensor
    semantic_attention_mask = torch.tensor(semantic_attention_mask)  # [len(item), sem_length]
    source_attention_mask = torch.tensor(source_attention_mask)  # [len(item), src_length]
    response_attention_mask = torch.tensor(response_attention_mask)  # [len(item), res_length]
    
    # convert list to tensor
    semantic_segment_ids = torch.tensor(semantic_segment_ids)  # [len(item), sem_length]
    source_segment_ids = torch.tensor(source_segment_ids)  # [len(item), src_length]
    response_segment_ids = torch.tensor(response_segment_ids)  # [len(item), res_length]
    
    # convert list to tensor
    semantic_token_utter_ids = torch.tensor(semantic_token_utter_ids)  # [len(item), sem_length]
    source_token_utter_ids = torch.tensor(source_token_utter_ids)  # [len(item), src_length]
    response_token_utter_ids = torch.tensor(response_token_utter_ids)  # [len(item), res_length]

    return source_tokens, response_tokens, semantic_tokens, \
        source_attention_mask, response_attention_mask, semantic_attention_mask, \
        source_segment_ids, response_segment_ids, semantic_segment_ids, \
        source_token_utter_ids, response_token_utter_ids, semantic_token_utter_ids, \
        source_author_ids, response_author_ids, time.time()-aaa

def select_negative_response(positive_response, candidate):

    if len(candidate) == 0:
        raise Exception('no candidates')
    idx = random.choice(range(len(candidate)))
    negative_response = candidate.pop(idx)

    if negative_response["Response"] == positive_response:
        negative_response = select_negative_response(positive_response, candidate)
    
    return negative_response

def group_worker(arg):
    res = []
    queue = arg[0]
    for a in arg[1]:
        res.append(worker(a))
        queue.put(1)
    return res

def worker(proc_arg):

    time = 0
    semantic_tokens_list, source_tokens_list, response_tokens_list = [], [], []
    semantic_attention_mask_list, source_attention_mask_list, response_attention_mask_list = [], [], []
    semantic_segment_ids_list, source_segment_ids_list, response_segment_ids_list = [], [], []
    semantic_token_utter_ids_list, source_token_utter_ids_list, response_token_utter_ids_list = [], [], []
    res_his_tokens_list, res_his_attention_mask_list, res_his_segment_ids_list, res_his_token_utter_ids_list = [], [], [], []
    res_his_res_tokens_list, res_his_res_attention_mask_list, res_his_res_segment_ids_list, res_his_res_token_utter_ids_list = [], [], [], []
    res_his_sem_tokens_list, res_his_sem_attention_mask_list, res_his_sem_segment_ids_list, res_his_sem_token_utter_ids_list = [], [], [], []
    label_list = []
    source_lm_label_ids_list, response_lm_label_ids_list, semantic_lm_label_ids_list = [], [], []
    
    source_author_ids_list, response_author_ids_list = [], []
    res_his_author_ids_list, res_his_res_author_ids_list = [], [] 

    item, responses, author, ctx_num_history, res_num_history, max_num_context, src_length, res_length, sem_length, \
        max_src_length, max_res_length, max_sem_length, tokenizer, pattern, sem_type, use_mlm, all_author_list = proc_arg



    ctx_item = item.get('ctx_histories')
    his_item = item.get('res_histories')
    ctx_l = len(ctx_item)
    if ctx_num_history > ctx_l:
        ctx_sub_item = ctx_item
        ctx_lead_len = ctx_num_history - ctx_l
    else:
        ctx_sub_item = ctx_item[ctx_l-ctx_num_history:]
        ctx_lead_len = 0
    
    res_l = len(his_item)
    if res_num_history > res_l + 1:
        his_sub_item = his_item
        his_lead_len = res_num_history - res_l - 1
    else:
        his_sub_item = his_item[res_l-res_num_history+1:]
        his_lead_len = 0

    if pattern == 4:

        candidate = responses.get(author)
        if not candidate:
            return None

        for val in candidate:

            label = val["label"]
            negative_response = None
            if label == 0:
                negative_response = val["text"]
            # print('sub_item:', len(sub_item))
            # print('lead_len:', lead_len)
            source_tokens, response_tokens, semantic_tokens, \
                source_attention_mask, response_attention_mask, semantic_attention_mask, \
                source_segment_ids, response_segment_ids, semantic_segment_ids, \
                source_token_utter_ids, response_token_utter_ids, semantic_token_utter_ids,\
                source_lm_label_ids, response_lm_label_ids, semantic_lm_label_ids, source_author_ids, response_author_ids, t \
                    = convert_item(ctx_sub_item, ctx_lead_len, ctx_num_history, max_num_context, src_length, res_length, sem_length, max_src_length, max_res_length, max_sem_length, sem_type, tokenizer, negative_response, pattern=pattern, all_author_list=all_author_list)

            res_his_tokens, res_his_res_tokens, res_his_sem_tokens, \
                res_his_attention_mask, res_his_res_attention_mask, res_his_sem_attention_mask, \
                res_his_segment_ids, res_his_res_segment_ids, res_his_sem_segment_ids, \
                res_his_token_utter_ids, res_his_res_token_utter_ids, res_his_sem_token_utter_ids, res_his_author_ids, res_his_res_author_ids, t2 \
                    = convert_res_his(his_sub_item, his_lead_len, res_num_history, max_num_context, src_length, res_length, sem_length, max_src_length, max_res_length, max_sem_length, sem_type, tokenizer, pattern=pattern, all_author_list=all_author_list)

            time += t+t2
            

            source_author_ids_list.append(source_author_ids)
            response_author_ids_list.append(response_author_ids)
            res_his_author_ids_list.append(res_his_author_ids)
            res_his_res_author_ids_list.append(res_his_res_author_ids)

            # semantic_tokens_list.append(semantic_tokens)
            source_tokens_list.append(source_tokens)
            response_tokens_list.append(response_tokens)
            semantic_tokens_list.append(semantic_tokens)
            res_his_tokens_list.append(res_his_tokens)
            res_his_res_tokens_list.append(res_his_res_tokens)
            res_his_sem_tokens_list.append(res_his_sem_tokens)
            
            # semantic_attention_mask_list.append(semantic_attention_mask)
            source_attention_mask_list.append(source_attention_mask)
            response_attention_mask_list.append(response_attention_mask)
            semantic_attention_mask_list.append(semantic_attention_mask)
            res_his_attention_mask_list.append(res_his_attention_mask)
            res_his_res_attention_mask_list.append(res_his_res_attention_mask)
            res_his_sem_attention_mask_list.append(res_his_sem_attention_mask)
            
            # semantic_segment_ids_list.append(semantic_segment_ids)
            source_segment_ids_list.append(source_segment_ids)
            response_segment_ids_list.append(response_segment_ids)
            semantic_segment_ids_list.append(semantic_segment_ids)
            res_his_segment_ids_list.append(res_his_segment_ids)
            res_his_res_segment_ids_list.append(res_his_res_segment_ids)
            res_his_sem_segment_ids_list.append(res_his_sem_segment_ids)
            
            # semantic_token_utter_ids_list.append(semantic_token_utter_ids)
            source_token_utter_ids_list.append(source_token_utter_ids)
            response_token_utter_ids_list.append(response_token_utter_ids)
            semantic_token_utter_ids_list.append(semantic_token_utter_ids)
            res_his_token_utter_ids_list.append(res_his_token_utter_ids)
            res_his_res_token_utter_ids_list.append(res_his_res_token_utter_ids)
            res_his_sem_token_utter_ids_list.append(res_his_sem_token_utter_ids)

            source_lm_label_ids_list.append(source_lm_label_ids)
            response_lm_label_ids_list.append(response_lm_label_ids)
            semantic_lm_label_ids_list.append(semantic_lm_label_ids)

            label_list.append(label)

    elif pattern == 5:

        label = int(item.get('label'))
        source_tokens, response_tokens, semantic_tokens, \
            source_attention_mask, response_attention_mask, semantic_attention_mask, \
            source_segment_ids, response_segment_ids, semantic_segment_ids, \
            source_token_utter_ids, response_token_utter_ids, semantic_token_utter_ids,\
            source_lm_label_ids, response_lm_label_ids, semantic_lm_label_ids, source_author_ids, response_author_ids, t \
                = convert_item(ctx_sub_item, ctx_lead_len, ctx_num_history, max_num_context, src_length, res_length, sem_length, max_src_length, max_res_length, max_sem_length, sem_type, tokenizer, pattern=pattern, all_author_list=all_author_list)
        
        res_his_tokens, res_his_res_tokens, res_his_sem_tokens, \
            res_his_attention_mask, res_his_res_attention_mask, res_his_sem_attention_mask, \
            res_his_segment_ids, res_his_res_segment_ids, res_his_sem_segment_ids, \
            res_his_token_utter_ids, res_his_res_token_utter_ids, res_his_sem_token_utter_ids, res_his_author_ids, res_his_res_author_ids, t2 \
                = convert_res_his(his_sub_item, his_lead_len, res_num_history, max_num_context, src_length, res_length, sem_length, max_src_length, max_res_length, max_sem_length, sem_type, tokenizer, pattern=pattern, all_author_list=all_author_list)

        time += t+t2
        
        source_author_ids_list.append(source_author_ids)
        response_author_ids_list.append(response_author_ids)
        res_his_author_ids_list.append(res_his_author_ids)
        res_his_res_author_ids_list.append(res_his_res_author_ids)

        # semantic_tokens_list.append(semantic_tokens)
        source_tokens_list.append(source_tokens)
        response_tokens_list.append(response_tokens)
        semantic_tokens_list.append(semantic_tokens)
        res_his_tokens_list.append(res_his_tokens)
        res_his_res_tokens_list.append(res_his_res_tokens)
        res_his_sem_tokens_list.append(res_his_sem_tokens)

        # semantic_attention_mask_list.append(semantic_attention_mask)
        source_attention_mask_list.append(source_attention_mask)
        response_attention_mask_list.append(response_attention_mask)
        semantic_attention_mask_list.append(semantic_attention_mask)
        res_his_attention_mask_list.append(res_his_attention_mask)
        res_his_res_attention_mask_list.append(res_his_res_attention_mask)
        res_his_sem_attention_mask_list.append(res_his_sem_attention_mask)

        # semantic_segment_ids_list.append(semantic_segment_ids)
        source_segment_ids_list.append(source_segment_ids)
        response_segment_ids_list.append(response_segment_ids)
        semantic_segment_ids_list.append(semantic_segment_ids)
        res_his_segment_ids_list.append(res_his_segment_ids)
        res_his_res_segment_ids_list.append(res_his_res_segment_ids)
        res_his_sem_segment_ids_list.append(res_his_sem_segment_ids)

        # semantic_token_utter_ids_list.append(semantic_token_utter_ids)
        source_token_utter_ids_list.append(source_token_utter_ids)
        response_token_utter_ids_list.append(response_token_utter_ids)
        semantic_token_utter_ids_list.append(semantic_token_utter_ids)
        res_his_token_utter_ids_list.append(res_his_token_utter_ids)
        res_his_res_token_utter_ids_list.append(res_his_res_token_utter_ids)
        res_his_sem_token_utter_ids_list.append(res_his_sem_token_utter_ids)

        source_lm_label_ids_list.append(source_lm_label_ids)
        response_lm_label_ids_list.append(response_lm_label_ids)
        semantic_lm_label_ids_list.append(semantic_lm_label_ids)

        label_list.append(label)

    else:
        label = 1
        negative_response = None
        

        # print('sub_item:', len(sub_item))
        # print('lead_len:', lead_len)
        source_tokens, response_tokens, semantic_tokens, \
            source_attention_mask, response_attention_mask, semantic_attention_mask, \
            source_segment_ids, response_segment_ids, semantic_segment_ids, \
            source_token_utter_ids, response_token_utter_ids, semantic_token_utter_ids,\
            source_lm_label_ids, response_lm_label_ids, semantic_lm_label_ids, source_author_ids, response_author_ids, t \
                = convert_item(ctx_sub_item, ctx_lead_len, ctx_num_history, max_num_context, src_length, res_length, sem_length, max_src_length, max_res_length, max_sem_length, sem_type, tokenizer, negative_response, use_mlm, pattern=pattern, all_author_list=all_author_list)
        
        res_his_tokens, res_his_res_tokens, res_his_sem_tokens, \
            res_his_attention_mask, res_his_res_attention_mask, res_his_sem_attention_mask, \
            res_his_segment_ids, res_his_res_segment_ids, res_his_sem_segment_ids, \
            res_his_token_utter_ids, res_his_res_token_utter_ids, res_his_sem_token_utter_ids, res_his_author_ids, res_his_res_author_ids, t2 \
                = convert_res_his(his_sub_item, his_lead_len, res_num_history, max_num_context, src_length, res_length, sem_length, max_src_length, max_res_length, max_sem_length, sem_type, tokenizer, pattern=pattern, all_author_list=all_author_list)
        
        time += t+t2
        source_author_ids_list.append(source_author_ids)
        response_author_ids_list.append(response_author_ids)
        res_his_author_ids_list.append(res_his_author_ids)
        res_his_res_author_ids_list.append(res_his_res_author_ids)

        # semantic_tokens_list.append(semantic_tokens)
        source_tokens_list.append(source_tokens)
        response_tokens_list.append(response_tokens)
        semantic_tokens_list.append(semantic_tokens)
        res_his_tokens_list.append(res_his_tokens)
        res_his_res_tokens_list.append(res_his_res_tokens)
        res_his_sem_tokens_list.append(res_his_sem_tokens)
        
        # semantic_attention_mask_list.append(semantic_attention_mask)
        source_attention_mask_list.append(source_attention_mask)
        response_attention_mask_list.append(response_attention_mask)
        semantic_attention_mask_list.append(semantic_attention_mask)
        res_his_attention_mask_list.append(res_his_attention_mask)
        res_his_res_attention_mask_list.append(res_his_res_attention_mask)
        res_his_sem_attention_mask_list.append(res_his_sem_attention_mask)
        
        # semantic_segment_ids_list.append(semantic_segment_ids)
        source_segment_ids_list.append(source_segment_ids)
        response_segment_ids_list.append(response_segment_ids)
        semantic_segment_ids_list.append(semantic_segment_ids)
        res_his_segment_ids_list.append(res_his_segment_ids)
        res_his_res_segment_ids_list.append(res_his_res_segment_ids)
        res_his_sem_segment_ids_list.append(res_his_sem_segment_ids)
        
        # semantic_token_utter_ids_list.append(semantic_token_utter_ids)
        source_token_utter_ids_list.append(source_token_utter_ids)
        response_token_utter_ids_list.append(response_token_utter_ids)
        semantic_token_utter_ids_list.append(semantic_token_utter_ids)
        res_his_token_utter_ids_list.append(res_his_token_utter_ids)
        res_his_res_token_utter_ids_list.append(res_his_res_token_utter_ids)
        res_his_sem_token_utter_ids_list.append(res_his_sem_token_utter_ids)

        source_lm_label_ids_list.append(source_lm_label_ids)
        response_lm_label_ids_list.append(response_lm_label_ids)
        semantic_lm_label_ids_list.append(semantic_lm_label_ids)

        label_list.append(label)

        label = 0
        positive_response = ctx_sub_item[-1]["Response"]
        candidate = deepcopy(responses)
        negative_response = select_negative_response(positive_response, candidate)

        source_tokens, response_tokens, semantic_tokens, \
            source_attention_mask, response_attention_mask, semantic_attention_mask, \
            source_segment_ids, response_segment_ids, semantic_segment_ids, \
            source_token_utter_ids, response_token_utter_ids, semantic_token_utter_ids,\
            source_lm_label_ids, response_lm_label_ids, semantic_lm_label_ids, source_author_ids, response_author_ids, t \
                = convert_item(ctx_sub_item, ctx_lead_len, ctx_num_history, max_num_context, src_length, res_length, sem_length, max_src_length, max_res_length, max_sem_length, sem_type, tokenizer, negative_response, use_mlm, pattern=pattern, all_author_list=all_author_list)
        
        res_his_tokens, res_his_res_tokens, res_his_sem_tokens, \
            res_his_attention_mask, res_his_res_attention_mask, res_his_sem_attention_mask, \
            res_his_segment_ids, res_his_res_segment_ids, res_his_sem_segment_ids, \
            res_his_token_utter_ids, res_his_res_token_utter_ids, res_his_sem_token_utter_ids, res_his_author_ids, res_his_res_author_ids, t2 \
                = convert_res_his(his_sub_item, his_lead_len, res_num_history, max_num_context, src_length, res_length, sem_length, max_src_length, max_res_length, max_sem_length, sem_type, tokenizer, pattern=pattern, all_author_list=all_author_list)

        time += t+t2
        source_author_ids_list.append(source_author_ids)
        response_author_ids_list.append(response_author_ids)
        res_his_author_ids_list.append(res_his_author_ids)
        res_his_res_author_ids_list.append(res_his_res_author_ids)

        # semantic_tokens_list.append(semantic_tokens)
        source_tokens_list.append(source_tokens)
        response_tokens_list.append(response_tokens)
        semantic_tokens_list.append(semantic_tokens)
        res_his_tokens_list.append(res_his_tokens)
        res_his_res_tokens_list.append(res_his_res_tokens)
        res_his_sem_tokens_list.append(res_his_sem_tokens)
        
        # semantic_attention_mask_list.append(semantic_attention_mask)
        source_attention_mask_list.append(source_attention_mask)
        response_attention_mask_list.append(response_attention_mask)
        semantic_attention_mask_list.append(semantic_attention_mask)
        res_his_attention_mask_list.append(res_his_attention_mask)
        res_his_res_attention_mask_list.append(res_his_res_attention_mask)
        res_his_sem_attention_mask_list.append(res_his_sem_attention_mask)
        
        # semantic_segment_ids_list.append(semantic_segment_ids)
        source_segment_ids_list.append(source_segment_ids)
        response_segment_ids_list.append(response_segment_ids)
        semantic_segment_ids_list.append(semantic_segment_ids)
        res_his_segment_ids_list.append(res_his_segment_ids)
        res_his_res_segment_ids_list.append(res_his_res_segment_ids)
        res_his_sem_segment_ids_list.append(res_his_sem_segment_ids)
        
        # semantic_token_utter_ids_list.append(semantic_token_utter_ids)
        source_token_utter_ids_list.append(source_token_utter_ids)
        response_token_utter_ids_list.append(response_token_utter_ids)
        semantic_token_utter_ids_list.append(semantic_token_utter_ids)
        res_his_token_utter_ids_list.append(res_his_token_utter_ids)
        res_his_res_token_utter_ids_list.append(res_his_res_token_utter_ids)
        res_his_sem_token_utter_ids_list.append(res_his_sem_token_utter_ids)

        source_lm_label_ids_list.append(source_lm_label_ids)
        response_lm_label_ids_list.append(response_lm_label_ids)
        semantic_lm_label_ids_list.append(semantic_lm_label_ids)

        label_list.append(label)

    return semantic_tokens_list, source_tokens_list, response_tokens_list, \
            semantic_attention_mask_list, source_attention_mask_list, response_attention_mask_list, \
            semantic_segment_ids_list, source_segment_ids_list, response_segment_ids_list, \
            semantic_token_utter_ids_list, source_token_utter_ids_list, response_token_utter_ids_list, \
            res_his_tokens_list, res_his_attention_mask_list, res_his_segment_ids_list, res_his_token_utter_ids_list, \
            res_his_res_tokens_list, res_his_res_attention_mask_list, res_his_res_segment_ids_list, res_his_res_token_utter_ids_list, \
            res_his_sem_tokens_list, res_his_sem_attention_mask_list, res_his_sem_segment_ids_list, res_his_sem_token_utter_ids_list, \
            label_list, source_lm_label_ids_list, response_lm_label_ids_list, semantic_lm_label_ids_list,\
            source_author_ids_list, response_author_ids_list, res_his_author_ids_list, res_his_res_author_ids_list            


def pbar_listen(total, desc, queue):
    pbar = None
    for item in iter(queue.get, None):
        if pbar is None:
            pbar = tqdm(total=total, desc=desc)
        pbar.update()

def convert_reddit_to_features(reddit: Reddit, src_length, res_length, sem_length, tokenizer, ctx_loop_count=3, res_loop_count=3, pattern=0, sem_type=0, responses=None, use_mlm=False, cpu_process_num=6, all_author_list=None):
    semantic_tokens_list, source_tokens_list, response_tokens_list = [], [], []
    semantic_attention_mask_list, source_attention_mask_list, response_attention_mask_list = [], [], []
    semantic_segment_ids_list, source_segment_ids_list, response_segment_ids_list = [], [], []
    semantic_token_utter_ids_list, source_token_utter_ids_list, response_token_utter_ids_list = [], [], []
    res_his_tokens_list, res_his_attention_mask_list, res_his_segment_ids_list, res_his_token_utter_ids_list = [], [], [], []
    res_his_res_tokens_list, res_his_res_attention_mask_list, res_his_res_segment_ids_list, res_his_res_token_utter_ids_list = [], [], [], []
    res_his_sem_tokens_list, res_his_sem_attention_mask_list, res_his_sem_segment_ids_list, res_his_sem_token_utter_ids_list = [], [], [], []
    label_list = []
    source_lm_label_ids_list, response_lm_label_ids_list, semantic_lm_label_ids_list = [], [], []

    source_author_ids_list, response_author_ids_list, res_his_author_ids_list, res_his_res_author_ids_list = [], [], [], []

    # maximum number of contexts
    max_num_context = 10 + 1

    # Account for [CLS], [SEP], Utterance Token
    max_src_length = src_length - 2 - max_num_context
    max_res_length = res_length - 2 - 1
    max_sem_length = sem_length - 2 - 1

    if pattern == 4 and not responses:
        pattern == 0

    if pattern != 4 or not responses:
        responses = []
        for idx, key in enumerate(reddit):
            items = reddit.author_dialog[idx]
            ctx_histories = items.get('ctx_histories')
            for item in ctx_histories:
                response = {"Response":item.get("Response"), "response_author":item.get("response_author")}
                if response:
                    responses.append(response)

    start = time.time()
    #process_num = mp.cpu_count()
    process_num = cpu_process_num

    print(f'Multi Process start. Process num = {process_num}.')
    proc_args = []
    for idx, item in enumerate(tqdm(reddit, desc="data load")):
        author = reddit.authors[idx]
        arg = (item, responses, author, ctx_loop_count, res_loop_count, max_num_context, src_length, res_length, sem_length, max_src_length, max_res_length, max_sem_length, tokenizer, pattern, sem_type, use_mlm, all_author_list)
        proc_args.append(arg)

    group_size = len(proc_args) // process_num + 1
    queue = mp.Manager().Queue()
    group_proc_args = [
        [queue, proc_args[i:i+group_size]]
        for i in range(0, len(proc_args), group_size)
    ]
    listener = mp.Process(target=pbar_listen, args=(len(proc_args), 'parallel convert', queue))
    listener.start()

    print(f'convert_start: {time.time()}')
    results = None
    if True: # multiprocess or not
        p = mp.Pool(process_num)
        results = [y for x in p.map(group_worker, group_proc_args) for y in x]
        # results = list(tqdm(p.imap_unordered(worker, proc_args), total=len(proc_args), desc='parallel convert'))
        # p.close()
        # p.join()
    else:
        results = []
        for arg in tqdm(proc_args):
            results.append(worker(arg))
    queue.put(None)
    listener.join()
    print(f'convert_end: {time.time()}')
    for result in tqdm(results, desc="results data"):

        if not result:
            continue

        semantic_tokens, source_tokens, response_tokens, \
        semantic_attention_mask, source_attention_mask, response_attention_mask, \
        semantic_segment_ids, source_segment_ids, response_segment_ids, \
        semantic_token_utter_ids, source_token_utter_ids, response_token_utter_ids, \
        res_his_tokens, res_his_attention_mask, res_his_segment_ids, res_his_token_utter_ids, \
        res_his_res_tokens, res_his_res_attention_mask, res_his_res_segment_ids, res_his_res_token_utter_ids, \
        res_his_sem_tokens, res_his_sem_attention_mask, res_his_sem_segment_ids, res_his_sem_token_utter_ids, \
        label, source_lm_label_ids, response_lm_label_ids, semantic_lm_label_ids,\
        source_author_ids, response_author_ids, res_his_author_ids, res_his_res_author_ids = result

        semantic_tokens_list.extend(semantic_tokens)
        source_tokens_list.extend(source_tokens)
        response_tokens_list.extend(response_tokens)
        semantic_attention_mask_list.extend(semantic_attention_mask)
        source_attention_mask_list.extend(source_attention_mask)
        response_attention_mask_list.extend(response_attention_mask)
        semantic_segment_ids_list.extend(semantic_segment_ids)
        source_segment_ids_list.extend(source_segment_ids)
        response_segment_ids_list.extend(response_segment_ids)
        semantic_token_utter_ids_list.extend(semantic_token_utter_ids)
        source_token_utter_ids_list.extend(source_token_utter_ids)
        response_token_utter_ids_list.extend(response_token_utter_ids)
        res_his_tokens_list.extend(res_his_tokens)
        res_his_attention_mask_list.extend(res_his_attention_mask)
        res_his_segment_ids_list.extend(res_his_segment_ids)
        res_his_token_utter_ids_list.extend(res_his_token_utter_ids)
        res_his_res_tokens_list.extend(res_his_res_tokens)
        res_his_res_attention_mask_list.extend(res_his_res_attention_mask)
        res_his_res_segment_ids_list.extend(res_his_res_segment_ids)
        res_his_res_token_utter_ids_list.extend(res_his_res_token_utter_ids)
        res_his_sem_tokens_list.extend(res_his_sem_tokens)
        res_his_sem_attention_mask_list.extend(res_his_sem_attention_mask)
        res_his_sem_segment_ids_list.extend(res_his_sem_segment_ids)
        res_his_sem_token_utter_ids_list.extend(res_his_sem_token_utter_ids)
        label_list.extend(label)
        source_lm_label_ids_list.extend(source_lm_label_ids)
        response_lm_label_ids_list.extend(response_lm_label_ids)
        semantic_lm_label_ids_list.extend(semantic_lm_label_ids)

        source_author_ids_list.extend(source_author_ids)
        response_author_ids_list.extend(response_author_ids)
        res_his_author_ids_list.extend(res_his_author_ids)
        res_his_res_author_ids_list.extend(res_his_res_author_ids)

    end = time.time()
    print(f'Load Time = {(end-start):.3f}')
    if len(source_tokens_list) == 0:
        raise Exception('Created count 0.')
    dtorch = DTorch()
    semantic_tokens_batch = dtorch.stack(semantic_tokens_list)  # [batch_size, len(item), sem_length]
    source_tokens_batch = dtorch.stack(source_tokens_list)  # [batch_size, len(item), src_length]
    response_tokens_batch = dtorch.stack(response_tokens_list)  # [batch_size, len(item), res_length]
    res_his_tokens_batch = dtorch.stack(res_his_tokens_list)
    res_his_res_tokens_batch = dtorch.stack(res_his_res_tokens_list)
    res_his_sem_tokens_batch = dtorch.stack(res_his_sem_tokens_list)
    
    # stacking attention_mask
    semantic_attention_mask_batch = dtorch.stack(semantic_attention_mask_list)  # [batch_size, len(item), sem_length]
    source_attention_mask_batch = dtorch.stack(source_attention_mask_list)  # [batch_size, len(item), src_length]
    response_attention_mask_batch = dtorch.stack(response_attention_mask_list)  # [batch_size, len(item), res_length]
    res_his_attention_mask_batch = dtorch.stack(res_his_attention_mask_list)
    res_his_res_attention_mask_batch = dtorch.stack(res_his_res_attention_mask_list)
    res_his_sem_attention_mask_batch = dtorch.stack(res_his_sem_attention_mask_list)
    
    # stacking segment_ids
    semantic_segment_ids_batch = dtorch.stack(semantic_segment_ids_list)  # [batch_size, len(item), sem_length]
    source_segment_ids_batch = dtorch.stack(source_segment_ids_list)  # [batch_size, len(item), src_length]
    response_segment_ids_batch = dtorch.stack(response_segment_ids_list)  # [batch_size, len(item), res_length]
    res_his_segment_ids_batch = dtorch.stack(res_his_segment_ids_list)
    res_his_res_segment_ids_batch = dtorch.stack(res_his_res_segment_ids_list)
    res_his_sem_segment_ids_batch = dtorch.stack(res_his_sem_segment_ids_list)
    
    # stacking token_utter_ids
    semantic_token_utter_ids_batch = dtorch.stack(semantic_token_utter_ids_list)  # [batch_size, len(item), sem_length]
    source_token_utter_ids_batch = dtorch.stack(source_token_utter_ids_list)  # [batch_size, len(item), src_length]
    response_token_utter_ids_batch = dtorch.stack(response_token_utter_ids_list)  # [batch_size, len(item), res_length]
    res_his_token_utter_ids_batch = dtorch.stack(res_his_token_utter_ids_list)
    res_his_res_token_utter_ids_batch = dtorch.stack(res_his_res_token_utter_ids_list)
    res_his_sem_token_utter_ids_batch = dtorch.stack(res_his_sem_token_utter_ids_list)
    source_lm_label_ids_batch = dtorch.stack(source_lm_label_ids_list)
    response_lm_label_ids_batch = dtorch.stack(response_lm_label_ids_list)
    semantic_lm_label_ids_batch = dtorch.stack(semantic_lm_label_ids_list)

    label_batch = torch.LongTensor(label_list)
    
    source_author_ids_batch = dtorch.stack(source_author_ids_list)
    response_author_ids_batch = dtorch.stack(response_author_ids_list)
    res_his_author_ids_batch = dtorch.stack(res_his_author_ids_list)
    res_his_res_author_ids_batch = dtorch.stack(res_his_res_author_ids_list)
   
    # print('source_tokens_batch:', source_tokens_batch.shape)
    # shape batch
    semantic_batch = {"input_ids": semantic_tokens_batch, "token_type_ids": semantic_segment_ids_batch, "attention_mask": semantic_attention_mask_batch, "token_utter_ids": semantic_token_utter_ids_batch, "lm_label_ids": semantic_lm_label_ids_batch}
    source_batch = {"input_ids": source_tokens_batch, "token_type_ids": source_segment_ids_batch, "attention_mask": source_attention_mask_batch, "token_utter_ids": source_token_utter_ids_batch, "lm_label_ids": source_lm_label_ids_batch, "author_ids": source_author_ids_batch}
    res_his_batch = {"input_ids": res_his_tokens_batch, "token_type_ids": res_his_segment_ids_batch, "attention_mask": res_his_attention_mask_batch, "token_utter_ids": res_his_token_utter_ids_batch, "author_ids": res_his_author_ids_batch}
    res_his_res_batch = {"input_ids": res_his_res_tokens_batch, "token_type_ids": res_his_res_segment_ids_batch, "attention_mask": res_his_res_attention_mask_batch, "token_utter_ids": res_his_res_token_utter_ids_batch, "author_ids": res_his_res_author_ids_batch}
    res_his_sem_batch = {"input_ids": res_his_sem_tokens_batch, "token_type_ids": res_his_sem_segment_ids_batch, "attention_mask": res_his_sem_attention_mask_batch, "token_utter_ids": res_his_sem_token_utter_ids_batch}
    response_batch = {"input_ids": response_tokens_batch, "token_type_ids": response_segment_ids_batch, "attention_mask": response_attention_mask_batch, "token_utter_ids": response_token_utter_ids_batch, "labels": label_batch, "lm_label_ids": response_lm_label_ids_batch, "author_ids": response_author_ids_batch}
    # sentence_features = {'semantic': semantic_batch, 'source': source_batch, 'response': response_batch}
    # return sentence_features
    return source_batch, response_batch, semantic_batch, res_his_batch, res_his_res_batch, res_his_sem_batch


def load_responses(data_dir):

    responses = {}

    with codecs.open(data_dir, 'r', 'utf_8') as f:
        all_data = f.readlines()
    for data in all_data:
        data = data.split('\t')
        label = int(data[0])
        context_author = data[1]
        response_author = data[2]
        response_created_utc = data[3]
        response = data[-1]
        if context_author in responses:
            responses[context_author].append({'label': label, 'text': response, 'response_author': response_author, 'response_created_utc': response_created_utc})
        else:
            responses[context_author] = [{'label': label, 'text': response, 'response_author': response_author, 'response_created_utc': response_created_utc}]
    
    return responses


def get_json_data(authors, json_data_path, max_slide_num=10, ctx_loop_count=5, res_loop_count=5, pattern=0, responses=None):
    # print(authors)
    print(f'get_json_start: {time.time()}')
    # json_data_path = '/work/data/reddit_nfl/201809101112_20190102_nfl_minill3_test.json' if type == 1 else '/work/data/reddit_nfl/201809101112_20190102_nfl_minill3_train.json'

    author_list_path = None
    # author_list_path = '/work/data/reddit_nfl/author_list_min30_201809101112_20190102_nfl_minill3_test.txt' if type == 1 else '/work/data/reddit_nfl/author_list_min30_201809101112_20190102_nfl_minill3_train.txt'
    json_data = None
    with open(json_data_path) as f:
        json_data = json.loads(f.read())
    # if 'pinetar321' not in json_data:
    #     print('pinetar321 not included')
    print(f'get_json_json_data_read: {time.time()}')
    author_list = None
    extended_author_list = None
    
    # (1)C-R alignment module
    if author_list_path is None:
        author_list = [k for k in json_data]
        extended_author_list = author_list
    else:
        with open(author_list_path) as f:
            author_list = f.read().strip().splitlines()
        extended_author_list = author_list
        for author in author_list:
            for history in json_data[author]:
                response_author = history['response_author']
                if response_author not in extended_author_list:
                    extended_author_list.append(response_author)
    print(f'get_json_extended_author_list_calculated: {time.time()}')
    authors_histories = {}
    for author in extended_author_list:
        author_histories = {}
        for history in json_data[author]:
            context_created_utc = history['context_created_utc']
            if context_created_utc not in author_histories:
                author_histories[context_created_utc] = history
        author_histories = [author_histories[k] for k in author_histories]
        author_histories = sorted(author_histories, key=lambda x: x['context_created_utc'])
        authors_histories[author] = author_histories
    print(f'get_json_authors_histories_calculated: {time.time()}')
    r = authors_histories

    redis_data = {}

    if not authors:
        authors = []
        keys = [k for k in authors_histories]

        for key in keys:
            author = key
            authors.append(author)

        authors = set(authors)

    for author in tqdm(authors, desc="get_json_data"):
        # (2)Data augmentation module
        if pattern == 1:

            for i in range(max_slide_num):

                data = get_author_data(r, author, ctx_loop_count, res_loop_count, i)
                if data:
                    redis_data[author + "_" + str(i)] = data
                else:
                    break
        elif pattern == 2:

            for l in range(ctx_loop_count):
                data = get_author_data(r, author, l + 1, res_loop_count, 0)
                if data:
                    redis_data[author + "_" + str(l)] = data
                else:
                    break

        elif pattern == 3:

            for i in range(max_slide_num):
                for l in range(ctx_loop_count):
                    data = get_author_data(r, author, l + 1, res_loop_count, i)
                    if data:
                        redis_data[author + "_" + str(i) + "_" + str(l)] = data
                    else:
                        break

        elif pattern == 5 and responses:

            response = responses.get(author)
            rets = get_collect_res_history_data(r, author, ctx_loop_count, res_loop_count, 0, do_cut=False, responses=response)
            if rets:
                for i, ret in enumerate(rets):
                    redis_data[author + "_" + str(i)] = ret

        else:
            data = get_author_data(r, author, ctx_loop_count, res_loop_count, 0, do_cut=False)
            if data:
                redis_data[author] = data

    print(f'get_json_end: {time.time()}')
    return redis_data

def index_by_predicate(lst, pred):
    for i,v in enumerate(lst):
        if pred(v):
            return i
    return None

def get_author_data(r, context_author, ctx_loop_count, res_loop_count, start=0, do_cut=True):
    ctx_histories = r[context_author]
    num_ctx_histories = len(ctx_histories)
    ctx_histories = ctx_histories[
        max([num_ctx_histories - ctx_loop_count - start, 0]):max([num_ctx_histories - start, 0])
    ]
    num_ctx_histories = len(ctx_histories)

    if do_cut and num_ctx_histories != ctx_loop_count:
        return None

    history = ctx_histories[-1]
    res_author = history['response_author']
    res_histories = []
    if res_author in r:
        res_histories = r[res_author]
        res_time = history['response_created_utc']
        res_histories = res_histories[:index_by_predicate(res_histories, lambda x: x['context_created_utc'] >= res_time) or len(res_histories)]
        res_histories = res_histories[max([len(res_histories) - (res_loop_count - 1), 0]):]

    return {'ctx_histories': ctx_histories, 'res_histories': res_histories}

def get_collect_res_history_data(r, context_author, ctx_loop_count, res_loop_count, start=0, do_cut=True, responses=None):

    ret = []
    ctx_histories = r[context_author]
    num_ctx_histories = len(ctx_histories)
    ctx_histories = ctx_histories[
        max([num_ctx_histories - ctx_loop_count - start, 0]):max([num_ctx_histories - start, 0])
    ]
    num_ctx_histories = len(ctx_histories)

    if do_cut and num_ctx_histories != ctx_loop_count:
        return None

    if not responses:
        return None

    for response in responses:
        copy_ctx_histories = deepcopy(ctx_histories)
        res_histories = []
        res_author = response['response_author']
        copy_ctx_histories[-1]['Response'] = response['text']
        if res_author in r:
            res_time = response['response_created_utc']
            res_histories = r[res_author]
            res_histories = res_histories[:index_by_predicate(res_histories, lambda x: x['context_created_utc'] >= res_time) or len(res_histories)]
            res_histories = res_histories[max([len(res_histories) - (res_loop_count - 1), 0]):]

        ret.append({'ctx_histories': copy_ctx_histories, 'res_histories': res_histories, 'label': response['label']})

    return ret
