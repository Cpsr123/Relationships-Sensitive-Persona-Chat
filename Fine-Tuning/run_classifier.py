"""BERT finetuning runner."""
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE,cached_path
from model import RedditLMF
from transformers import BertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import itertools
from sklearn import metrics
import random

from transformers import AdamW
import torch.nn as nn
import torch.nn.utils as utils
from collections import OrderedDict

import sys
sys.path.append('./')
from common.hais_utils_msc_bertfp import *
from common.utils import *
from common.metrics import Metrics
from common.hais_functions import *

def calc_result_multi(predict_lists, truth_lists, label_names):
    results = {}
    exclude_zero = False

    predict_lists = np.array(predict_lists).T
    truth_lists = np.array(truth_lists).T
    for i, name in enumerate(label_names):
        predict_list = predict_lists[i]
        truth_list = truth_lists[i]
        non_zeros = np.array([i for i, e in enumerate(predict_list) if e != 0 or (not exclude_zero)])
        predict_list = np.array(predict_list).reshape(-1)
        truth_list = np.array(truth_list)
        predict_list1 = (predict_list[non_zeros] > 0)
        truth_list1 = (truth_list[non_zeros] > 0)
        test_preds_a7 = np.clip(predict_list, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(truth_list, a_min=-3., a_max=3.)
        acc7 = accuracy_7(test_preds_a7,test_truth_a7)
        f_score = f1_score(predict_list1, truth_list1, average='weighted')
        acc = accuracy_score(truth_list1, predict_list1)
        corr = np.corrcoef(predict_list, truth_list)[0][1]
        mae = np.mean(np.absolute(predict_list - truth_list))
        result = {'acc':acc,
                'F1':f_score,
                'mae':mae,
                'corr':corr,
                'acc7':acc7}
        logger.info("***** %s result *****", name)
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
        results[name] = result

    return results


"""BERT finetuning runner."""
def main(i):
    parser = argparse.ArgumentParser()

    ## Required parameters
    # parser.add_argument("--data_dir", default='nttcp', type=str)
   # parser.add_argument("--data_dir", default='msc_v2', type=str,
    #                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # parser.add_argument("--bert_model", default='../bert-base-japanese-whole-word-masking', type=st)
    parser.add_argument("--bert_model", default='./pretrained', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name", default='Multi', type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir", default='Cross-Modal-BERT-master/msc_output', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--src_length", default=35, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--res_length", default=25, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--sem_length", default=25, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train", default=True,
                        help="Whether to run training.'store_true'")
    parser.add_argument("--do_test", default=True,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", default=True,
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=50, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=50, type=int,
                        help="Total batch size for eval.")
    #parser.add_argument("--test_batch_size", default=5, type=int,
    #                    help="Total batch size for test.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.5e-5")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Whether not to use CUDA when available")
    #parser.add_argument('--seed', type=int, default=11111,
    #                    help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--score_file_path', type=str, default='Cross-Modal-BERT-master/msc_output/scores')
    parser.add_argument('--ctx_loop_count', type=int, default=5)
    parser.add_argument('--res_loop_count', type=int, default=5)
    parser.add_argument('--max_slide_num', type=int, default=10)
    parser.add_argument('--train_convert_pattern', type=int, default=0)
    parser.add_argument('--eval_convert_pattern', type=int, default=4)
    parser.add_argument('--test_convert_pattern', type=int, default=4)

    parser.add_argument('--load_checkpoint', type=str, default='')

    parser.add_argument('--target_train_authors', type=str, default=None)
    #parser.add_argument('--target_test_authors', type=str, default=None)
    parser.add_argument('--target_dev_authors', type=str, default=None)

    parser.add_argument('--train_data_path', type=str, default=None)
    #parser.add_argument('--test_data_path', type=str, default=None)
    parser.add_argument('--dev_data_path', type=str, default=None)

    parser.add_argument('--author_list_path', type=str, default=None)

    parser.add_argument('--author_list_len', type=int)

    parser.add_argument('--sem_type', type=int, default=0)

    parser.add_argument("--topic_num", type=int, default=50, help="topic_num")

    parser.add_argument("--head_num", type=int, default=1, help="head_num")

    parser.add_argument("--cpu_process_num", type=int, default=6, help="cpu process num.")

    parser.add_argument('--responses_tsv', type=str, default=None)

    parser.add_argument("--model_type", type=int, default=0, help="")

    parser.add_argument('--use_input_history', action="store_true", help="Set this flag if you want to use input histories.")
    parser.add_argument('--use_res_history', action="store_true", help="Set this flag if you want to use response histories.")
    parser.add_argument('--use_utterance_embedding_latest', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_utterance_embedding_input_history', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_utterance_embedding_res_history', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_semantic_embedding_latest', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_semantic_embedding_input_history', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_semantic_embedding_res_history', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_author_embedding_latest', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_author_embedding_input_history', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_author_embedding_res_history', action="store_true", help="Set this flag if you want to use utterance embedding.")

    parser.add_argument('--use_train_input_history', action="store_true", help="Set this flag if you want to use input histories.")
    parser.add_argument('--use_train_res_history', action="store_true", help="Set this flag if you want to use response histories.")
    parser.add_argument('--use_train_utterance_embedding_latest', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_train_utterance_embedding_input_history', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_train_utterance_embedding_res_history', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_train_semantic_embedding_latest', action="store_true", help="")
    parser.add_argument('--use_train_semantic_embedding_input_history', action="store_true", help="")
    parser.add_argument('--use_train_semantic_embedding_res_history', action="store_true", help="")
    parser.add_argument('--use_train_author_embedding_latest', action="store_true", help="")
    parser.add_argument('--use_train_author_embedding_input_history', action="store_true", help="")
    parser.add_argument('--use_train_author_embedding_res_history', action="store_true", help="")

    parser.add_argument('--use_test_input_history', action="store_true", help="Set this flag if you want to use input histories.")
    parser.add_argument('--use_test_res_history', action="store_true", help="Set this flag if you want to use response histories.")
    parser.add_argument('--use_test_utterance_embedding_latest', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_test_utterance_embedding_input_history', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_test_utterance_embedding_res_history', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_test_semantic_embedding_latest', action="store_true", help="")
    parser.add_argument('--use_test_semantic_embedding_input_history', action="store_true", help="")
    parser.add_argument('--use_test_semantic_embedding_res_history', action="store_true", help="")
    parser.add_argument('--use_test_author_embedding_latest', action="store_true", help="")
    parser.add_argument('--use_test_author_embedding_input_history', action="store_true", help="")
    parser.add_argument('--use_test_author_embedding_res_history', action="store_true", help="")

    args = parser.parse_args()
    processors = {
        "multi": PgProcessor,
    }

    num_labels_task = {
        "multi": 1,
    }

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = 1
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    seed_num = np.random.randint(1,10000)
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed_num)

    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_test` must be True.")

    #if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #    raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()
    # tokenizer = BertJapaneseTokenizer.from_pretrained(args.bert_model)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    special_tokens_dict = {'eos_token': '[eos]'}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    train_set = None
    num_train_optimization_steps = None
    if args.do_train:
        authors = {
                "train": get_authors_from_txt(args.target_train_authors),
                "dev": get_authors_from_txt(args.target_dev_authors),
                "test": None
                  }

        json_data_path = {
                'train': args.train_data_path,
                'dev': args.dev_data_path,
                'test': None
                }
        response_list = load_responses(args.responses_tsv)
        train_set, valid_set, _ = load_reddit(json_data_path, authors, args.max_slide_num, args.ctx_loop_count, args.res_loop_count, args.train_convert_pattern, args.eval_convert_pattern, args.test_convert_pattern, response_list)

    all_author_list = get_authors_from_txt(args.author_list_path)
    args.author_list_len= len(all_author_list)

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format("-1"))
    model = RedditLMF(args)

    load_parameter = torch.load(args.load_checkpoint, map_location=device)

    new_state_dict = OrderedDict()
    load_parameter_names = []
    for k, v in load_parameter.items():
        #if k.startswith('bert_1.cls'):
        #    continue
        if k.startswith('bert_1.'):
            k1 = k[7:]
            k1 = "bert_2." + k1
            new_state_dict[k1] = v
            load_parameter_names.append(k1)
        new_state_dict[k] = v
        load_parameter_names.append(k)
    load_parameter_names.sort()
    model.load_state_dict(state_dict=new_state_dict, strict=False)

    model_parameter_names = []
    for name, param in model.named_parameters():
        model_parameter_names.append(name)
        param.requires_grad = False
        if "bert_2" in name:
            param.requires_grad = True
    model_parameter_names.sort()
    missing_key = set(load_parameter_names) ^ set(model_parameter_names)
    logger.info("Missing Key : {}".format(missing_key))

    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    max_acc = 0
    min_loss = 100

    if args.do_train:
        source_batch, response_batch, semantic_batch, res_his_batch, res_his_res_batch, res_his_sem_batch = convert_reddit_to_features(train_set,
                                                                                                                                       args.src_length, 
                                                                                                                                       args.res_length, 
                                                                                                                                       args.sem_length, 
                                                                                                                                       tokenizer, 
                                                                                                                                       args.ctx_loop_count, 
                                                                                                                                       args.res_loop_count, 
                                                                                                                                       args.train_convert_pattern, 
                                                                                                                                       args.sem_type,
                                                                                                                                       None,
                                                                                                                                       False,
                                                                                                                                       args.cpu_process_num,
                                                                                                                                       all_author_list=all_author_list
                                                                                                                                       )
        train_set_len = len(source_batch["input_ids"])
        num_train_optimization_steps = (train_set_len / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_set))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_labels = response_batch['labels'].to(torch.float)
        all_src_input_ids = source_batch['input_ids'].to(torch.long)
        all_src_token_type_ids = source_batch['token_type_ids'].to(torch.long)
        all_src_attention_mask = source_batch['attention_mask'].to(torch.long)
        all_src_token_uutr_ids = source_batch['token_utter_ids'].to(torch.long)
        all_res_input_ids = response_batch['input_ids'].to(torch.long)
        all_res_token_type_ids = response_batch['token_type_ids'].to(torch.long)
        all_res_attention_mask = response_batch['attention_mask'].to(torch.long)
        all_res_token_uutr_ids = response_batch['token_utter_ids'].to(torch.long)
        all_sem_input_ids = semantic_batch['input_ids'].to(torch.long)
        all_sem_token_type_ids = semantic_batch['token_type_ids'].to(torch.long)
        all_sem_attention_mask = semantic_batch['attention_mask'].to(torch.long)
        all_sem_token_uutr_ids = semantic_batch['token_utter_ids'].to(torch.long)
        all_res_his_input_ids = res_his_batch['input_ids'].to(torch.long)
        all_res_his_token_type_ids = res_his_batch['token_type_ids'].to(torch.long)
        all_res_his_attention_mask = res_his_batch['attention_mask'].to(torch.long)
        all_res_his_token_uutr_ids = res_his_batch['token_utter_ids'].to(torch.long)
        all_res_his_res_input_ids = res_his_res_batch['input_ids'].to(torch.long)
        all_res_his_res_token_type_ids = res_his_res_batch['token_type_ids'].to(torch.long)
        all_res_his_res_attention_mask = res_his_res_batch['attention_mask'].to(torch.long)
        all_res_his_res_token_uutr_ids = res_his_res_batch['token_utter_ids'].to(torch.long)
        all_res_his_sem_input_ids = res_his_sem_batch['input_ids'].to(torch.long)
        all_res_his_sem_token_type_ids = res_his_sem_batch['token_type_ids'].to(torch.long)
        all_res_his_sem_attention_mask = res_his_sem_batch['attention_mask'].to(torch.long)
        all_res_his_sem_token_uutr_ids = res_his_sem_batch['token_utter_ids'].to(torch.long)
        all_src_token_author_ids = source_batch["author_ids"].to(torch.long)
        all_res_token_author_ids =response_batch["author_ids"].to(torch.long)
        all_res_his_token_author_ids =res_his_batch["author_ids"].to(torch.long)
        all_res_his_res_token_author_ids =res_his_res_batch["author_ids"].to(torch.long)

        train_data = TensorDataset(
            all_labels,
            all_src_input_ids, all_src_token_type_ids, all_src_attention_mask, all_src_token_uutr_ids,
            all_res_input_ids, all_res_token_type_ids, all_res_attention_mask, all_res_token_uutr_ids,
            all_sem_input_ids, all_sem_token_type_ids, all_sem_attention_mask, all_sem_token_uutr_ids,
            all_res_his_input_ids, all_res_his_token_type_ids, all_res_his_attention_mask, all_res_his_token_uutr_ids,
            all_res_his_res_input_ids, all_res_his_res_token_type_ids, all_res_his_res_attention_mask, all_res_his_res_token_uutr_ids,
            all_res_his_sem_input_ids, all_res_his_sem_token_type_ids, all_res_his_sem_attention_mask, all_res_his_sem_token_uutr_ids,
            all_src_token_author_ids, all_res_token_author_ids, all_res_his_token_author_ids, all_res_his_res_token_author_ids
        )
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)

        print('train_data_count: ', all_src_input_ids.shape)

        ## Evaluate for each epcoh
        source_batch, response_batch, semantic_batch, res_his_batch, res_his_res_batch, res_his_sem_batch = convert_reddit_to_features(valid_set, 
                                                                                                                                       args.src_length, 
                                                                                                                                       args.res_length, 
                                                                                                                                       args.sem_length, 
                                                                                                                                       tokenizer, 
                                                                                                                                       args.ctx_loop_count, 
                                                                                                                                       args.res_loop_count, 
                                                                                                                                       args.eval_convert_pattern, 
                                                                                                                                       args.sem_type, 
                                                                                                                                       response_list,
                                                                                                                                       False,
                                                                                                                                       args.cpu_process_num,
                                                                                                                                       all_author_list=all_author_list
                                                                                                                                       )

        valid_set_len = len(source_batch["input_ids"])
        all_labels = response_batch['labels'].to(torch.float)
        all_src_input_ids = source_batch['input_ids'].to(torch.long)
        all_src_token_type_ids = source_batch['token_type_ids'].to(torch.long)
        all_src_attention_mask = source_batch['attention_mask'].to(torch.long)
        all_src_token_uutr_ids = source_batch['token_utter_ids'].to(torch.long)
        all_res_input_ids = response_batch['input_ids'].to(torch.long)
        all_res_token_type_ids = response_batch['token_type_ids'].to(torch.long)
        all_res_attention_mask = response_batch['attention_mask'].to(torch.long)
        all_res_token_uutr_ids = response_batch['token_utter_ids'].to(torch.long)
        all_sem_input_ids = semantic_batch['input_ids'].to(torch.long)
        all_sem_token_type_ids = semantic_batch['token_type_ids'].to(torch.long)
        all_sem_attention_mask = semantic_batch['attention_mask'].to(torch.long)
        all_sem_token_uutr_ids = semantic_batch['token_utter_ids'].to(torch.long)
        all_res_his_input_ids = res_his_batch['input_ids'].to(torch.long)
        all_res_his_token_type_ids = res_his_batch['token_type_ids'].to(torch.long)
        all_res_his_attention_mask = res_his_batch['attention_mask'].to(torch.long)
        all_res_his_token_uutr_ids = res_his_batch['token_utter_ids'].to(torch.long)
        all_res_his_res_input_ids = res_his_res_batch['input_ids'].to(torch.long)
        all_res_his_res_token_type_ids = res_his_res_batch['token_type_ids'].to(torch.long)
        all_res_his_res_attention_mask = res_his_res_batch['attention_mask'].to(torch.long)
        all_res_his_res_token_uutr_ids = res_his_res_batch['token_utter_ids'].to(torch.long)
        all_res_his_sem_input_ids = res_his_sem_batch['input_ids'].to(torch.long)
        all_res_his_sem_token_type_ids = res_his_sem_batch['token_type_ids'].to(torch.long)
        all_res_his_sem_attention_mask = res_his_sem_batch['attention_mask'].to(torch.long)
        all_res_his_sem_token_uutr_ids = res_his_sem_batch['token_utter_ids'].to(torch.long)
        all_src_token_author_ids = source_batch["author_ids"].to(torch.long)
        all_res_token_author_ids =response_batch["author_ids"].to(torch.long)
        all_res_his_token_author_ids =res_his_batch["author_ids"].to(torch.long)
        all_res_his_res_token_author_ids =res_his_res_batch["author_ids"].to(torch.long)

        eval_data = TensorDataset(
            all_labels,
            all_src_input_ids, all_src_token_type_ids, all_src_attention_mask, all_src_token_uutr_ids,
            all_res_input_ids, all_res_token_type_ids, all_res_attention_mask, all_res_token_uutr_ids,
            all_sem_input_ids, all_sem_token_type_ids, all_sem_attention_mask, all_sem_token_uutr_ids,
            all_res_his_input_ids, all_res_his_token_type_ids, all_res_his_attention_mask, all_res_his_token_uutr_ids,
            all_res_his_res_input_ids, all_res_his_res_token_type_ids, all_res_his_res_attention_mask, all_res_his_res_token_uutr_ids,
            all_res_his_sem_input_ids, all_res_his_sem_token_type_ids, all_res_his_sem_attention_mask, all_res_his_sem_token_uutr_ids,
            all_src_token_author_ids, all_res_token_author_ids, all_res_his_token_author_ids, all_res_his_res_token_author_ids
        )
        eval_dataloader = DataLoader(eval_data, batch_size=args.eval_batch_size)

        print('eval_data_count: ', all_src_input_ids.shape)

        loss_func = nn.BCELoss()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=True)

        epoch = 0
        patience = 0
        init_clip_max_norm = 5.0
        best_result = [0, 0, 0, 0, 0, 0]
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            epoch += 1

            args.use_input_history = args.use_train_input_history
            args.use_res_history = args.use_train_res_history
            args.use_utterance_embedding_latest = args.use_train_utterance_embedding_latest
            args.use_utterance_embedding_input_history = args.use_train_utterance_embedding_input_history
            args.use_utterance_embedding_res_history = args.use_train_utterance_embedding_res_history
            args.use_semantic_embedding_latest = args.use_train_semantic_embedding_latest
            args.use_semantic_embedding_input_history = args.use_train_semantic_embedding_input_history
            args.use_semantic_embedding_res_history = args.use_train_semantic_embedding_res_history
            args.use_author_embedding_latest = args.use_train_author_embedding_latest
            args.use_author_embedding_input_history = args.use_train_author_embedding_input_history
            args.use_author_embedding_res_history = args.use_train_author_embedding_res_history

            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if epoch - 1 >= 2 and patience >= 3:
                    logger.info("Reload the best model...")
                    output_model_file = os.path.join(args.output_dir, "best.pt")
                    model.load_state_dict(state_dict=torch.load(output_model_file))

                    # adjust_learning_rate
                    decay_rate=.5
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * decay_rate
                        args.learning_rate = param_group['lr']
                        logger.info("Decay learning rate to: {}".format(args.learning_rate))
                        patience = 0

                optimizer.zero_grad()

                batch = tuple(t.to(device) for t in batch)
                labels, \
                src_input_ids, src_token_type_ids, src_attention_mask, src_token_uutr_ids, \
                res_input_ids, res_token_type_ids, res_attention_mask, res_token_uutr_ids, \
                sem_input_ids, sem_token_type_ids, sem_attention_mask, sem_token_uutr_ids, \
                res_his_input_ids, res_his_token_type_ids, res_his_attention_mask, res_his_token_uutr_ids, \
                res_his_res_input_ids, res_his_res_token_type_ids, res_his_res_attention_mask, res_his_res_token_uutr_ids, \
                res_his_sem_input_ids, res_his_sem_token_type_ids, res_his_sem_attention_mask, res_his_sem_token_uutr_ids, \
                src_token_author_ids, res_token_author_ids, res_his_token_author_ids, res_his_res_token_author_ids = batch
                logits, _ = model(labels,
                    src_input_ids, src_token_type_ids, src_attention_mask, src_token_uutr_ids,
                    res_input_ids, res_token_type_ids, res_attention_mask, res_token_uutr_ids,
                    sem_input_ids, sem_token_type_ids, sem_attention_mask, sem_token_uutr_ids,
                    res_his_input_ids, res_his_token_type_ids, res_his_attention_mask, res_his_token_uutr_ids,
                    res_his_res_input_ids, res_his_res_token_type_ids, res_his_res_attention_mask, res_his_res_token_uutr_ids,
                    res_his_sem_input_ids, res_his_sem_token_type_ids, res_his_sem_attention_mask, res_his_sem_token_uutr_ids,
                    src_token_author_ids, res_token_author_ids, res_his_token_author_ids, res_his_res_token_author_ids,
                    args)

                logits = torch.sigmoid(logits)
                loss = loss_func(logits.squeeze(), target=labels)
                loss.backward()

                optimizer.step()

                if step % 100 == 0:
                    logger.info('Batch[{}] - loss: {}  batch_size:{}'.format(i, loss.item(),
                                                                   args.train_batch_size))

                if init_clip_max_norm is not None:
                    utils.clip_grad_norm_(model.parameters(), max_norm=init_clip_max_norm)

                tr_loss += loss.item()
            cnt = train_set_len // args.train_batch_size + 1
            logger.info("Average loss:{:.6f} ".format(tr_loss / cnt))
            logger.info("***** Running evaluation *****")
            logger.info("  Num examples = %d", valid_set_len)
            logger.info("  Batch size = %d", args.eval_batch_size)
            model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            if os.path.isfile(args.score_file_path + "_{}.txt".format(epoch)):
                os.remove(args.score_file_path + "_{}.txt".format(epoch))

            args.use_input_history = args.use_test_input_history
            args.use_res_history = args.use_test_res_history
            args.use_utterance_embedding_latest = args.use_test_utterance_embedding_latest
            args.use_utterance_embedding_input_history = args.use_test_utterance_embedding_input_history
            args.use_utterance_embedding_res_history = args.use_test_utterance_embedding_res_history
            args.use_semantic_embedding_latest = args.use_test_semantic_embedding_latest
            args.use_semantic_embedding_input_history = args.use_test_semantic_embedding_input_history
            args.use_semantic_embedding_res_history = args.use_test_semantic_embedding_res_history
            args.use_author_embedding_latest = args.use_test_author_embedding_latest
            args.use_author_embedding_input_history = args.use_test_author_embedding_input_history
            args.use_author_embedding_res_history = args.use_test_author_embedding_res_history
            for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                labels, \
                src_input_ids, src_token_type_ids, src_attention_mask, src_token_uutr_ids, \
                res_input_ids, res_token_type_ids, res_attention_mask, res_token_uutr_ids, \
                sem_input_ids, sem_token_type_ids, sem_attention_mask, sem_token_uutr_ids, \
                res_his_input_ids, res_his_token_type_ids, res_his_attention_mask, res_his_token_uutr_ids, \
                res_his_res_input_ids, res_his_res_token_type_ids, res_his_res_attention_mask, res_his_res_token_uutr_ids, \
                res_his_sem_input_ids, res_his_sem_token_type_ids, res_his_sem_attention_mask, res_his_sem_token_uutr_ids, \
                src_token_author_ids, res_token_author_ids, res_his_token_author_ids, res_his_res_token_author_ids = batch
                with torch.no_grad():
                    logits, _ = model(labels,
                        src_input_ids, src_token_type_ids, src_attention_mask, src_token_uutr_ids,
                        res_input_ids, res_token_type_ids, res_attention_mask, res_token_uutr_ids,
                        sem_input_ids, sem_token_type_ids, sem_attention_mask, sem_token_uutr_ids,
                        res_his_input_ids, res_his_token_type_ids, res_his_attention_mask, res_his_token_uutr_ids,
                        res_his_res_input_ids, res_his_res_token_type_ids, res_his_res_attention_mask, res_his_res_token_uutr_ids,
                        res_his_sem_input_ids, res_his_sem_token_type_ids, res_his_sem_attention_mask, res_his_sem_token_uutr_ids,
                        src_token_author_ids, res_token_author_ids, res_his_token_author_ids, res_his_res_token_author_ids,
                        args)

                logits = logits.squeeze()
                logits = logits.detach().cpu().numpy().tolist()
                labels = labels.tolist()

                with open(args.score_file_path + "_{}.txt".format(epoch), 'a') as output:
                    for score, label in zip(logits, labels):
                        output.write(
                                str(score) + '\t' +
                                str(int(label)) + '\n'
                                )

            metrics = Metrics(args.score_file_path + "_{}.txt".format(epoch))
            metrics.segment = 10
            result = metrics.evaluate_all_metrics()
            #logger.info("Evaluation Result:\nMAP:{}\tMRR:{}\tP@1:{}\tR1:{}\tR2:{}\tR5:{}".format(result[0], result[1], result[2], result[3], result[4], result[5]))
            print("Evaluation Result MAP:{}\tMRR:{}\tP@1:{}\tR1:{}\tR2:{}\tR5:{}".format(result[0], result[1], result[2], result[3], result[4], result[5]))

            if result[3] + result[4] + result[5] > best_result[3] + best_result[4] + best_result[5]:
                patience = 0
                best_result = result
                output_model_file = os.path.join(args.output_dir, "best.pt")
                torch.save(model.state_dict(), output_model_file)
                logger.info("save model!!!\n")
            else:
                patience += 1


        args.use_input_history = args.use_test_input_history
        args.use_res_history = False
        args.use_utterance_embedding_latest = args.use_test_utterance_embedding_latest
        args.use_utterance_embedding_input_history = args.use_test_utterance_embedding_input_history
        args.use_utterance_embedding_res_history = args.use_test_utterance_embedding_res_history
        args.use_semantic_embedding_latest = args.use_test_semantic_embedding_latest
        args.use_semantic_embedding_input_history = args.use_test_semantic_embedding_input_history
        args.use_semantic_embedding_res_history = args.use_test_semantic_embedding_res_history
        args.use_author_embedding_latest = args.use_test_author_embedding_latest
        args.use_author_embedding_input_history = args.use_test_author_embedding_input_history
        args.use_author_embedding_res_history = args.use_test_author_embedding_res_history

        output_model_file = os.path.join(args.output_dir, "best.pt")
        model.load_state_dict(state_dict=torch.load(output_model_file))
        model.eval()
        logger.info("### No Response History")
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            labels, \
            src_input_ids, src_token_type_ids, src_attention_mask, src_token_uutr_ids, \
            res_input_ids, res_token_type_ids, res_attention_mask, res_token_uutr_ids, \
            sem_input_ids, sem_token_type_ids, sem_attention_mask, sem_token_uutr_ids, \
            res_his_input_ids, res_his_token_type_ids, res_his_attention_mask, res_his_token_uutr_ids, \
            res_his_res_input_ids, res_his_res_token_type_ids, res_his_res_attention_mask, res_his_res_token_uutr_ids, \
            res_his_sem_input_ids, res_his_sem_token_type_ids, res_his_sem_attention_mask, res_his_sem_token_uutr_ids, \
            src_token_author_ids, res_token_author_ids, res_his_token_author_ids, res_his_res_token_author_ids = batch
            with torch.no_grad():
                logits, _ = model(labels,
                    src_input_ids, src_token_type_ids, src_attention_mask, src_token_uutr_ids,
                    res_input_ids, res_token_type_ids, res_attention_mask, res_token_uutr_ids,
                    sem_input_ids, sem_token_type_ids, sem_attention_mask, sem_token_uutr_ids,
                    res_his_input_ids, res_his_token_type_ids, res_his_attention_mask, res_his_token_uutr_ids,
                    res_his_res_input_ids, res_his_res_token_type_ids, res_his_res_attention_mask, res_his_res_token_uutr_ids,
                    res_his_sem_input_ids, res_his_sem_token_type_ids, res_his_sem_attention_mask, res_his_sem_token_uutr_ids,
                    src_token_author_ids, res_token_author_ids, res_his_token_author_ids, res_his_res_token_author_ids,
                    args)

            logits = logits.squeeze()
            logits = logits.detach().cpu().numpy().tolist()
            labels = labels.tolist()

            with open(args.score_file_path + "_No_Response_History.txt", 'a') as output:
                for score, label in zip(logits, labels):
                    output.write(
                            str(score) + '\t' +
                            str(int(label)) + '\n'
                            )

        metrics = Metrics(args.score_file_path + "_No_Response_History.txt")
        metrics.segment = 10
        result = metrics.evaluate_all_metrics()
        #logger.info("Evaluation Result(No res his):\nMAP:{}\tMRR:{}\tP@1:{}\tR1:{}\tR2:{}\tR5:{}".format(result[0], result[1], result[2], result[3], result[4], result[5]))
        print("Evaluation Result(No res his) MAP:{}\tMRR:{}\tP@1:{}\tR1:{}\tR2:{}\tR5:{}".format(result[0], result[1], result[2], result[3], result[4], result[5]))
        print("Best Result MAP:{}\tMRR:{}\tP@1:{}\tR1:{}\tR2:{}\tR5:{}".format(best_result[0], best_result[1], best_result[2], best_result[3], best_result[4], best_result[5]))
if __name__ == "__main__":
    for i in range(1):
        results = main(i)

