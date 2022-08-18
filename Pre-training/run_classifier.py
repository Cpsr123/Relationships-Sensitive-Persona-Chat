"""BERT finetuning runner."""
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm, trange
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from model import RedditLMF
from transformers import BertTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import itertools
import random

from transformers import AdamW
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import sys
sys.path.append('./')

from common.hais_utils_msc_bertfp import *
from common.utils import *
from common.metrics import Metrics
from common.utils import *
from common.hais_functions import *

num="0"

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0

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
    #parser.add_argument("--data_dir", default='msc_v2', type=str,
    #                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    # parser.add_argument("--bert_model", default='../bert-base-japanese-whole-word-masking', type=st)
    parser.add_argument("--bert_model", default='./pretrained', type=str,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--task_name", default='Multi', type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir", default='Cross-Modal-BERT-master/msc_output' + num, type=str,
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
    #parser.add_argument("--eval_batch_size", default=50, type=int,
    #                    help="Total batch size for eval.")
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
    #parser.add_argument('--score_file_path', type=str, default='Cross-Modal-BERT-master/msc_output' + num + '/scores')

    parser.add_argument('--ctx_loop_count', type=int, default=5)
    parser.add_argument('--res_loop_count', type=int, default=5)
    parser.add_argument('--max_slide_num', type=int, default=10)
    parser.add_argument('--train_convert_pattern', type=int, default=0)
    parser.add_argument('--eval_convert_pattern', type=int, default=4)
    parser.add_argument('--test_convert_pattern', type=int, default=4)

    parser.add_argument('--target_train_authors', type=str, default=None)
    #parser.add_argument('--target_test_authors', type=str, default=None)
    #parser.add_argument('--target_dev_authors', type=str, default=None)

    parser.add_argument('--train_data_path', type=str, default=None)
    #parser.add_argument('--test_data_path', type=str, default=None)
    #parser.add_argument('--dev_data_path', type=str, default=None)

    parser.add_argument('--author_list_path', type=str, default=None)
    parser.add_argument('--author_list_len', type=int)

    parser.add_argument('--sem_type', type=int, default=0)

    parser.add_argument("--topic_num", type=int, default=50, help="topic_num")

    parser.add_argument("--head_num", type=int, default=1, help="head_num")

    parser.add_argument("--cpu_process_num", type=int, default=6, help="cpu process num.")

    parser.add_argument("--model_type", type=int, default=0, help="0:direct 1:incremental")

    parser.add_argument('--use_input_history', action="store_true", help="Set this flag if you want to use input histories.")
    parser.add_argument('--use_res_history', action="store_true", help="Set this flag if you want to use response histories.")
    parser.add_argument('--use_utterance_embedding_latest', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_utterance_embedding_input_history', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_utterance_embedding_res_history', action="store_true", help="Set this flag if you want to use utterance embedding.")
    parser.add_argument('--use_semantic_embedding_latest', action="store_true", help="")
    parser.add_argument('--use_semantic_embedding_input_history', action="store_true", help="")
    parser.add_argument('--use_semantic_embedding_res_history', action="store_true", help="")
    parser.add_argument('--use_author_embedding_latest', action="store_true", help="")
    parser.add_argument('--use_author_embedding_input_history', action="store_true", help="")
    parser.add_argument('--use_author_embedding_res_history', action="store_true", help="")

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
                "dev": None,
                "test": None
                  }

        json_data_path = {
                'train': args.train_data_path,
                'dev': None,
                'test': None
                }

        train_set, _, _ = load_reddit(json_data_path, authors, args.max_slide_num, args.ctx_loop_count, args.res_loop_count, args.train_convert_pattern, args.eval_convert_pattern, args.test_convert_pattern)

    all_author_list = get_authors_from_txt(args.author_list_path)
    args.author_list_len= len(all_author_list)

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format("-1"))
    model = RedditLMF(args)

    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    max_acc = 0
    min_loss = 100
    # pre_train
    #response_list = load_responses(args.data_dir)
    response_list = None
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
                                                                                                                                       use_mlm=True,
                                                                                                                                       cpu_process_num=args.cpu_process_num,
                                                                                                                                       all_author_list=all_author_list
                                                                                                                                    )
        train_set_len = len(source_batch["input_ids"])
        num_train_steps = (train_set_len / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_set_len)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
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
        all_res_token_author_ids = response_batch["author_ids"].to(torch.long)
        all_res_his_token_author_ids = res_his_batch["author_ids"].to(torch.long)
        all_res_his_res_token_author_ids = res_his_res_batch["author_ids"].to(torch.long)

        all_src_lm_label_ids = source_batch['lm_label_ids'].to(torch.long)
        all_res_lm_label_ids = response_batch['lm_label_ids'].to(torch.long)

        train_data = TensorDataset(
            all_labels,
            all_src_input_ids, all_src_token_type_ids, all_src_attention_mask, all_src_token_uutr_ids,
            all_res_input_ids, all_res_token_type_ids, all_res_attention_mask, all_res_token_uutr_ids,
            all_sem_input_ids, all_sem_token_type_ids, all_sem_attention_mask, all_sem_token_uutr_ids,
            all_res_his_input_ids, all_res_his_token_type_ids, all_res_his_attention_mask, all_res_his_token_uutr_ids,
            all_res_his_res_input_ids, all_res_his_res_token_type_ids, all_res_his_res_attention_mask, all_res_his_res_token_uutr_ids,
            all_res_his_sem_input_ids, all_res_his_sem_token_type_ids, all_res_his_sem_attention_mask, all_res_his_sem_token_uutr_ids,
            all_src_lm_label_ids, all_res_lm_label_ids,
            all_src_token_author_ids, all_res_token_author_ids, all_res_his_token_author_ids, all_res_his_res_token_author_ids
        )
        train_dataloader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)

        print('train_data_count: ', all_src_input_ids.shape)

        loss_func = nn.BCELoss()
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=True)

        epoch = 0
        global_step = 0
        learning_rate=args.learning_rate
        before = 10
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            tr_loss = 0
            epoch += 1
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                optimizer.zero_grad()
                batch = tuple(t.to(device) for t in batch)
                labels, \
                src_input_ids, src_token_type_ids, src_attention_mask, src_token_uutr_ids, \
                res_input_ids, res_token_type_ids, res_attention_mask, res_token_uutr_ids, \
                sem_input_ids, sem_token_type_ids, sem_attention_mask, sem_token_uutr_ids, \
                res_his_input_ids, res_his_token_type_ids, res_his_attention_mask, res_his_token_uutr_ids, \
                res_his_res_input_ids, res_his_res_token_type_ids, res_his_res_attention_mask, res_his_res_token_uutr_ids, \
                res_his_sem_input_ids, res_his_sem_token_type_ids, res_his_sem_attention_mask, res_his_sem_token_uutr_ids, \
                src_lm_label_ids, res_lm_label_ids, \
                src_token_author_ids, res_token_author_ids, res_his_token_author_ids, res_his_res_token_author_ids = batch

                prediction_scores, seq_relationship_score = model(labels,
                    src_input_ids, src_token_type_ids, src_attention_mask, src_token_uutr_ids,
                    res_input_ids, res_token_type_ids, res_attention_mask, res_token_uutr_ids,
                    sem_input_ids, sem_token_type_ids, sem_attention_mask, sem_token_uutr_ids,
                    res_his_input_ids, res_his_token_type_ids, res_his_attention_mask, res_his_token_uutr_ids,
                    res_his_res_input_ids, res_his_res_token_type_ids, res_his_res_attention_mask, res_his_res_token_uutr_ids,
                    res_his_sem_input_ids, res_his_sem_token_type_ids, res_his_sem_attention_mask, res_his_sem_token_uutr_ids,
                    src_token_author_ids, res_token_author_ids, res_his_token_author_ids, res_his_res_token_author_ids,
                    args)

                history_length = src_input_ids.shape[1]
                source_lm_label_ids = src_lm_label_ids[:,history_length - 1, :]
                response_lm_label_ids = res_lm_label_ids[:, history_length - 1, :]

                lm_label_ids = torch.cat([source_lm_label_ids, response_lm_label_ids[:,1:]], dim=1)

                is_next = labels.to(torch.long)

                if lm_label_ids is not None and is_next is not None:
                    loss_fct = CrossEntropyLoss(ignore_index=-1)
                    masked_lm_loss = loss_fct(prediction_scores.view(-1, tokenizer.vocab_size),
                            lm_label_ids.view(-1))
                    next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), is_next.view(-1))
                    # TODO labels3の場合 next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 3), is_next.view(-1))
                    total_loss = masked_lm_loss + next_sentence_loss

                model.zero_grad()
                loss = total_loss

                if step % 100 == 0:
                    print('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(step, loss.item(),args.train_batch_size) )
                    logger.info('Batch[{}] - loss: {:.6f}  batch_size:{}'.format(step, loss.item(),args.train_batch_size) )

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                else:
                    loss.backward()

                tr_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    if global_step / num_train_steps < args.warmup_proportion:
                        lr_this_step = learning_rate * warmup_linear(global_step/num_train_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            averloss=tr_loss/step
            print("epoch: %d\taverageloss: %f\tstep: %d "%(epoch,averloss,step))
            print("current learning_rate: {}".format(learning_rate))
            logger.info("epoch: %d\taverageloss: %f\tstep: %d "%(epoch,averloss,step))
            logger.info("current learning_rate: {}".format(learning_rate))
            if global_step/num_train_steps > args.warmup_proportion and averloss > before - 0.01:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.9
                    learning_rate = param_group['lr']
                print("Decay learning rate to: {}".format(learning_rate))
                logger.info("Decay learning rate to: {}".format(learning_rate))

            before=averloss

            if True:
                # Save a trained model
                logger.info("** ** * Saving fine - tuned model ** ** * ")
                checkpoint_prefix = 'checkpoint' + str(epoch)
                output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_dir1 = output_dir + '/bert.pt'
                torch.save(model.state_dict(), output_dir1)

if __name__ == "__main__":
    for i in range(1):
        results = main(i)

