#!/bin/bash

CUDA_VISIBLE_DEVICES=0

NUM_TRAIN_EPOCHS=5

DATA_TYPE="minill3nfl10_data"
RESPONSES_TSV="/reddit_nfl_minill3nfl10_data_20220707/201809101112_20190102_nfl_minill3_test_responses.tsv"

TRAIN_BATCH_SIZE=32

LEARNING_RATE=1e-5

BERT_MODEL="bert-base-uncased"

MODEL_TYPE=0 # 0:direct 1:incremental
TRAIN_CONVERT_PATTERN=1
CTX_LOOP_COUNT=3
RES_LOOP_COUNT=3
MAX_SLIDE_NUM=5
SRC_LENGTH=35
RES_LENGTH=25
SEM_TYPE=0 # 0:title+selftext 1:title
SEM_LENGTH=25
TOPIC_NUM=50

EVAL_CONVERT_PATTERN=4

TARGET_TRAIN_AUTHORS="/reddit_nfl_minill3nfl10_data_20220707/author_list_min20_201809101112_20190102_nfl_minill3_train.txt"
TARGET_DEV_AUTHORS="/reddit_nfl_minill3nfl10_data_20220707/author_list_min20_201809101112_20190102_nfl_minill3_test.txt"

TRAIN_DATA_PATH="/reddit_nfl_minill3nfl10_data_20220707/201809101112_20190102_nfl_minill3_train.json"
DEV_DATA_PATH="/reddit_nfl_minill3nfl10_data_20220707/201809101112_20190102_nfl_minill3_test.json"

AUTHOR_LIST_PATH="/reddit_nfl_minill3nfl10_data_20220707/all_author_list.txt"

SET_NAME="\
${DATA_TYPE}_\
Epochs${NUM_TRAIN_EPOCHS}_\
batch${TRAIN_BATCH_SIZE}_\
Lr${LEARNING_RATE}_\
Trainpattern${TRAIN_CONVERT_PATTERN}_${CTX_LOOP_COUNT}_${RES_LOOP_COUNT}_${MAX_SLIDE_NUM}_${SRC_LENGTH}_${RES_LENGTH}_\
Semantic_${SEM_TYPE}_${SEM_LENGTH}"

LOAD_EPOCH=40
OUTPUT_DIR="Fine-Tuning/${SET_NAME}_loadepoch${LOAD_EPOCH}"
LOAD_CHECKPOINT="./Pre-training/minill3nfl10_data_Epochs40_batch32_Lr1e-5_Trainpattern${TRAIN_CONVERT_PATTERN}_${CTX_LOOP_COUNT}_${RES_LOOP_COUNT}_${MAX_SLIDE_NUM}_${SRC_LENGTH}_${RES_LENGTH}_Semantic_${SEM_TYPE}_${SEM_LENGTH}/checkpoint${LOAD_EPOCH}/bert.pt"

echo -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 Fine-Tuning/run_classifier.py \\
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \\
    --responses_tsv ${RESPONSES_TSV} \\
    --train_batch_size ${TRAIN_BATCH_SIZE} \\
    --learning_rate ${LEARNING_RATE} \\
    --bert_model ${BERT_MODEL} \\
    --train_convert_pattern ${TRAIN_CONVERT_PATTERN} \\
    --ctx_loop_count ${CTX_LOOP_COUNT} \\
    --res_loop_count ${RES_LOOP_COUNT} \\
    --max_slide_num ${MAX_SLIDE_NUM} \\
    --src_length ${SRC_LENGTH} \\
    --res_length ${RES_LENGTH} \\
    --topic_num ${TOPIC_NUM} \\
    --sem_type ${SEM_TYPE} \\
    --sem_length ${SEM_LENGTH} \\
    --eval_convert_pattern ${EVAL_CONVERT_PATTERN} \\
    --output_dir ${OUTPUT_DIR} \\
    --load_checkpoint ${LOAD_CHECKPOINT} \\
    --target_train_authors ${TARGET_TRAIN_AUTHORS} \\
    --target_dev_authors ${TARGET_DEV_AUTHORS} \\
    --train_data_path ${TRAIN_DATA_PATH} \\
    --dev_data_path ${DEV_DATA_PATH} \\
    --author_list_path ${AUTHOR_LIST_PATH} \\
    --score_file_path ${OUTPUT_DIR}/scores \\
    --model_type ${MODEL_TYPE} \\
    --use_train_input_history \\
    --use_train_res_history \\
    --use_train_utterance_embedding_latest \\
    --use_train_utterance_embedding_input_history \\
    --use_train_utterance_embedding_res_history \\
    --use_train_semantic_embedding_latest \\
    --use_train_semantic_embedding_input_history \\
    --use_train_semantic_embedding_res_history \\
    --use_train_author_embedding_latest \\
    --use_train_author_embedding_input_history \\
    --use_train_author_embedding_res_history \\
    --use_test_input_history \\
    --use_test_res_history \\
    --use_test_utterance_embedding_latest \\
    --use_test_utterance_embedding_input_history \\
    --use_test_utterance_embedding_res_history \\
    --use_test_semantic_embedding_latest \\
    --use_test_semantic_embedding_input_history \\
    --use_test_semantic_embedding_res_history \\
    --use_test_author_embedding_latest \\
    --use_test_author_embedding_input_history \\
    --use_test_author_embedding_res_history"
