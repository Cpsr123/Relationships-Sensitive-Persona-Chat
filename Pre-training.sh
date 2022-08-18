#!/bin/bash

CUDA_VISIBLE_DEVICES=0

NUM_TRAIN_EPOCHS=40

DATA_TYPE="minill3nfl10_data"

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

TARGET_TRAIN_AUTHORS="/reddit_nfl_minill3nfl10_data_20220707/author_list_min20_201809101112_20190102_nfl_minill3_train_top20.txt"
TRAIN_DATA_PATH="/reddit_nfl_minill3nfl10_data_20220707/201809101112_20190102_nfl_minill3_train.json"
AUTHOR_LIST_PATH="/reddit_nfl_minill3nfl10_data_20220707/all_author_list.txt"

SET_NAME="\
${DATA_TYPE}_\
Epochs${NUM_TRAIN_EPOCHS}_\
batch${TRAIN_BATCH_SIZE}_\
Lr${LEARNING_RATE}_\
Trainpattern${TRAIN_CONVERT_PATTERN}_${CTX_LOOP_COUNT}_${RES_LOOP_COUNT}_${MAX_SLIDE_NUM}_${SRC_LENGTH}_${RES_LENGTH}_\
Semantic_${SEM_TYPE}_${SEM_LENGTH}"


OUTPUT_DIR="Pre-training/${SET_NAME}"

echo -e "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 Pre-training/run_classifier.py \\
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \\
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
    --output_dir ${OUTPUT_DIR} \\
    --target_train_authors ${TARGET_TRAIN_AUTHORS} \\
    --train_data_path ${TRAIN_DATA_PATH} \\
    --author_list_path ${AUTHOR_LIST_PATH} \\
    --model_type ${MODEL_TYPE} \\
    --use_input_history \\
    --use_res_history \\
    --use_utterance_embedding_latest \\
    --use_utterance_embedding_input_history \\
    --use_utterance_embedding_res_history \\
    --use_semantic_embedding_latest \\
    --use_semantic_embedding_input_history \\
    --use_semantic_embedding_res_history \\
    --use_author_embedding_latest \\
    --use_author_embedding_input_history \\
    --use_author_embedding_res_history "
