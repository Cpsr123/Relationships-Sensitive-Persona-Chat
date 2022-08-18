# 環境構築
```
sudo nvidia-docker run -it -d --name LRF-HAIS-reddit \
              nvidia/cuda:11.0.3-devel-ubuntu20.04 \
             /bin/bash

apt update -y
apt upgrade -y
apt dist-upgrade -y
apt autoremove -y
apt autoclean -y

apt install -y python3 python3-pip vim wget unzip git
pip3 install --upgrade pip
pip3 install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install tqdm==4.62.3 \
             pytorch-pretrained-bert==0.6.2 \
             pytorch-transformers==1.2.0 \
             numpy==1.21.4 \
             sklearn==0.0 \
             matplotlib==3.5.1 \
             transformers==4.16.2


日本語環境設定
apt install -y tzdata && \
    ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
以下選択
6. Asia
	79. Tokyo

apt install -y language-pack-ja-base language-pack-ja ibus-mozc
以下選択
55. Japanese
	1. Japanese

echo 'export LANG=ja_JP.UTF-8' | tee -a ~/.bashrc
echo 'export LANGUAGE=ja_JP:ja' | tee -a ~/.bashrc
```

# 実行コマンド(Pre-training)
```
CUDA_VISIBLE_DEVICES=0 python3 Pre-training/run_classifier.py \
    --num_train_epochs 10 \
    --train_batch_size 32 \
    --learning_rate 1e-5 \
    --bert_model bert-base-uncased \
    --train_convert_pattern 1 \
    --ctx_loop_count 10 \
    --res_loop_count 3 \
    --max_slide_num 15 \
    --src_length 50 \
    --res_length 30 \
    --output_dir Pre-training/training \
    --target_train_authors /reddit_nfl_data/author_list_min20_201809101112_20190102_nfl_minill3_train.txt \
    --train_data_path /reddit_nfl_data/201809101112_20190102_nfl_minill3_train.json \
    --author_list_path /reddit_nfl_data/author_list_all_201809101112_20190102_nfl_minill3.txt \
    --use_input_history \
    --use_res_history
```
# 実行コマンド(Fine-Tuning)
```
CUDA_VISIBLE_DEVICES=0 python3 Fine-Tuning/run_classifier.py \
    --num_train_epochs 5 \
    --responses_tsv /reddit_nfl_data/201809101112_20190102_nfl_minill3_test_responses.tsv\
    --train_batch_size 32 \
    --learning_rate 1e-5 \
    --bert_model bert-base-uncased \
    --train_convert_pattern 1 \
    --ctx_loop_count 10 \
    --res_loop_count 3 \
    --max_slide_num 15 \
    --src_length 50 \
    --res_length 30 \
    --eval_convert_pattern 4 \
    --output_dir Fine-Tuning/fine-tuning \
    --load_checkpoint ./Pre-training/training/checkpoint10/bert.pt \
    --target_train_authors /reddit_nfl_data/author_list_min20_201809101112_20190102_nfl_minill3_train.txt \
    --target_dev_authors /reddit_nfl_data/author_list_min20_201809101112_20190102_nfl_minill3_test.txt \
    --train_data_path /reddit_nfl_data/201809101112_20190102_nfl_minill3_train.json \
    --dev_data_path /reddit_nfl_data/201809101112_20190102_nfl_minill3_test.json \
    --author_list_path /reddit_nfl_data/author_list_all_201809101112_20190102_nfl_minill3.txt \
    --score_file_path Fine-Tuning/training/scores \
    --use_train_input_history \
    --use_train_res_history \
    --use_test_input_history \
    --use_test_res_history
```
