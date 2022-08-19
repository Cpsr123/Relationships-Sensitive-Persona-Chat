# Reddit dataset (NFL)
Download the dataset from the following dropbox:  
https://www.dropbox.com/s/vp3n4zqhyshx8er/reddit_nfl_data.zip?dl=0
```
201809101112_20190102_nfl_minill3_train.json % The training dataset
201809101112_20190102_nfl_minill3_test.json % The test dataset
201809101112_20190102_nfl_minill3_test_responses.tsv % Response candidate set (positive1:negative9)
author_list_all_201809101112_20190102_nfl_minill3.txt % All speaker IDs
author_list_min20_201809101112_20190102_nfl_minill3_train.txt % Target training speaker IDs
author_list_min20_201809101112_20190102_nfl_minill3_test.txt % Target test speaker IDs
```

# Json format
The example of the format of the dataset is as below:
```
{"author_name":
    [
        {
            "Example": "1789",   % ID
            "author/0": "Raiichu_LoL",　% Speaker of the previous utterance of the current utterance
            "context_author": "Lee1100",　% Speaker of the current utterance
            "context_created_utc": "1535770782", 
            "created_utc/0": "1535770504",　
            "response_author": "OrangeAndBlack",　% Responder
            "response_created_utc": "1535804229",
            "selftext": "",
            "subreddit": "nfl",
            "thread_id": "9bzqj9",
            "title": "[Trotter] Can’t believe the Raiders would be crazy enough to seriously consider trading Mack, but I’m hearing offers are coming in and they’re weighing them. Strange considering people in org call Mack their best player. Not best defensive player. But player. Period."  
            "context/0": "This feels like Odell all over again. The regular season needs to start ASAP.",  % The previous utterance of the current utterance
            "Context": "It's impossible at the moment, but can you imagine him on the Eagles with their D-Line?",   % Current utterance
            "Response": "I can, but I want to be able to afford Wentz in 3 years..."
        }
    ]
}
```

# Setup
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
```

# Pre-training
```
CUDA_VISIBLE_DEVICES=0 python3 Pre-training/run_classifier.py \
    --num_train_epochs 10 \
    --output_dir Pre-training/training \
    --target_train_authors /reddit_nfl_data/author_list_min20_201809101112_20190102_nfl_minill3_train.txt \
    --train_data_path /reddit_nfl_data/201809101112_20190102_nfl_minill3_train.json \
    --author_list_path /reddit_nfl_data/author_list_all_201809101112_20190102_nfl_minill3.txt \
    --use_input_history \
    --use_res_history
```
# Fine-Tuning
```
CUDA_VISIBLE_DEVICES=0 python3 Fine-Tuning/run_classifier.py \
    --num_train_epochs 5 \
    --responses_tsv /reddit_nfl_data/201809101112_20190102_nfl_minill3_test_responses.tsv\
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

# Reproduce the results of our Model (RSPC) for NFL dataset.  
Download the checkpoint file for RSPC(kc=10,kr=10,l=15) from the following URL and place it under the checkpoint directory (it is the current directory in the foloowing command):  
https://www.dropbox.com/s/ggx85eyqy3oykbi/rspc_bert.pt?dl=0

```
CUDA_VISIBLE_DEVICES=0 python3 Fine-Tuning/run_classifier.py \
    --responses_tsv /reddit_nfl_data/201809101112_20190102_nfl_minill3_test_responses.tsv\
    --output_dir Fine-Tuning/fine-tuning \
    --load_checkpoint ./rspc_bert.pt \
    --target_train_authors /reddit_nfl_data/author_list_min20_201809101112_20190102_nfl_minill3_train.txt \
    --target_dev_authors /reddit_nfl_data/author_list_min20_201809101112_20190102_nfl_minill3_test.txt \
    --train_data_path /reddit_nfl_data/201809101112_20190102_nfl_minill3_train.json \
    --dev_data_path /reddit_nfl_data/201809101112_20190102_nfl_minill3_test.json \
    --author_list_path /reddit_nfl_data/author_list_all_201809101112_20190102_nfl_minill3.txt \
    --score_file_path Fine-Tuning/training/scores \
    --use_train_input_history \
    --use_train_res_history \
    --use_test_input_history \
    --use_test_res_history \
    --do_train False
```
