# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, TransformerDecoderLayer, TransformerEncoderLayer
from transformers import BertForPreTraining, BertForSequenceClassification
from transformers import BertConfig as BertConfig

logger = logging.getLogger(__name__)

class UtteranceEmbedding(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.self_attention_1 = TransformerEncoderLayer(config.hidden_size,
                                        args.head_num,
                                        config.intermediate_size,
                                        config.hidden_dropout_prob,
                                        config.hidden_act,
                                        batch_first=True)

        self.topic_weights = nn.Parameter(torch.Tensor(np.random.normal(size=(args.topic_num, config.hidden_size))))
        self.topic_bias = nn.Parameter(torch.Tensor(np.zeros(args.topic_num)))
        self.topic_table = nn.Parameter(torch.Tensor(np.random.normal(size=(args.topic_num, config.hidden_size))))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs_embeds, input_utterance_ids, response_utterance_ids):
        input_zeros = torch.zeros(input_utterance_ids.shape, dtype=torch.int, device="cuda")
        res_zeros = torch.zeros(response_utterance_ids.shape, dtype=torch.int, device="cuda")
        input_utter_ids = torch.cat([input_utterance_ids, res_zeros[:,1:]], dim=1)
        res_utter_ids = torch.cat([input_zeros, response_utterance_ids[:,1:]], dim=1)

        batch = np.shape(inputs_embeds)[0]
        seq_length = np.shape(inputs_embeds)[1]
        width = np.shape(inputs_embeds)[2]

        hidden_tensor = self.self_attention_1(inputs_embeds)
        utter_ids = input_utter_ids 
        context_max = torch.max(utter_ids.reshape([-1]))
        topic_data = []
        # input
        for i in range(1, context_max + 1):
            utterance_mask = torch.eq(utter_ids, -i)
            utterance_mask = utterance_mask.unsqueeze(-1)
            utterance_mask = torch.tile(utterance_mask, [1, 1, width])

            token_mask = torch.eq(utter_ids, i)
            token_mask = token_mask.unsqueeze(-1)
            token_mask = torch.tile(token_mask, [1, 1, width])

            t_input = torch.where(token_mask, hidden_tensor, torch.zeros_like(hidden_tensor))
            mp, _ = torch.max(t_input, dim=1, keepdim=False)
            logits = torch.matmul(mp, self.topic_weights.permute(1, 0))
            logits = logits + self.topic_bias
            probs = self.softmax(logits)

            utterance_embedding = torch.matmul(probs, self.topic_table)
            utterance_embedding = utterance_embedding.unsqueeze(1)
            utterance_embedding = torch.tile(utterance_embedding, [1, seq_length, 1])

            token_mp = torch.where(token_mask, 
                                   utterance_embedding, 
                                   torch.zeros_like(utterance_embedding))
            utterance_embedding = torch.where(utterance_mask, 
                                              utterance_embedding, 
                                              torch.zeros_like(utterance_embedding))
            topic_data.append(token_mp + 2 * utterance_embedding)

        # responses
        utter_ids = res_utter_ids 
        utterance_mask = torch.eq(utter_ids, -1)
        utterance_mask = utterance_mask.unsqueeze(-1)
        utterance_mask = torch.tile(utterance_mask, [1, 1, width])

        token_mask = torch.eq(utter_ids, 1)
        token_mask = token_mask.unsqueeze(-1)
        token_mask = torch.tile(token_mask, [1, 1, width])

        t_input = torch.where(token_mask, hidden_tensor, torch.zeros_like(hidden_tensor))
        mp, _ = torch.max(t_input, dim=1, keepdim=False)
        logits = torch.matmul(mp, self.topic_weights.permute(1, 0))
        logits = logits + self.topic_bias
        probs = self.softmax(logits)

        utterance_embedding = torch.matmul(probs, self.topic_table)
        utterance_embedding = utterance_embedding.unsqueeze(1)
        utterance_embedding = torch.tile(utterance_embedding, [1, seq_length, 1])

        token_mp = torch.where(token_mask, 
                               utterance_embedding, 
                               torch.zeros_like(utterance_embedding))
        utterance_embedding = torch.where(utterance_mask, 
                                          utterance_embedding, 
                                          torch.zeros_like(utterance_embedding))
        topic_data.append(token_mp + 2 * utterance_embedding)

        topic_data = torch.stack(topic_data)
        topic_embedding = torch.sum(topic_data, dim=0)

        return inputs_embeds + topic_embedding

class RedditLMF(nn.Module):
    def __init__(self, args):    
        super().__init__()

        config = BertConfig.from_pretrained(args.bert_model)
        config.output_hidden_states = True

        self.bert_1 = BertForPreTraining.from_pretrained(args.bert_model, config=config)
        self.bert_1.cls.seq_relationship = nn.Linear(config.hidden_size, 2)

        #config = BertConfig.from_pretrained(args.bert_model, num_labels=1)
        #config.output_hidden_states = True

        #self.bert_1 = BertForSequenceClassification.from_pretrained(args.bert_model, config=config)
        #self.bert_2 = BertForSequenceClassification.from_pretrained(args.bert_model, config=config)

        word_embedding = self.bert_1.bert.embeddings.word_embeddings
        self.word_embedding_1 = copy.deepcopy(word_embedding)
        
        self.author_embedding_1 = nn.Embedding(args.author_list_len, config.hidden_size, padding_idx=0)
        self.author_embedding_1.weight.data.normal_(mean=0.0, std=config.initializer_range)

        self.utter_embedding = UtteranceEmbedding(args, config)
        
        layer = TransformerDecoderLayer(d_model=config.hidden_size,
                                        nhead=args.head_num,
                                        dim_feedforward=config.intermediate_size,
                                        dropout=config.hidden_dropout_prob,
                                        activation=config.hidden_act,
                                        batch_first=True)

        self.source_his_layer_1 = copy.deepcopy(layer)
        self.source_his_layer_2 = copy.deepcopy(layer)
        self.response_his_layer_1 = copy.deepcopy(layer)
        self.response_his_layer_2 = copy.deepcopy(layer)
 
        self.res_layers_1 = copy.deepcopy(layer)
        self.res_layers_2 = copy.deepcopy(layer)

        self.latest_semantic_layer_1 = copy.deepcopy(layer)
        self.source_semantic_layer_1 = copy.deepcopy(layer)
        self.response_semantic_layer_1 = copy.deepcopy(layer)

        self.norm = nn.LayerNorm(config.hidden_size)

    def history_context_direct(self, latest_hidden_states,
                        source_ids, source_attention_mask, source_token_type_ids, source_utterance_ids,
                        response_ids, response_attention_mask, response_token_type_ids, response_utterance_ids,
                        semantic_ids, semantic_attention_mask, semantic_token_type_ids,
                        source_author_ids, response_author_ids,
                        source_flag, use_utterance_embedding, use_semantic_embedding, use_author_embedding):

        history_list = [] 
        for index in range(source_ids.shape[1]):
            source_ids_n = source_ids[:,index,:]
            source_attention_mask_n = source_attention_mask[:,index,:]
            source_segment_ids_n = source_token_type_ids[:,index,:]
            source_utterance_ids_n = source_utterance_ids[:,index,:]

            response_ids_n = response_ids[:,index,:]
            response_attention_mask_n = response_attention_mask[:,index,:]
            response_segment_ids_n = response_token_type_ids[:,index,:]
            response_utterance_ids_n = response_utterance_ids[:,index,:]

            semantic_ids_n = semantic_ids[:,index,:]
            semantic_attention_mask_n = semantic_attention_mask[:,index,:]
            semantic_segment_ids_n = semantic_token_type_ids[:,index,:]

            source_author_ids_n = source_author_ids[:,index,:]
            response_author_ids_n = response_author_ids[:,index,:]

            token_ids = torch.cat([source_ids_n, response_ids_n[:,1:]], dim=1)
            attention_mask = torch.cat([source_attention_mask_n, response_attention_mask_n[:,1:]], dim=1)
            segment_ids = torch.cat([source_segment_ids_n, response_segment_ids_n[:,1:]], dim=1)
            author_ids = torch.cat([source_author_ids_n, response_author_ids_n[:,1:]], dim=1)

            input_embeds = self.word_embedding_1(token_ids)

            if use_utterance_embedding:
                input_embeds = self.utter_embedding(input_embeds, 
                                                    source_utterance_ids_n,
                                                    response_utterance_ids_n)
            if use_author_embedding:
                input_embeds = input_embeds + self.author_embedding_1(author_ids)

            outputs = self.bert_1(inputs_embeds=input_embeds,
                                  attention_mask=attention_mask,
                                  token_type_ids=segment_ids)
            hidden_states = outputs.hidden_states[-1]

            if use_semantic_embedding:
                semantic_embeds = self.word_embedding_1(semantic_ids_n)
                outputs = self.bert_1(inputs_embeds=semantic_embeds,
                                      attention_mask=semantic_attention_mask_n,
                                      token_type_ids=semantic_segment_ids_n)
                semantic_hidden_states = outputs.hidden_states[-1]

                if source_flag:
                    hidden_states = self.source_semantic_layer_1(hidden_states, semantic_hidden_states)
                else:
                    hidden_states = self.response_semantic_layer_1(hidden_states, semantic_hidden_states)

            if index == 0:
                h = hidden_states
            else:
                h = history_list[index - 1]

            if source_flag:
                # h_n = ST(h_n-1, c_n) + c_n
                history = self.source_his_layer_1(h, hidden_states)
                # h'_n = ST(h_n, c_N) + c_N
                history_dash = self.source_his_layer_2(history, latest_hidden_states)
            else:
                # h_n = ST(h_n-1, c_n) + c_n
                history = self.response_his_layer_1(h, hidden_states)
                # h'_n = ST(h_n, c_N) + c_N
                history_dash = self.response_his_layer_2(history, latest_hidden_states)

            history_list.append(history_dash)

        is_first = True
        for history in history_list:
            if not is_first:
                return_embedding = return_embedding + history
            else:
                return_embedding = history
                is_first = False
        return_embedding = self.norm(return_embedding)

        return return_embedding
    
    def history_context_incremental(self, latest_hidden_states,
                        source_ids, source_attention_mask, source_token_type_ids, source_utterance_ids,
                        response_ids, response_attention_mask, response_token_type_ids, response_utterance_ids,
                        semantic_ids, semantic_attention_mask, semantic_token_type_ids,
                        source_author_ids, response_author_ids,
                        source_flag, use_utterance_embedding, use_semantic_embedding, use_author_embedding):

        history_list = [] 
        for index in range(source_ids.shape[1]):
            source_ids_n = source_ids[:,index,:]
            source_attention_mask_n = source_attention_mask[:,index,:]
            source_segment_ids_n = source_token_type_ids[:,index,:]
            source_utterance_ids_n = source_utterance_ids[:,index,:]

            response_ids_n = response_ids[:,index,:]
            response_attention_mask_n = response_attention_mask[:,index,:]
            response_segment_ids_n = response_token_type_ids[:,index,:]
            response_utterance_ids_n = response_utterance_ids[:,index,:]

            semantic_ids_n = semantic_ids[:,index,:]
            semantic_attention_mask_n = semantic_attention_mask[:,index,:]
            semantic_segment_ids_n = semantic_token_type_ids[:,index,:]

            source_author_ids_n = source_author_ids[:,index,:]
            response_author_ids_n = response_author_ids[:,index,:]

            token_ids = torch.cat([source_ids_n, response_ids_n[:,1:]], dim=1)
            attention_mask = torch.cat([source_attention_mask_n, response_attention_mask_n[:,1:]], dim=1)
            segment_ids = torch.cat([source_segment_ids_n, response_segment_ids_n[:,1:]], dim=1)
            author_ids = torch.cat([source_author_ids_n, response_author_ids_n[:,1:]], dim=1)

            input_embeds = self.word_embedding_1(token_ids)

            if use_utterance_embedding:
                input_embeds = self.utter_embedding(input_embeds, 
                                                    source_utterance_ids_n,
                                                    response_utterance_ids_n)
            if use_author_embedding:
                input_embeds = input_embeds + self.author_embedding_1(author_ids)

            outputs = self.bert_1(inputs_embeds=input_embeds,
                                  attention_mask=attention_mask,
                                  token_type_ids=segment_ids)
            hidden_states = outputs.hidden_states[-1]

            if use_semantic_embedding:
                semantic_embeds = self.word_embedding_1(semantic_ids_n)
                outputs = self.bert_1(inputs_embeds=semantic_embeds,
                                      attention_mask=semantic_attention_mask_n,
                                      token_type_ids=semantic_segment_ids_n)
                semantic_hidden_states = outputs.hidden_states[-1]

                if source_flag:
                    hidden_states = self.source_semantic_layer_1(hidden_states, semantic_hidden_states)
                else:
                    hidden_states = self.response_semantic_layer_1(hidden_states, semantic_hidden_states)

            if index == 0:
                history_embeds = hidden_states
            else:
                pass

            if source_flag:
                history_embeds = self.source_his_layer_1(history_embeds, hidden_states)
            else:
                history_embeds = self.response_his_layer_1(history_embeds, hidden_states)

        return history_embeds
    
    def forward(self, labels,
                    src_input_ids, src_token_type_ids, src_attention_mask, src_token_uutr_ids,
                    res_input_ids, res_token_type_ids, res_attention_mask, res_token_uutr_ids,
                    sem_input_ids, sem_token_type_ids, sem_attention_mask, sem_token_uutr_ids,
                    res_his_input_ids, res_his_token_type_ids, res_his_attention_mask, res_his_token_uutr_ids,
                    res_his_res_input_ids, res_his_res_token_type_ids, res_his_res_attention_mask, res_his_res_token_uutr_ids,
                    res_his_sem_input_ids, res_his_sem_token_type_ids, res_his_sem_attention_mask, res_his_sem_token_uutr_ids, 
                    src_token_author_ids, res_token_author_ids, res_his_token_author_ids, res_his_res_token_author_ids,
                    args):
        '''
        semantics_ids: (batch_size, input_length, semantics_seq_length, 1)
        sources_ids: (batch_size, input_length, sources_seq_length, 1)
        response_ids: (batch_size, input_length, response_seq_length, 1)
        '''
        context_histories = []
        response_histories = []
        # batch_size = src_input_ids.shape[0]
        ctx_history_length = src_input_ids.shape[1]
        res_history_length = res_his_input_ids.shape[1]
        # 直近context + responseのembedding        
        response_input_ids = res_input_ids[:,-1,:]
        response_attention_mask = res_attention_mask[:,-1,:]
        response_segment_ids = res_token_type_ids[:,-1,:]
        response_utter_ids = res_token_uutr_ids[:,-1,:]

        source_input_ids = src_input_ids[:,-1,:]
        source_segment_ids = src_token_type_ids[:,-1,:]
        source_attention_mask = src_attention_mask[:,-1,:]
        source_utter_ids = src_token_uutr_ids[:,-1,:]

        semantic_input_ids = sem_input_ids[:,-1,:]
        semantic_attention_mask = sem_attention_mask[:,-1,:]
        semantic_segment_ids = sem_token_type_ids[:,-1,:]

        source_author_ids = src_token_author_ids[:,-1,:]
        response_author_ids = res_token_author_ids[:,-1,:]

        input_ids = torch.cat([source_input_ids, response_input_ids[:,1:]], dim=1)
        attention_mask = torch.cat([source_segment_ids, response_attention_mask[:,1:]], dim=1)
        segment_ids = torch.cat([source_attention_mask, response_segment_ids[:,1:]], dim=1)

        author_ids = torch.cat([source_author_ids, response_author_ids[:,1:]], dim=1)

        inputs_embeds = self.word_embedding_1(input_ids)

        if args.use_utterance_embedding_latest:
            inputs_embeds = self.utter_embedding(inputs_embeds, source_utter_ids, response_utter_ids)

        if args.use_author_embedding_latest:
            inputs_embeds = inputs_embeds + self.author_embedding_1(author_ids)

        outputs = self.bert_1(inputs_embeds=inputs_embeds,
                              attention_mask=attention_mask,
                              token_type_ids=segment_ids)
        latest_hidden_states = outputs.hidden_states[-1]

        if args.use_semantic_embedding_latest:
            semantic_embeds = self.word_embedding_1(semantic_input_ids)
            outputs = self.bert_1(inputs_embeds=semantic_embeds,
                                  attention_mask=semantic_attention_mask,
                                  token_type_ids=semantic_segment_ids)
            latest_semantic_hidden_states = outputs.hidden_states[-1]
            latest_hidden_states = self.latest_semantic_layer_1(latest_hidden_states, latest_semantic_hidden_states)

        if args.use_res_history:
            if args.res_loop_count == 1:
                res_his_input_ids = source_input_ids.unsqueeze(1)
                res_his_attention_mask = source_attention_mask.unsqueeze(1)
                res_his_token_type_ids = source_segment_ids.unsqueeze(1)
                res_his_token_uutr_ids = source_utter_ids.unsqueeze(1)

                res_his_res_input_ids = response_input_ids.unsqueeze(1)
                res_his_res_attention_mask = response_attention_mask.unsqueeze(1)
                res_his_res_token_type_ids = response_segment_ids.unsqueeze(1)
                res_his_res_token_uutr_ids = response_utter_ids.unsqueeze(1)

                res_his_sem_input_ids = semantic_input_ids.unsqueeze(1)
                res_his_sem_attention_mask = semantic_attention_mask.unsqueeze(1)
                res_his_sem_token_type_ids = semantic_segment_ids.unsqueeze(1)

                res_his_token_author_ids = source_author_ids.unsqueeze(1)
                res_his_res_token_author_ids = response_author_ids.unsqueeze(1)

            else:
                res_his_input_ids = torch.cat([res_his_input_ids, source_input_ids.unsqueeze(1)], dim=1)
                res_his_attention_mask = torch.cat([res_his_attention_mask, source_attention_mask.unsqueeze(1)], dim=1)
                res_his_token_type_ids = torch.cat([res_his_token_type_ids, source_segment_ids.unsqueeze(1)], dim=1)
                res_his_token_uutr_ids = torch.cat([res_his_token_uutr_ids, source_utter_ids.unsqueeze(1)], dim=1)
    
                res_his_res_input_ids = torch.cat([res_his_res_input_ids, response_input_ids.unsqueeze(1)], dim=1)
                res_his_res_attention_mask = torch.cat([res_his_res_attention_mask, response_attention_mask.unsqueeze(1)], dim=1)
                res_his_res_token_type_ids = torch.cat([res_his_res_token_type_ids, response_segment_ids.unsqueeze(1)], dim=1)
                res_his_res_token_uutr_ids = torch.cat([res_his_res_token_uutr_ids, response_utter_ids.unsqueeze(1)], dim=1)
    
                res_his_sem_input_ids = torch.cat([res_his_sem_input_ids, semantic_input_ids.unsqueeze(1)], dim=1)
                res_his_sem_attention_mask = torch.cat([res_his_sem_attention_mask, semantic_attention_mask.unsqueeze(1)], dim=1)
                res_his_sem_token_type_ids = torch.cat([res_his_sem_token_type_ids, semantic_segment_ids.unsqueeze(1)], dim=1)
    
                res_his_token_author_ids = torch.cat([res_his_token_author_ids, source_author_ids.unsqueeze(1)], dim=1)
                res_his_res_token_author_ids = torch.cat([res_his_res_token_author_ids, response_author_ids.unsqueeze(1)], dim=1)

            if args.model_type == 0:
                response_context_history_embedding = self.history_context_direct(
                        latest_hidden_states,
                        res_his_input_ids, res_his_attention_mask, res_his_token_type_ids, res_his_token_uutr_ids,
                        res_his_res_input_ids, res_his_res_attention_mask, res_his_res_token_type_ids, res_his_res_token_uutr_ids,
                        res_his_sem_input_ids, res_his_sem_attention_mask, res_his_sem_token_type_ids,
                        res_his_token_author_ids, res_his_res_token_author_ids,
                        False,
                        args.use_utterance_embedding_res_history,
                        args.use_semantic_embedding_res_history,
                        args.use_author_embedding_res_history)
            elif args.model_type == 1:
                response_context_history_embedding = self.history_context_incremental(
                        latest_hidden_states,
                        res_his_input_ids, res_his_attention_mask, res_his_token_type_ids, res_his_token_uutr_ids,
                        res_his_res_input_ids, res_his_res_attention_mask, res_his_res_token_type_ids, res_his_res_token_uutr_ids,
                        res_his_sem_input_ids, res_his_sem_attention_mask, res_his_sem_token_type_ids,
                        res_his_token_author_ids, res_his_res_token_author_ids,
                        False,
                        args.use_utterance_embedding_res_history,
                        args.use_semantic_embedding_res_history,
                        args.use_author_embedding_res_history)

        if args.use_input_history:
            if args.model_type == 0:
                source_context_history_embedding = self.history_context_direct(
                        latest_hidden_states,
                        src_input_ids, src_attention_mask, src_token_type_ids, src_token_uutr_ids,
                        res_input_ids, res_attention_mask, res_token_type_ids, res_token_uutr_ids,
                        sem_input_ids, sem_attention_mask, sem_token_type_ids,
                        src_token_author_ids, res_token_author_ids,
                        True,
                        args.use_utterance_embedding_input_history,
                        args.use_semantic_embedding_input_history,
                        args.use_author_embedding_input_history)
            elif args.model_type == 1:
                source_context_history_embedding = self.history_context_incremental(
                        latest_hidden_states,
                        src_input_ids, src_attention_mask, src_token_type_ids, src_token_uutr_ids,
                        res_input_ids, res_attention_mask, res_token_type_ids, res_token_uutr_ids,
                        sem_input_ids, sem_attention_mask, sem_token_type_ids,
                        src_token_author_ids, res_token_author_ids,
                        True,
                        args.use_utterance_embedding_input_history,
                        args.use_semantic_embedding_input_history,
                        args.use_author_embedding_input_history)

        all_embedding = latest_hidden_states
        if args.use_res_history:
            all_embedding = self.res_layers_1(all_embedding, response_context_history_embedding)
        if args.use_input_history:
            all_embedding = self.res_layers_2(all_embedding, source_context_history_embedding)
        
        outputs = self.bert_1(inputs_embeds=all_embedding)

        prediction_logits = outputs.prediction_logits
        seq_relationship_logits = outputs.seq_relationship_logits

        return prediction_logits, seq_relationship_logits

#        outputs = self.bert_2(inputs_embeds=all_embedding)
#
#        loss = outputs.loss
#        logits = outputs.logits
#
#        return logits, loss
