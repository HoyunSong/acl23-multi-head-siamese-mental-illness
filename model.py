import numpy as np
import argparse, time
import os

import torch

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import json

from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, BertModel

from utils import generate_examples


class DepressionQuestionModel(nn.Module):
    def __init__(self, args, device, tokenizer, bert_model, sent_model=None):
        super(DepressionQuestionModel, self).__init__()

        self.args = args
        self.device = device
        
        self.bert_model = bert_model
        if sent_model:
            self.sent_model = sent_model
        else:
            self.sent_model = bert_model
        

        # prepare questions
        with open(self.args.question_path.format(self.args.task_name), 'r') as fp:
            questions = json.load(fp)

        sym_num = len(questions)
        q_num = sum([len(val) for key, val in questions.items()])
        cached_question_file = os.path.join(
            args.cache_dir if args.cache_dir is not None else args.data_dir,
            "cached_question_{}_{}_{}_{}".format(
                self.sent_model.__class__.__name__,
                str(sym_num),
                str(q_num),
                args.task_name,
            ),
        )

        if os.path.exists(cached_question_file):
            print("*** Loading questions from cached file {}".format(cached_question_file))
            self.question_embeddings = torch.load(cached_question_file)
            self.question_embeddings = [torch.squeeze(torch.stack(value), 1).to(self.device) for key, value in self.question_embeddings.items()]

        else:
            print("*** Saving questions into cached file {}".format(cached_question_file))
            self.question_embeddings = {key: [self.sent_model(**tokenizer.encode_plus(
                                                s, max_length=args.max_seq_length,
                                                padding='max_length',
                                                truncation='longest_first',
                                                return_tensors="pt",)).last_hidden_state for s in value]
                                            for key, value in questions.items()}
            torch.save(self.question_embeddings, cached_question_file)
            self.question_embeddings = [torch.squeeze(torch.stack(value), 1).to(self.device) for key, value in self.question_embeddings.items()]

        # Conv layer 1
        self.conv1 = nn.Conv1d(self.args.max_seq_length, self.args.hiddne_size1, 5, stride=1)
        self.mp1 = nn.MaxPool1d(2)

        self.conv1_2 = nn.Conv1d(self.args.max_seq_length, self.args.hiddne_size1, 2, stride=1)
        self.mp1_2 = nn.MaxPool1d(2)

        self.conv1_3 = nn.Conv1d(self.args.max_seq_length, self.args.hiddne_size1, 3, stride=1)
        self.mp1_3 = nn.MaxPool1d(2)

        
        # Conv layer 2
        self.conv2 = nn.Conv1d(self.args.hiddne_size1, 1, 5, stride=1)
        self.mp2 = nn.MaxPool1d(2)

        self.conv2_2 = nn.Conv1d(self.args.hiddne_size1, 1, 2, stride=1)
        self.mp2_2 = nn.MaxPool1d(2)

        self.conv2_3 = nn.Conv1d(self.args.hiddne_size1, 1, 3, stride=1)
        self.mp2_3 = nn.MaxPool1d(2)
        


        self.sym_layer = torch.nn.Linear(sym_num, self.args.num_labels)
        
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()



    def forward(self, inputs, labels):

        input1 = self.bert_model(**inputs).last_hidden_state
        input2s = self.question_embeddings

        distances = []

        for sym_id, input2 in enumerate(input2s):

            # Conv layer 1 for input 1
            i1 = self.conv1(input1)
            i1 = self.mp1(i1)

            
            # Conv layer 1 for input 2
            i2 = self.conv1(input2)
            i2 = self.mp1(i2)


            #############################################
            # Conv layer 1 for input 1
            i1_3 = self.conv1_3(input1)
            i1_3 = self.mp1_3(i1)

            
            # Conv layer 1 for input 2
            i2_3 = self.conv1_3(input2)
            i2_3 = self.mp1_3(i2)

            # Conv layer 1 for input 1
            i1_2 = self.conv1_2(input1)
            i1_2 = self.mp1_2(i1)

            
            # Conv layer 1 for input 2
            i2_2 = self.conv1_2(input2)
            i2_2 = self.mp1_2(i2)
            #############################################


            # Conv layer 2 for input 1
            i1 = self.conv2(i1)
            embed1 = self.mp2(i1)

            
            # Conv layer 2 for input 2
            i2 = self.conv2(i2)
            embed2 = self.mp2(i2)


            #############################################
            # Conv layer 2 for input 1
            i1_2 = self.conv2_2(i1_2)
            embed1_2 = self.mp2_2(i1_2)

            
            # Conv layer 2 for input 2
            i2_2 = self.conv2_2(i2_2)
            embed2_2 = self.mp2_2(i2_2)

            # Conv layer 2 for input 1
            i1_3 = self.conv2(i1_3)
            embed1_3 = self.mp2(i1_3)

            
            # Conv layer 2 for input 2
            i2_3 = self.conv2(i2_3)
            embed2_3 = self.mp2(i2_3)
            #############################################


            input_num = embed1.size()[0]
            q_num = embed2.size()[0]

            # get embeddings for input 1
            embed1 = torch.unsqueeze(embed1, 1)
            embed1 = embed1.expand(-1, q_num, -1, -1)
            embed1 = torch.squeeze(embed1, 2)

            
            # get embeddings for input 2
            embed2 = torch.unsqueeze(embed2,0)
            embed2 = embed2.expand(input_num, -1, -1, -1)
            embed2 = torch.squeeze(embed2, 2)


            #############################################
            # get embeddings for input 1
            embed1_2 = torch.unsqueeze(embed1_2, 1)
            embed1_2 = embed1_2.expand(-1, q_num, -1, -1)
            embed1_2 = torch.squeeze(embed1_2, 2)

            
            # get embeddings for input 2
            embed2_2 = torch.unsqueeze(embed2_2,0)
            embed2_2 = embed2_2.expand(input_num, -1, -1, -1)
            embed2_2 = torch.squeeze(embed2_2, 2)

            # get embeddings for input 1
            embed1_3 = torch.unsqueeze(embed1_3, 1)
            embed1_3 = embed1_3.expand(-1, q_num, -1, -1)
            embed1_3 = torch.squeeze(embed1_3, 2)

            
            # get embeddings for input 2
            embed2_3 = torch.unsqueeze(embed2_3,0)
            embed2_3 = embed2_3.expand(input_num, -1, -1, -1)
            embed2_3 = torch.squeeze(embed2_3, 2)
            #############################################

            cos = nn.CosineSimilarity(dim=2, eps=self.args.eps)
            distance = cos(embed1, embed2)
            distance_2 = cos(embed1_2, embed2_2)
            distance_3 = cos(embed1_3, embed2_3)

            d_5 = torch.mean(distance,1)
            d_2 = torch.mean(distance_2,1)
            d_3 = torch.mean(distance_3,1)

            dd = torch.stack([d_2, d_3, d_5], -1)

            d = torch.mean(dd, 1)
            if input_num==1:
                d = torch.unsqueeze(d,0)

            distances.append(d)
        
        
        distance = torch.stack(distances, -1)
        distance = torch.squeeze(distance,1)

        
        out = self.sym_layer(distance)
        if input_num!=1:
            out = torch.squeeze(out)
        

        out = self.softmax(out)

        loss = self.loss(out, labels)

        return [loss, out, distance]


        


if __name__ == '__main__':
    from train import get_args
    args = get_args()
    os.environ["CUDA_VISIBLE_DEVICES"]= '1'
    model = DepressionQuestionModel(
                args=args,
                tokenizer=AutoTokenizer.from_pretrained(
                    args.model_name_or_path,
                    cache_dir=args.cache_dir,
                ),
                bert_model=BertModel.from_pretrained(
                    args.model_name_or_path, 
                    cache_dir=args.cache_dir,
                    num_labels=args.num_labels,
                ),
                sent_model=SentenceTransformer(args.sent_model_name_or_path),
            )
    import IPython; IPython.embed(); exit(1)

    
