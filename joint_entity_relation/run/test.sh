#!/bin/bash
./lstmparse -T ../data/train.parser -d ../data/test.parser -s list-graph --data_type text --pretrained_dim 200 --hidden_dim 20 --bilstm_hidden_dim 10 --lstm_input_dim 10 --input_dim 10 --action_dim 50 --pos_dim 50 --rel_dim 50 -w ../data/glove.6B.200d.txt --dynet_mem 2048m --max_itr 2000 -P --is_cycle -m model_name
