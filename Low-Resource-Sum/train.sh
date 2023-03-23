#!/bin/bash
{
python main.py --save_model 1 --do_train --seed 21 --data_name cnn_dailymail
python main.py --save_model 1 --do_test --seed 21 --data_name cnn_dailymail
python main.py --save_model 1 --do_test --seed 21 --data_name cnn_dailymail --num_data 10
python main.py --save_model 1 --do_test --seed 21 --data_name cnn_dailymail --num_data 100
}