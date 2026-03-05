python train_link_prediction.py --dataset_name wikipedia --model_name DyDNet --patch_size 1 --max_input_sequence_length 64 --num_runs 5 --gpu 0

python train_link_prediction.py --dataset_name uci --model_name DyDNet --load_best_configs --num_runs 5 --gpu 0

python evaluate_link_prediction.py --dataset_name wikipedia --model_name DyDNet --patch_size 1 --max_input_sequence_length 64 --num_runs 5 --gpu 0 --negative_sample_strategy historical

python evaluate_link_prediction.py --dataset_name uci --model_name DyDNet --load_best_configs --num_runs 5 --gpu 0 --negative_sample_strategy historical
 





