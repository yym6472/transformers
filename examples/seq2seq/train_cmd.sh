export DATA_DIR=/home/disk2/lyj2019/workspace/my_paper/dataset/SAMsum/finished_src_tgt_datas
CUDA_VISIBLE_DEVICES=1 ./finetune.sh \
    --data_dir $DATA_DIR \
    --train_batch_size=1 \
    --eval_batch_size=1 \
    --output_dir=SAMsum_results \
    --num_train_epochs 6 \
    --model_name_or_path facebook/bart-base