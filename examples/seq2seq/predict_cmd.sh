export DATA_DIR=/home/disk2/lyj2019/workspace/my_paper/dataset/SAMsum/finished_src_tgt_datas
CUDA_VISIBLE_DEVICES=1 ./run_eval.py ./SAMsum_results/best_tfmr/ $DATA_DIR/test.source dbart_test_generations.txt \
    --reference_path $DATA_DIR/test.target \
    --score_path cnn_rouge.json \
    --task summarization \
    --n_obs -1 \
    --device cuda \
    --max_source_length 1024 \
    --max_target_length 56 \
    --fp16 \
    --bs 32