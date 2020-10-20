# #!/bin/bash

export DATA_DIR=/home/disk2/lyj2019/workspace/my_paper/dataset/SAMsum/finished_src_tgt_datas

for dir in `ls`
do
    if [ ${dir:0:6} == "SAMsum" ];
    then
        echo "Start prediction for $dir"
        CUDA_VISIBLE_DEVICES=1 ./run_eval.py ./$dir/best_tfmr/ $DATA_DIR/test.source ./$dir/test_generations.txt \
        --reference_path $DATA_DIR/test.target \
        --score_path ./$dir/rouge.json \
        --task summarization \
        --n_obs -1 \
        --device cuda \
        --max_source_length 1024 \
        --max_target_length 56 \
        --fp16 \
        --bs 32
        echo "Finished prediction for $dir"
    fi
done