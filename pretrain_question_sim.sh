TASKS="semeval"
TYPES="b_r bs_r"
# TYPES="s_r"
# MODELS="roberta-base-nli-mean-tokens bert-base-nli-mean-tokens"
# NON_PRETRAINED_MODELS="roberta-base bert-base-uncased"
MODELS="roberta-base-nli-mean-tokens bert-base-nli-mean-tokens"
NON_PRETRAINED_MODELS="roberta-base bert-base-uncased"
SRUN_PARAM="--gres gpu:volta:1 --nodes 1 --ntasks-per-node 1 --cpus-per-task 8 -C volta32gb --partition learnfair --time 4000 --mem-per-cpu 7G"
num_epoch=10
seq_length=128
for TASK in $TASKS
do
    for TYPE in $TYPES
    do
        for MODEL in $NON_PRETRAINED_MODELS
        do
            echo ${TASK}_${TYPE}_${num_epoch}_${MODEL}
            srun --job-name ${TASK}_${TYPE}_${num_epoch}_${MODEL} --output /checkpoint/xiaojianwu/sentBert/${TASK}/${TASK}_${TYPE}_${num_epoch}_${MODEL}_train.log \
                --error /checkpoint/xiaojianwu/sentBert/${TASK}/${TASK}_${TYPE}_${num_epoch}_${MODEL}_train.stderr \
                ${SRUN_PARAM} \
                python examples/training_transformers/training_questionsim_continue_training.py \
                --task-name ${TASK} --type-name ${TYPE} --num-epoches ${num_epoch} \
                --model-name ${MODEL} --base-folder-path /checkpoint/xiaojianwu/sentBert/${TASK} \
                --model-length ${seq_length} &
        done

        for MODEL in $MODELS
        do
            echo ${TASK}_${TYPE}_${num_epoch}_${MODEL}
            srun --job-name ${TASK}_${TYPE}_${num_epoch}_${MODEL} --output /checkpoint/xiaojianwu/sentBert/${TASK}/${TASK}_${TYPE}_${num_epoch}_${MODEL}_train.log \
                --error /checkpoint/xiaojianwu/sentBert/${TASK}/${TASK}_${TYPE}_${num_epoch}_${MODEL}_train.stderr \
                ${SRUN_PARAM} \
                python examples/training_transformers/training_questionsim_continue_training.py \
                --task-name ${TASK} --type-name ${TYPE} --num-epoches ${num_epoch} \
                --pretrained --model-name ${MODEL} --base-folder-path /checkpoint/xiaojianwu/sentBert/${TASK} \
                --model-length ${seq_length} &
        done
    done
done  