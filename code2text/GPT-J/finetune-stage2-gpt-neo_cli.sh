DATASET_NAME="Explanations"

JOB_NAME="ft-stage2-gpt-neo"
# JOB_NAME="ft-stage2-gpt2"

MODEL="/mnt/default/CheckpointsGithubCodeDoc-Jan10-12-50-2022/checkpoint-31000"
# MODEL="gpt2"
BLOCK_SIZE=512
BATCH_SIZE=1

while getopts :j:m:s:b:h flag
do
    case "${flag}" in
        j) JOB_NAME=${OPTARG};;
        m) MODEL=${OPTARG};;
        s) BLOCK_SIZE=${OPTARG};;
        b) BATCH_SIZE=${OPTARG};;
        h) # help
            echo "Usage: $0 [-j JOB_NAME] [-m MODEL] [-s BLOCK_SIZE] [-b BATCH_SIZE] NPROC"
            echo "  -j JOB_NAME: Job name (default: $JOB_NAME)"
            echo "  -m MODEL: Model name (default: $MODEL)"
            echo "  -s BLOCK_SIZE: Block size (default: $BLOCK_SIZE)"
            echo "  -b BATCH_SIZE: Batch size (default: $BATCH_SIZE)"
            echo "  NPROC: Number of processes"
            exit 0;;
        \?) # invalid flag
            echo "Invalid option: -${OPTARG}" >&2
            echo "use -h for help"
            exit 1;;
    esac
done
NPROC=${@:$OPTIND:1}

echo "setting JOB_NAME=$JOB_NAME, MODEL=$MODEL, BLOCK_SIZE=$BLOCK_SIZE, BATCH_SIZE=$BATCH_SIZE, NPROC=$NPROC"


python -m torch.distributed.launch \
    --nproc_per_node $NPROC finetune-gpt-neo.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET_NAME \
    --validation_split_percentage 10 \
    --block_size $BLOCK_SIZE \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $((BATCH_SIZE*2)) \
    --evaluation_strategy steps \
    --eval_steps 250 \
    --do_train \
    --do_eval \
    --output_dir /mnt/default/$JOB_NAME-${DATASET_NAME}-$(date +%b%d-%H-%M-%Y)/ \
    --overwrite_output_dir \
    --logging_steps 50 \
    --logging_dir $AMLT_OUTPUT_DIR/logs \
    --save_steps 250 \
    --report_to tensorboard \
    --run_name $JOB_NAME-$DATASET_NAME-$(date +%b%d-%H-%M-%Y)
    # --fp16 \
    # --fp16_opt_level O3 \

# python -m torch.distributed.launch \
#     --nproc_per_node 4 finetune-gpt-neo.py \
#     --model_name_or_path EleutherAI/gpt-neo-1.3B \
#     --dataset_name Expanations \
#     --validation_split_percentage 5 \
#     --max_train_samples 10000\
#     --max_eval_samples 1000\
#     --block_size 256 \
#     --per_device_train_batch_size 2 \
#     --per_device_eval_batch_size 2 \
#     --evaluation_strategy steps \
#     --eval_steps 50 \
#     --do_train \
#     --do_eval \
#     --output_dir /mnt/default/checkpoints1020-fp32/ \
#     --overwrite_output_dir \
#     --logging_steps 50 \
#     --logging_dir $AMLT_OUTPUT_DIR/logs \
#     # --fp16 \
#     # --fp16_opt_level O3 \
#     --save_steps 50