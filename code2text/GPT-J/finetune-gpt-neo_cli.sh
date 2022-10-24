# ========== SET DATASET ==========
DATASET_NAME="GithubCodeDocStep1n2Filtered"
SPLIT_PERC=5
LOG_STEP=250
SAVE_STEP=500

# DATASET_NAME="GithubCodeDoc"
# SPLIT_PERC=1
# LOG_STEP=1250
# SAVE_STEP=2500

# ======= SET JOB & MODEL NAME =======
# Here are some job and model examples, prefer to set in yaml:

# JOB_NAME="ft-stage1-gpt-neo"
# MODEL="EleutherAI/gpt-neo-1.3B"

# JOB_NAME="ft-stage1-gpt-neo-27"
# MODEL="EleutherAI/gpt-neo-2.7B"

# JOB_NAME="ft-stage1-gpt-j"
# MODEL="EleutherAI/gpt-j-6B"

# JOB_NAME="ft-stage1-gpt2"
# MODEL="gpt2"

# JOB_NAME="ft-stage2-gpt-neo"
# MODEL="/mnt/default/ft-stage1-gpt-neo-GithubCodeDoc-Jan16-16-49-2022/checkpoint-80000"

BLOCK_SIZE=2048
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
# ensure JOB_NAME, MODEL and NPROC are set
if [ -z "$JOB_NAME" ]; then
    echo "JOB_NAME is not set"
    exit 1
fi
if [ -z "$MODEL" ]; then
    echo "MODEL is not set"
    exit 1
fi
if [ -z "$NPROC" ]; then
    echo "NPROC is not set"
    exit 1
fi

echo "setting JOB_NAME=$JOB_NAME, MODEL=$MODEL, BLOCK_SIZE=$BLOCK_SIZE, BATCH_SIZE=$BATCH_SIZE, NPROC=$NPROC"

# python -m torch.distributed.launch \
#     --nproc_per_node $NPROC --nnodes=2 --node_rank=$OMPI_COMM_WORLD_RANK finetune-gpt-neo.py \
deepspeed finetune-gpt-neo.py \
    --model_name_or_path $MODEL \
    --dataset_name $DATASET_NAME \
    --validation_split_percentage $SPLIT_PERC \
    --block_size $BLOCK_SIZE \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $((BATCH_SIZE*2)) \
    --evaluation_strategy steps \
    --eval_steps $SAVE_STEP \
    --do_train \
    --do_eval \
    --output_dir /mnt/default/$JOB_NAME-${DATASET_NAME}-$(date +%b%d-%H-%M-%Y)/ \
    --overwrite_output_dir \
    --logging_steps $LOG_STEP \
    --logging_dir $AMLT_OUTPUT_DIR/logs \
    --save_steps $SAVE_STEP \
    --num_train_epochs 3 \
    --deepspeed ds_config.json \
    --report_to tensorboard \
    --run_name $JOB_NAME-$DATASET_NAME-$(date +%b%d-%H-%M-%Y) \
    --fp16
    # --fp16_backend amp \
    # --fp16_opt_level O1

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