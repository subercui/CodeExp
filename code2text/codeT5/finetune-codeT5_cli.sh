# ========== SET DATASET ==========
DATASET_NAME="GithubCodeDocStep1n2Filtered"
SPLIT_PERC=5
LOG_STEP=250
SAVE_STEP=500

# DATASET_NAME="GithubCodeDoc"
# SPLIT_PERC=1
# LOG_STEP=2500
# SAVE_STEP=10000

# ======= SET JOB & MODEL NAME =======
# Here are some job and model examples, prefer to set in yaml:

# JOB_NAME="ft-stage1-codeT5"
# MODEL="Salesforce/codet5-base"

MAX_LENGTH=1024
BATCH_SIZE=2

while getopts :j:m:s:b:h flag
do
    case "${flag}" in
        j) JOB_NAME=${OPTARG};;
        m) MODEL=${OPTARG};;
        s) MAX_LENGTH=${OPTARG};;
        b) BATCH_SIZE=${OPTARG};;
        h) # help
            echo "Usage: $0 [-j JOB_NAME] [-m MODEL] [-s BLOCK_SIZE] [-b BATCH_SIZE] NPROC"
            echo "  -j JOB_NAME: Job name (default: $JOB_NAME)"
            echo "  -m MODEL: Model name (default: $MODEL)"
            echo "  -s MAX_LENGTH: Max source and target length (default: $MAX_LENGTH)"
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

echo "setting JOB_NAME=$JOB_NAME, MODEL=$MODEL, DATASET=$DATASET_NAME, MAX_LENGTH=$MAX_LENGTH, BATCH_SIZE=$BATCH_SIZE, NPROC=$NPROC"

python -m torch.distributed.launch \
    --nproc_per_node $NPROC finetune-codeT5.py \
    --model_name_or_path $MODEL \
    --do_train \
    --do_eval \
    --text_column code \
    --summary_column docstring \
    --train_file "/mnt/default/data/${DATASET_NAME}.train.jsonl" \
    --validation_split_percentage $SPLIT_PERC \
    --max_source_length $MAX_LENGTH \
    --max_target_length $MAX_LENGTH \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $((BATCH_SIZE*2)) \
    --evaluation_strategy steps \
    --eval_steps $SAVE_STEP \
    --output_dir /mnt/default/$JOB_NAME-${DATASET_NAME}-$(date +%b%d-%H-%M-%Y)/ \
    --overwrite_output_dir \
    --logging_steps $LOG_STEP \
    --logging_dir $AMLT_OUTPUT_DIR/logs \
    --save_steps $SAVE_STEP \
    --num_train_epochs 10 \
    --report_to tensorboard \
    --run_name $JOB_NAME-$DATASET_NAME-$(date +%b%d-%H-%M-%Y) \
    --fp16
    # --fp16_backend amp \
    # --fp16_opt_level O1