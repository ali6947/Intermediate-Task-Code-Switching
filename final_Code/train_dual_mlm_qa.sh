# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-QA_EN_HI}
MODEL=${2:-bert-base-multilingual-cased}
MODEL_TYPE=${3:-bert}
TRAIN_FILE=${4:-train-v2.0.json}
PREDICT_FILE=${5:-dev-v2.0.json}
MLM_DATA_FILE=${6:-"$REPO/final_Data/MLM/generalCS_movieCS_nlipremise.txt"}
DATA_DIR=${7:-"$REPO/final_Data"}
OUT_DIR=${8:-"$REPO/Results"}

export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

# EPOCH=4
# BATCH_SIZE=4 #1
# MLM_BATCH_SIZE=4 #1
# EVAL_BATCH_SIZE=4 #2
MAX_SEQ=512
MLM_MAX_SEQ=256

EPOCH=4
if [ $MODEL_TYPE == 'bert' ]; then
  BATCH_SIZE=4
  EVAL_BATCH_SIZE=4
  GRADIENT_ACC=10
  LOGGING_STEPS=800
  MLM_BATCH_SIZE=4
elif [ $MODEL_TYPE == 'xlm-roberta' ]; then
  BATCH_SIZE=1
  MLM_BATCH_SIZE=1
  EVAL_BATCH_SIZE=1
  LOGGING_STEPS=200
  GRADIENT_ACC=40
fi

python3.6 $PWD/final_Code/dual_qa_mlm.py \
  --data_dir $DATA_DIR/QA_EN_HI \
  --output_dir $OUT_DIR/$TASK \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --do_eval \
  --num_train_epochs $EPOCH \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
  --max_seq_length $MAX_SEQ \
  --overwrite_output_dir \
  --seed 52 \
  --logging_steps 700 \
  --gradient_accumulation_steps 10 \
  --do_train \
  --train_file $TRAIN_FILE \
  --predict_file $PREDICT_FILE \
   --evaluate_during_training \
  --mlm_datapath $MLM_DATA_FILE \
  --mlm_per_gpu_train_batch_size $MLM_BATCH_SIZE \
  --mlm_max_seq_length $MLM_MAX_SEQ \
    # --save_model #to save the pretrained model please
  # --save_folder #location to save the model, creates folder if it does not exist



