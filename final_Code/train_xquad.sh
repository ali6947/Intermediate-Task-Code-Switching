# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-QA_EN_HI}
MODEL=${2:-bert-base-multilingual-cased}
MODEL_TYPE=${3:-bert}
TRAIN_FILE=${4:-train-v2.0.json}
PREDICT_FILE=${5:-dev-v2.0.json}
PRETR_TRAIN_FILE=${6:-train-v2.0.json}
PRETR_PREDICT_FILE=${7:-dev-v2.0.json}
DATA_DIR=${8:-"$REPO/final_Data"}
OUT_DIR=${9:-"$REPO/Results"}

export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

EPOCH=4
BATCH_SIZE=4 #1
MLM_BATCH_SIZE=4 #1
EVAL_BATCH_SIZE=4 #2
MAX_SEQ=512
PRETR_EPOCH=3


python3.6 $PWD/final_Code/sequential_qa.py \
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
  --logging_steps 7000 \
  --gradient_accumulation_steps 10 \
  --do_train \
  --train_file $TRAIN_FILE \
  --predict_file $PREDICT_FILE \
  --pretr_train_file $PRETR_TRAIN_FILE \
  --pretr_predict_file $PRETR_PREDICT_FILE \
  --num_pretr_epochs $PRETR_EPOCH \
  # --save_model \
  # --save_folder \ #name of directory to save the model in comes here
  



