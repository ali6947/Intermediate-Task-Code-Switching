# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
NLI_TASK=${1:-NLI_EN_HI}
MODEL=${2:-bert-base-multilingual-cased}
MODEL_TYPE=${3:-bert}
DATA_DIR=${4:-"$REPO/final_data"}
OUT_DIR=${5:-"$REPO/Results"}
QA_TASK=QA_EN_HI
export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

EPOCH=3
BATCH_SIZE=8
QA_BATCH_SIZE=4
MAX_SEQ=256
QA_MAX_SEQ=512

python3.6 $PWD/final_Code/xnli_qa_mtl.py \
  --data_dir $DATA_DIR/$NLI_TASK \
  --qa_data_dir $DATA_DIR/$QA_TASK \
  --output_dir $OUT_DIR/$NLI_TASK \
  --qa_output_dir $OUT_DIR/$QA_TASK \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --language en \
  --do_train \
  --do_eval \
  --evaluate_during_training \
  --num_train_epochs $EPOCH \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --qa_per_gpu_train_batch_size $QA_BATCH_SIZE \
  --gradient_accumulation_steps 10 \
  --max_seq_length $MAX_SEQ \
  --qa_max_seq_length $QA_MAX_SEQ \
  --overwrite_output_dir \
  --save_steps -1  \
  --train_file train_squad_bilingual.json \
  --predict_file dev_squad_bilingual.json \
  --nli_logging_steps 1000 \
  --qa_logging_steps 1000 \
  --seed 42 \
  # --save_model_qa \
  # --save_model_nli \
  # --qa_save_folder \
  # --nli_save_folder # #location to save the model, creates folder if it does not exist



