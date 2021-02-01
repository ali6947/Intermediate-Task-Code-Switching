# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-NLI_EN_HI}
MODEL=${2:-bert-base-multilingual-cased}
# MODEL=${2:-"$REPO/azure_ml/pytorch_model.bin"}
MODEL_TYPE=${3:-bert}
DATA_DIR=${4:-"$REPO/final_Data"}
ROMA="romanised"
OUT_DIR=${5:-"$REPO/Results"}

export NVIDIA_VISIBLE_DEVICES=2
export CUDA_VISIBLE_DEVICES=2


EPOCH=4
BATCH_SIZE=4 #set to match the exact GLUECOS repo
MAX_SEQ=256

python3.6 $PWD/final_Code/run_xnli_vanilla.py \
  --data_dir $DATA_DIR \
  --output_dir $OUT_DIR/$TASK \
  --model_type $MODEL_TYPE \
  --model_name_or_path $MODEL \
  --language en \
  --do_train \
  --do_eval \
  --num_train_epochs $EPOCH \
  --per_gpu_train_batch_size $BATCH_SIZE \
  --max_seq_length $MAX_SEQ \
  --overwrite_output_dir \
  --overwrite_cache \
  --save_steps -1 \
  --seed 52 \
  --logging_steps 8000 \
  --per_gpu_eval_batch_size 4 \
  --gradient_accumulation_steps 10 \
  
