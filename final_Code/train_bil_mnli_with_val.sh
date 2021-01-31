# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-NLI_EN_HI}
MODEL=${2:-bert-base-multilingual-cased}
# MODEL=${2:-"$REPO/azure_ml/pytorch_model.bin"}
MODEL_TYPE=${3:-bert}
DATA_DIR1=${4:-"$REPO/final_Data"}
DATA_DIR2=${5:-"$REPO/final_Data"}
ROMA="romanised"
OUT_DIR=${6:-"$REPO/Results"}

export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0


EPOCH=4
BATCH_SIZE=4 #set to match the exact GLUECOS repo
MAX_SEQ=256

python3.6 $PWD/final_Code/bil_mnli_with_val.py \
  --data_dir1 $DATA_DIR1 \
  --data_dir2 $DATA_DIR2 \
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
  --logging_steps 800 \
  --per_gpu_eval_batch_size 4 \
  --gradient_accumulation_steps 10 \
  # --save_model #to save the pretrained model please
  # --save_folder #location to save the model, creates folder if it does not exist
  
