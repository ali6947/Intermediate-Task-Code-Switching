# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
TASK=${1:-NLI_EN_HI}
MODEL=${2:-bert-base-multilingual-cased}
# MODEL=${2:-"$REPO/azure_ml/pytorch_model.bin"}
MODEL_TYPE=${3:-bert}
DATA_DIR=${4:-"$REPO/final_Data"}

MLM_DATA_FILE=${5:-"$REPO/final_Data/MLM/generalCS_movieCS_nlipremise.txt"}
OUT_DIR=${6:-"$REPO/Results"}
export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0
# export RANK=0
# export WORLD_SIZE=2
# export MASTER_ADDR=127.0.0.1
# export MASTER_PORT=80

EPOCH=4
if [ $MODEL_TYPE == 'bert' ]; then
  BATCH_SIZE=4
  EVAL_BATCH_SIZE=4
  GRADIENT_ACC=10
  LOGGING_STEPS=800
elif [ $MODEL_TYPE == 'xlm-roberta' ]; then
  BATCH_SIZE=1
  EVAL_BATCH_SIZE=1
  LOGGING_STEPS=200
  GRADIENT_ACC=40
fi
MAX_SEQ=256

python3.6 $PWD/final_Code/bil_mnli_mlm_dual.py \
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
  --logging_steps $LOGGING_STEPS \
  --per_gpu_eval_batch_size $EVAL_BATCH_SIZE \
  --gradient_accumulation_steps $GRADIENT_ACC \
  --mlm_datapath $MLM_DATA_FILE \
  # --save_model #to save the pretrained model please
  # --save_folder #location to save the model, creates folder if it does not exist
  
