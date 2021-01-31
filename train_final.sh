#!/usr/bin/bash
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

REPO=$PWD
MODEL=${1:-bert-base-multilingual-cased}
MODEL_TYPE=${2:-bert}
TASK=${3:-POS_EN_ES}
DATA_DIR=${4:-"$REPO/final_Data"}
CODE_DIR=${5:-"$REPO/final_Code"}
OUTPUT_DIR=${6:-"$REPO/Results"}
mkdir -p $OUTPUT_DIR
echo "Fine-tuning $MODEL on $TASK"

if [ $TASK == 'GLUECOS_QA_EN_HI' ]; then
  bash $CODE_DIR/train_qa_vanilla.sh $TASK $MODEL $MODEL_TYPE 
elif [ $TASK == 'engSQUAD_QA_EN_HI' ]; then
  bash $CODE_DIR/train_qa_vanilla.sh $TASK $MODEL $MODEL_TYPE train_squad_eng.json dev_squad_eng.json
elif [ $TASK == 'roman_hinSQUAD_QA_EN_HI' ]; then
  bash $CODE_DIR/train_qa_vanilla.sh $TASK $MODEL $MODEL_TYPE train_roman_hi_squad.json dev_roman_hi_squad.json  
elif [ $TASK == 'bilinugualSQUAD_QA_EN_HI' ]; then
  bash $CODE_DIR/train_qa_vanilla.sh $TASK $MODEL $MODEL_TYPE train_squad_bilingual.json dev_squad_bilingual.json

elif [ $TASK == 'dual_MLM1_bilSQUAD_EN_HI' ]; then
  bash $CODE_DIR/train_dual_mlm_qa.sh $TASK $MODEL $MODEL_TYPE train_squad_bilingual.json dev_squad_bilingual.json $DATA_DIR/MLM/generalCS.txt
elif [ $TASK == 'dual_MLM2_bilSQUAD_EN_HI' ]; then
  bash $CODE_DIR/train_dual_mlm_qa.sh $TASK $MODEL $MODEL_TYPE train_squad_bilingual.json dev_squad_bilingual.json $DATA_DIR/MLM/movieCS_generalCS.txt
elif [ $TASK == 'dual_MLM3_bilSQUAD_EN_HI' ]; then
  bash $CODE_DIR/train_dual_mlm_qa.sh $TASK $MODEL $MODEL_TYPE train_squad_bilingual.json dev_squad_bilingual.json $DATA_DIR/MLM/generalCS_movieCS_qactxt.txt

elif [ $TASK == 'xquad_bilingual_MT_test_then_gluecos_QA_EN_HI' ]; then
  bash $CODE_DIR/train_xquad.sh $TASK $MODEL $MODEL_TYPE train-v2.0.json dev-v2.0.json xquad_machine_translated_bilingual.json
elif [ $TASK == 'xquad_bilingual_GoogTranslit_test_then_gluecos_QA_EN_HI' ]; then
  bash $CODE_DIR/train_xquad.sh $TASK $MODEL $MODEL_TYPE train-v2.0.json dev-v2.0.json xquad_google_translit_bilingual.json
elif [ $TASK == 'xquad_bilingual_IndicTranslit_test_then_gluecos_QA_EN_HI' ]; then
  bash $CODE_DIR/train_xquad.sh $TASK $MODEL $MODEL_TYPE train-v2.0.json dev-v2.0.json xquad_indic_translit_bilingual.json

elif [ $TASK == 'GLUECOS_NLI_EN_HI' ]; then
  bash $CODE_DIR/train_nli_vanilla.sh $TASK $MODEL $MODEL_TYPE $DATA_DIR/NLI_EN_HI/gluecos
elif [ $TASK == 'engMNLI_NLI_EN_HI' ]; then
  bash $CODE_DIR/train_mnli_with_val.sh $TASK $MODEL $MODEL_TYPE $DATA_DIR/NLI_EN_HI/english_MNLI
elif [ $TASK == 'roman_hin_MNLI_NLI_EN_HI' ]; then
  bash $CODE_DIR/train_mnli_with_val.sh $TASK $MODEL $MODEL_TYPE $DATA_DIR/NLI_EN_HI/romanised_hindi_MNLI
elif [ $TASK == 'bilingual_MNLI_NLI_EN_HI' ]; then
  bash $CODE_DIR/train_bil_mnli_with_val.sh $TASK $MODEL $MODEL_TYPE $DATA_DIR/NLI_EN_HI/english_MNLI $DATA_DIR/NLI_EN_HI/romanised_hindi_MNLI
 
elif [ $TASK == 'bilingXNLI_GoogMT_EN_HI' ]; then
  bash $CODE_DIR/train_bil_mnli_with_val.sh $TASK $MODEL $MODEL_TYPE $DATA_DIR/NLI_EN_HI/XNLI_eng $DATA_DIR/NLI_EN_HI/XNLI_GoogMT
elif [ $TASK == 'bilingXNLI_GoogTranslit_EN_HI' ]; then
  bash $CODE_DIR/train_bil_mnli_with_val.sh $TASK $MODEL $MODEL_TYPE DATA_DIR/NLI_EN_HI/XNLI_eng $DATA_DIR/NLI_EN_HI/XNLI_GoogTranslit
elif [ $TASK == 'bilingXNLI_IndicTranslit_EN_HI' ]; then
  bash $CODE_DIR/train_bil_mnli_with_val.sh $TASK $MODEL $MODEL_TYPE $DATA_DIR/NLI_EN_HI/XNLI_eng $DATA_DIR/NLI_EN_HI/XNLI_IndicTranslit


elif [ $TASK == 'dual_MLM1_bilMNLI_EN_HI' ]; then
  bash $CODE_DIR/train_dual_mlm_bilmnli.sh $TASK $MODEL $MODEL_TYPE $DATA_DIR/NLI_EN_HI $DATA_DIR/MLM/generalCS.txt
elif [ $TASK == 'dual_MLM2_bilMNLI_EN_HI' ]; then
  bash $CODE_DIR/train_dual_mlm_bilmnli.sh $TASK $MODEL $MODEL_TYPE $DATA_DIR/NLI_EN_HI $DATA_DIR/MLM/movieCS_generalCS.txt
elif [ $TASK == 'dual_MLM3_bilMNLI_EN_HI' ]; then
  bash $CODE_DIR/train_dual_mlm_bilmnli.sh $TASK $MODEL $MODEL_TYPE $DATA_DIR/NLI_EN_HI $DATA_DIR/MLM/generalCS_movieCS_nlipremise.txt
elif [ $TASK == 'dual_NLI_QA_EN_HI' ]; then
  bash $CODE_DIR/train_dual_nli_qa.sh 'NLI_EN_HI' $MODEL $MODEL_TYPE $DATA_DIR
fi

