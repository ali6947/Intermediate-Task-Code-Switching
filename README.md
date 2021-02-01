# Intermediate-Task-Code-Switching
This is the repo for the experiments mentioned in the paper

## Datasets
The datasets required for the experiments can be found at the following link: [Data](https://drive.google.com/file/d/1lafT_uN-gpQ3OzproZQ5zihQBwxV1-pm/view?usp=sharing)
This includes the following datasets in the given structure
```
final_Data
  ├── MLM
  │   ├── generalCS
  │   ├── generalCS_movieCS_nlipremise
  │   ├── generalCS_movieCS_qactxt
  │   ├── movieCS_generalCS
  │   ├── movieCS_nlipremise
  │   └── movieCS_qactxt
  ├── QA_EN_HI
  │   ├── GLUECoS (filename: train-v2.0.json & dev-v2.0.json)
  │   ├── SQuAD (English, Romanised Hindi & Bilingual)
  │   ├── XQuAD Test set in English
  │   ├── XQuAD Machine Translated  & Transliterated to Romanised Hindi using Google API
  │   ├── XQuAD Transliterated to Romanised Hindi using Google API
  │   └── XQuAD Transliterated to Romanised Hindi using IndicTrans
  └── NLI_EN_HI
      ├── english_MNLI
      ├── gluecos
      ├── romanised_hindi_MNLI
      ├── XNLI_eng
      ├── XNLI_GoogMT
      ├── XNLI_GoogTranslit
      └── XNLI_IndicTranslit
```

## Training models
The code provides methods to perform intermediate training on different datasets as well as finetuning on the GLUECoS benchmarks

The following intermediate routines are available
1. QA
   -  on monolingual SQuAD (English, Romanised Hindi)
   -  on bilingual SQuAD
   -  interspersed MLM and bilingual SQuAD 
   -  on machine translated (Google API) XQuAD Test-set (bilingual version)
   -  on XQuAD Test-set transliterated using Indictrans (bilingual version)
   -  on XQuAD Test-set transliterated using Google API (bilingual version)

2. NLI
   -  on monolingual MNLI (English, Romanised Hindi)
   -  on bilingual MNLI
   -  interspersed MLM and bilingual MNLI 
   -  on machine translated (Google API) XNLI Test-set (bilingual version)
   -  on XNLI Test-set transliterated using Indictrans (bilingual version)
   -  on XNLI Test-set transliterated using Google API (bilingual version)

3. Interspersed QA and NLI 

Also available are the methods for finetuning the model on GLUECoS NLI and QA benchmarks

## Training requirements
The code uses pytorch(v)(add version)
The requirements are listed in the file `Code/requirements.txt`. They can be installed via 
    ```
    pip install -r Code/requirements.txt
    ```

### Training
The command below can be used to run both intermediate and fine-tuning experiments. The training scripts uses the Huggingface library and support any models based on BERT, XLM, XLM-Roberta and similar models. 

```
bash train_final.sh MODEL MODEL_TYPE TASK
```
Example Usage
```
bash train_final.sh bert-base-multilingual-cased bert GLUECOS_QA_EN_HI
```

The Tasks available are
- GLUECOS_QA_EN_HI
- engSQUAD_QA_EN_HI
- roman_hinSQUAD_QA_EN_HI
- bilingualSQUAD_QA_EN_HI
- dual_MLM1_bilSQUAD_EN_HI
- dual_MLM2_bilSQUAD_EN_HI
- dual_MLM3_bilSQUAD_EN_HI
- xquad_bilingual_MT_test_then_gluecos_QA_EN_HI
- xquad_bilingual_GoogTranslit_test_then_gluecos_QA_EN_HI
- xquad_bilingual_IndicTranslit_test_then_gluecos_QA_EN_HI
- GLUECOS_NLI_EN_HI
- engMNLI_NLI_EN_HI
- roman_hin_MNLI_NLI_EN_HI
- bilingual_MNLI_NLI_EN_HI
- bilingXNLI_GoogMT_EN_HI
- bilingXNLI_GoogTranslit_EN_HI
- bilingXNLI_IndicTranslit_EN_HI
- dual_MLM1_bilMNLI_EN_HI
- dual_MLM2_bilMNLI_EN_HI
- dual_MLM3_bilMNLI_EN_HI
- dual_NLI_QA_EN_HI

Note that: MLM1 refers to GeneralCS. MLM2 refers to GeneralCS+MovieCS. MLM3 refers to GeneralCS+MovieCS+QA contexts for QA intermediate task training, and GeneralCS+MovieCS+NLI Premise for NLI intermediate task training.
### Evaluation
To get the test set results on the GLUECoS benchmarks please refer to the [GLUECoS repo](https://github.com/microsoft/GLUECoS)
