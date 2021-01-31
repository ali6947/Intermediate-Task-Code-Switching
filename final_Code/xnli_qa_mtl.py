# coding=utf-8
# Copyright (c) Microsoft Corporation. Licensed under the MIT license.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning multi-lingual models on XNLI (Bert, DistilBERT, XLM).
    Adapted from `examples/run_glue.py`"""


import argparse
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,BertForMaskedLM,BertForQuestionAnswering,RobertaForQuestionAnswering,
    BertTokenizer,
    XLMConfig, XLMForSequenceClassification, XLMTokenizer,
    XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling, LineByLineTextDataset,
    squad_convert_examples_to_features,

)
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import xnli_output_modes as output_modes
from transformers import xnli_processors as processors

from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

NLI_PROB=0.5
QA_MODEL_FOLDER='wrongt_qa_save_xlm_r_pretr_on_bil_squad_bil_xnli'
NLI_MODEL_FOLDER='wrongt_nli_save_xlm_r_pretr_on_bil_squad_bil_xnli'
class GLUECoSNLIProcessor(processors['xnli']):
    def get_labels(self):
        return ["contradiction", "entailment"]


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "xlm-roberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    # "joeddav-xlm-roberta-large-xnli":()
}

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class MaskedLMDataset(Dataset):
    def __init__(self, file, tokenizer,max_len1):
        self.tokenizer = tokenizer
        self.max_len = max_len1
        self.lines = self.load_lines(file)
        self.dataset = self.encode_lines(self.lines)

    def load_lines(self, file):
        with open(file) as f:
            lines = [
                line
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
        return lines
    
    def encode_lines(self, lines):
        batch_encoding = self.tokenizer.batch_encode_plus(
            lines, add_special_tokens=True, truncation=True, max_length=self.max_len,pad_to_max_length=True)
        # print(torch.tensor(batch_encoding)[:10])
        # print([len(i) for i in batch_encoding["input_ids"]])
        # print(torch.tensor(batch_encoding["token_type_ids"]).shape)
        # print(torch.tensor(batch_encoding["attention_mask"]).shape)
        return TensorDataset(torch.tensor(batch_encoding["input_ids"]),torch.tensor(batch_encoding["token_type_ids"]),\
            torch.tensor(batch_encoding["attention_mask"]))

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.dataset[idx]
        # return torch.tensor(self.ids[idx], dtype=torch.long)


def train(args, nli_dataset, model, nli_layer,loss_nli_fn, tokenizer,qa_dataset):
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.qa_train_batch_size = args.qa_per_gpu_train_batch_size * max(1, args.n_gpu)
    args.mlm_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    nli_sampler = RandomSampler(nli_dataset) if args.local_rank == -1 else DistributedSampler(nli_dataset)
    nli_dataloader = DataLoader(nli_dataset, sampler=nli_sampler, batch_size=args.train_batch_size)

    qa_sampler = RandomSampler(qa_dataset) if args.local_rank == -1 else DistributedSampler(qa_dataset)
    qa_dataloader = DataLoader(qa_dataset, sampler=qa_sampler, batch_size=args.qa_train_batch_size)
    

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // ((len(nli_dataloader)+len(qa_dataloader)) // args.gradient_accumulation_steps) + 1
    else:
        t_total = (len(nli_dataloader)+len(qa_dataloader)) // args.gradient_accumulation_steps * args.num_train_epochs
        # m_total = len(mlm_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    t_total = (2*len(nli_dataloader)+len(qa_dataloader))#// args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        {"params":[p for n,p in nli_layer.named_parameters()],"weight_decay":args.weight_decay},
        # {"params":[p for n,p in mlm_layer.named_parameters()],"weight_decay":args.weight_decay},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", 2*len(nli_dataset)+len(qa_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", 2*args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        2*args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss,tr_loss_qa, logging_loss = 0.0, 0.0,0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    best_val_acc_nli,best_f1_qa=0,0
    for _ in train_iterator:
        steps_done=0
        # nli_fin=False
        # mlm_fin=False
        drnli,nrnli,lossnli,lossqa,iters,drmlm =0,0,0,0,0,0
        # nli_epoch_iterator = iter(nli_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        # mlm_epoch_iterator = iter(mlm_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        nli_dataloader = DataLoader(nli_dataset, sampler=nli_sampler, batch_size=args.train_batch_size)
        qa_dataloader = DataLoader(qa_dataset, sampler=qa_sampler, batch_size=args.qa_train_batch_size)
        nli_epoch_iterator = iter(nli_dataloader)
        qa_epoch_iterator = iter(qa_dataloader)
        pbar = tqdm(total=t_total)
        upd_steps=0
        while steps_done<t_total:
            r = random.uniform(0, 1)
            upd_steps+=1
            if r<NLI_PROB:
                # print(nli_steps)
                for icnt in range(args.gradient_accumulation_steps):
                    batch_nli=next(nli_epoch_iterator,None)
                    if batch_nli==None:
                        # nli_epoch_iterator.close()
                        nli_dataloader = DataLoader(nli_dataset, sampler=nli_sampler, batch_size=args.train_batch_size)
                        nli_epoch_iterator = iter(nli_dataloader)
                        batch_nli=next(nli_epoch_iterator,None)
                    steps_done+=1
                    pbar.update(1)
                   
                    model.train()
                    nli_layer.train()
                    batch_nli = tuple(t.to(args.device) for t in batch_nli)
                    # inputs_nli = {"input_ids": batch_nli[0], "attention_mask": batch_nli[1]}#, "labels": batch[3]}
                    inputs_nli = {"input_ids": torch.cat((batch_nli[0],batch_nli[4]),0), "attention_mask": torch.cat((batch_nli[1],batch_nli[5]),0)}
                     #"labels": torch.cat((batch_nli[3],batch_nli[7]),0)}

                    # if args.model_type != "distilbert":
                    #     inputs_nli["token_type_ids"] = (
                    #         batch_nli[2] if args.model_type in ["bert"] else None
                    #     )  # XLM and DistilBERT don't use segment_ids
                    if args.model_type != "distilbert":
                        inputs_nli["token_type_ids"] = (
                            torch.cat((batch_nli[2],batch_nli[6]),0) if args.model_type in ["bert"] else None
                        )  # XLM and DistilBERT don't use segment_ids
                    outputs_nli = model(**inputs_nli)
                    hidden = outputs_nli[-1][-1]  # model outputs are always tuple in transformers (see doc)
                    nli_pred = nli_layer(hidden[:, 0, :])
                    _,nli_labels=torch.max(nli_pred,-1)
                    
                    # nrnli+=torch.eq(nli_labels,batch_nli[3]).sum().item()
                    # drnli+=float(batch_nli[3].shape[0])
                    # loss_nli = loss_nli_fn(nli_pred, batch_nli[3])
                    nrnli+=torch.eq(nli_labels,torch.cat((batch_nli[3],batch_nli[7]),0)).sum().item()
                    drnli+=float(batch_nli[3].shape[0])+float(batch_nli[7].shape[0])
                    loss_nli = loss_nli_fn(nli_pred, torch.cat((batch_nli[3],batch_nli[7]),0))
                    # loss_nli = loss_pred
                    lossnli+=loss_nli.item()

                    iters+=1

                    if args.n_gpu > 1:
                        loss_nli = loss_nli.mean()  # mean() to average on multi-gpu parallel training
                    if args.gradient_accumulation_steps > 1:
                        loss_nli = loss_nli / args.gradient_accumulation_steps

                    loss_nli.backward()
                    tr_loss += loss_nli.item()
                       
                    # if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                nli_layer.zero_grad()
                global_step += 1
                # if (nli_steps)%200==0:
                #     results,val_acc = evaluate(args, model, tokenizer, nli_layer, loss_nli_fn)
                #     if val_acc>best_val_acc:
                #         logger.info('temporary model saved!')
                #         best_val_acc=val_acc
                #         model.save_pretrained('./pretrained_on_xnli_squad')
                #         tokenizer.save_pretrained('./pretrained_on_xnli_squad')
                #         torch.save(args, os.path.join('./pretrained_on_xnli_squad', "training_args.bin"))
    
                    # logging_loss = tr_loss

                    # if args.max_steps > 0 and global_step > args.max_steps:
                    #     epoch_iterator.close()
                    #     break
            else:
                for icnt_qa in range(args.gradient_accumulation_steps):
                    qa_batch=next(qa_epoch_iterator,None)
                    if qa_batch==None:
                        # nli_epoch_iterator.close()
                        qa_dataloader = DataLoader(qa_dataset, sampler=qa_sampler, batch_size=args.qa_train_batch_size)
                        qa_epoch_iterator = iter(qa_dataloader)
                        qa_batch=next(qa_epoch_iterator,None)
                    model.train()
                    qa_batch = tuple(t.to(args.device) for t in qa_batch)
                    steps_done+=1
                    pbar.update(1)
                    qa_inputs = {
                        "input_ids": qa_batch[0],
                        "attention_mask": qa_batch[1],
                        "token_type_ids": qa_batch[2],
                        "start_positions": qa_batch[3],
                        "end_positions": qa_batch[4],
                    }

                    if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                        del qa_inputs["token_type_ids"]

                    if args.model_type in ["xlnet", "xlm"]:
                        qa_inputs.update({"cls_index": qa_batch[5], "p_mask": qa_batch[6]})
                        if args.version_2_with_negative:
                            qa_inputs.update({"is_impossible": qa_batch[7]})
                        if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                            qa_inputs.update(
                                {"langs": (torch.ones(qa_batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                            )

                    qa_outputs = model(**qa_inputs)
                    # model outputs are always tuple in transformers (see doc)
                    loss_qa = qa_outputs[0]
                    lossqa+=loss_qa.item()
                    iters+=1
                    if args.n_gpu > 1:
                        loss_qa = loss_qa.mean()  # mean() to average on multi-gpu parallel training
                    if args.gradient_accumulation_steps > 1:
                        loss_qa = loss_qa / args.gradient_accumulation_steps
                    loss_qa.backward()

                    tr_loss_qa += loss_qa.item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                # if (step+1)%1000==0:
                #     results,val_acc = evaluate(args, model, tokenizer)
                #     if val_acc>best_val_acc:
                #         logger.info('temporary model saved!')
                #         best_val_acc=val_acc
                #         model.save_pretrained('./pretrained_on_bilingual_xnli')
                #         tokenizer.save_pretrained('./pretrained_on_bilingual_xnli')
                #         torch.save(args, os.path.join('./pretrained_on_bilingual_xnli', "training_args.bin"))
    
                #     logging_loss = tr_loss

                # if args.max_steps > 0 and global_step > args.max_steps:
                #     epoch_iterator.close()
                #     break
            # print(upd_steps,end='&&&&\n')
            if args.local_rank == -1 and upd_steps%args.qa_logging_steps==0:
                results_qa = evaluate_qa(args, model, tokenizer)
                if results_qa['f1']>best_f1_qa:
                    best_f1_qa=results_qa['f1']
                    if args.save_model_qa:
                        logger.info("New best f1 at %s, saving model",str(best_f1_qa))
                        model.save_pretrained(args.qa_save_folder)
                        tokenizer.save_pretrained(args.qa_save_folder)
                        torch.save(args, os.path.join(args.qa_save_folder, "training_args.bin"))
                    else:
                        logger.info("Best f1 at %s, no model saved",str(best_f1_qa))
                else:
                    logger.info(" best f1 still at %s, no model saved",str(best_f1_qa))

            if args.local_rank == -1 and upd_steps%args.nli_logging_steps==0:
                results_nli,val_acc = evaluate(args, model, tokenizer, nli_layer, loss_nli_fn)
                if val_acc>best_val_acc_nli:
                    best_val_acc_nli=val_acc
                    if args.save_model_nli:
                        logger.info("New best NLI at %s, saving model",str(best_val_acc_nli))
                        model.save_pretrained(args.nli_save_folder)
                        tokenizer.save_pretrained(nli_save_folder)
                        torch.save(nli_layer.state_dict(),os.path.join(nli_save_folder,'nli_layer.bin'))
                        torch.save(args, os.path.join(args.nli_save_folder, "training_args.bin"))
                    else:
                        logger.info("New best NLI at %s, no model saved",str(best_val_acc_nli))
                else:
                    logger.info("Best NLI acc still at %s, no model saved",str(best_val_acc_nli))


        # if args.max_steps > 0 and global_step > args.max_steps:
        #     train_iterator.close()
        #     break

        # logger.info('nli acc=%s ',str(nrnli/drnli))
        # logger.info('loss nli=%s',str(lossnli/iters))
        # logger.info('loss mlm=%s',str(lossmlm/drmlm))

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    return global_step, tr_loss / global_step

def evaluate_qa(args, model, tokenizer, prefix=""):
    qa_dataset, qa_examples, qa_features = load_and_cache_examples_squad(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.qa_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.qa_output_dir)

    args.qa_eval_batch_size = args.qa_per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    qa_eval_sampler = SequentialSampler(qa_dataset)
    qa_eval_dataloader = DataLoader(qa_dataset, sampler=qa_eval_sampler, batch_size=args.qa_eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(qa_dataset))
    logger.info("  Batch size = %d", args.qa_eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for qa_batch in tqdm(qa_eval_dataloader, desc="Evaluating"):
        model.eval()
        qa_batch = tuple(t.to(args.device) for t in qa_batch)

        with torch.no_grad():
            qa_inputs = {
                "input_ids": qa_batch[0],
                "attention_mask": qa_batch[1],
                "token_type_ids": qa_batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del qa_inputs["token_type_ids"]

            qa_example_indices = qa_batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                qa_inputs.update({"cls_index": qa_batch[4], "p_mask": qa_batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    qa_inputs.update(
                        {"langs": (torch.ones(qa_batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            qa_outputs = model(**qa_inputs)
            # print(type(qa_outputs))
            # [print(oi.shape) for oi in qa_outputs]
        for qa_i, qa_example_index in enumerate(qa_example_indices):
            eval_feature = qa_features[qa_example_index.item()]
            unique_id = int(eval_feature.unique_id)
            qa_output = [to_list(qa_output_iter[qa_i]) for qa_output_iter in qa_outputs]
            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(qa_output) >= 5:
                start_logits = qa_output[0]
                start_top_index = qa_output[1]
                end_logits = qa_output[2]
                end_top_index = qa_output[3]
                cls_logits = qa_output[4]

                qa_result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits,_ = qa_output
                qa_result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(qa_result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(qa_dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.qa_output_dir, "predictions.json")
    output_nbest_file = os.path.join(args.qa_output_dir, "nbest_predictions.json")

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.qa_output_dir, "null_odds.json")
    else:
        output_null_log_odds_file = None

    qa_predictions = compute_predictions_logits(
        qa_examples,
        qa_features,
        all_results,
        args.n_best_size,
        args.max_answer_length,
        args.do_lower_case,
        output_prediction_file,
        output_nbest_file,
        output_null_log_odds_file,
        args.verbose_logging,
        args.version_2_with_negative,
        args.null_score_diff_threshold,
        tokenizer,
    )

    # Compute the F1 and exact scores.
    qa_results = squad_evaluate(qa_examples, qa_predictions)
    # print(results)
    # results = {}
    return qa_results

def evaluate(args, model, tokenizer, nli_classifier, loss_nli_fn, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples_nli(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        eval_preds = None
        out_label_ids = None
        actual_labels=None
        for eval_batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            nli_classifier.eval()
            eval_batch = tuple(t.to(args.device) for t in eval_batch)

            with torch.no_grad():
                # eval_inputs = {"input_ids": eval_batch[0], "attention_mask": eval_batch[1]}
                eval_inputs = {"input_ids": torch.cat((eval_batch[0],eval_batch[4]),0), "attention_mask": torch.cat((eval_batch[1],eval_batch[5]),0)}
                # if args.model_type != "distilbert":
                #     eval_inputs["token_type_ids"] = (
                #         eval_batch[2] if args.model_type in ["bert"] else None
                #     )  # XLM and DistilBERT don't use segment_ids
                if args.model_type != "distilbert":
                    eval_inputs["token_type_ids"] = (
                        torch.cat((eval_batch[2],eval_batch[6]),0) if args.model_type in ["bert"] else None
                    )  # XLM and DistilBERT don't use segment_ids
                eval_outputs = model(**eval_inputs)
                eval_hidden = eval_outputs[-1][-1]
                eval_nli_pred = nli_classifier(eval_hidden[:, 0, :])
                #pos_pred = pos_pred.permute(0, 2, 1)
                eval_loss_pred = loss_nli_fn(eval_nli_pred, torch.cat((eval_batch[3],eval_batch[7]),0))
                # print(pos_pred.shape)
                logits = eval_nli_pred#.max(dim = 1)[1]
                #tmp_eval_loss, logits = outputs[:2]
                eval_loss += eval_loss_pred.item()
            nb_eval_steps += 1
            if eval_preds is None:
                eval_preds = logits.detach().cpu().numpy()
                out_label_ids = eval_batch[3].detach().cpu().numpy()
                out_label_ids = np.append(out_label_ids, eval_batch[7].detach().cpu().numpy(), axis=0)
            else:
                eval_preds = np.append(eval_preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, eval_batch[3].detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, eval_batch[7].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            # preds[:,1]=-100
            eval_preds = np.argmax(eval_preds, axis=1)
        else:
            raise ValueError("No other `output_mode` for XNLI.")
        # print(preds.shape)
        # print(out_label_ids.shape)
        # print(preds[:10])
        # print(out_label_ids[:10])
        valacc=float(np.sum(np.equal(eval_preds,out_label_ids)))/float(eval_preds.shape[0])
        logger.info("Validation set accuracy: %s", str(valacc))
        # result = compute_metrics(eval_task, preds, out_label_ids)
        # results.update(result)

        logger.info("***** Eval results {} *****".format(prefix))
        # for key in sorted(result.keys()):
        #     logger.info("  %s = %s", key, str(result[key]))
        eval_label_list = GLUECoSNLIProcessor(language=args.language, train_language=args.train_language).get_labels()
        # print(label_list)
        eval_pred_labels = [eval_label_list[x] for x in eval_preds]

    return eval_pred_labels,valacc


def load_and_cache_examples_nli(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = GLUECoSNLIProcessor(language=args.language, train_language=args.train_language)
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "eng_nli_ver_qanlimtl_cached_{}_{}_{}_{}_{}".format(
            "test" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
            str(args.train_language if (not evaluate and args.train_language is not None) else args.language),
        ),
    )
    cached_features_file_rom = os.path.join(
        args.data_dir,
        "roman_nli_ver_qanlimtl_cached_{}_{}_{}_{}_{}".format(
            "test" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
            str(args.train_language if (not evaluate and args.train_language is not None) else args.language),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache and evaluate==False:
        logger.info("Loading features from cached file %s", cached_features_file)
        features1 = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s",os.path.join(args.data_dir,'english_MNLI'))
        # print("hey")
        label_list = processor.get_labels()
        examples1 = (
            processor.get_test_examples(os.path.join(args.data_dir,'english_MNLI')) if evaluate else processor.get_train_examples(os.path.join(args.data_dir,'english_MNLI'))
        )
        # print(examples[0])
        # print(len(examples[0].text_a.split(' '))+len(examples[0].text_b.split(' ')))
        features1 = convert_examples_to_features(
            examples1, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode,
        )
        # print(features[0])
        # print(len(features[0].input_ids))
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features1, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features1], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features1], dtype=torch.long)
    #print(features[0])
    if args.model_type=='bert':
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features1], dtype=torch.long)
    else:
        all_token_type_ids = torch.tensor([tokenizer.create_token_type_ids_from_sequences(f.attention_mask) for f in features1], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features1], dtype=torch.long)
    else:
        raise ValueError("No other `output_mode` for XNLI.")

    if os.path.exists(cached_features_file_rom) and not args.overwrite_cache and evaluate==False:
        logger.info("Loading features from cached file %s", cached_features_file_rom)
        features2 = torch.load(cached_features_file_rom)

    else:
        logger.info("Creating romanised features from dataset file at %s", os.path.join(args.data_dir,'romanised_hindi_MNLI'))
            # print("hey")
        # label_list = processor.get_labels()
        examples2 = (
            processor.get_test_examples(os.path.join(args.data_dir,'romanised_hindi_MNLI')) if evaluate else processor.get_train_examples(os.path.join(args.data_dir,'romanised_hindi_MNLI'))
        )
        # print(examples[0])
        # print(len(examples[0].text_a.split(' '))+len(examples[0].text_b.split(' ')))
        features2 = convert_examples_to_features(
            examples2, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode,
        )
        # print(features[0])
        # print(len(features[0].input_ids))
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file_rom)
            torch.save(features2, cached_features_file_rom)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids2 = torch.tensor([f.input_ids for f in features2], dtype=torch.long)
    all_attention_mask2 = torch.tensor([f.attention_mask for f in features2], dtype=torch.long)
    #print(features[0])
    if args.model_type=='bert':
        all_token_type_ids2 = torch.tensor([f.token_type_ids for f in features2], dtype=torch.long)
    else:
        all_token_type_ids2 = torch.tensor([tokenizer.create_token_type_ids_from_sequences(f.attention_mask) for f in features2], dtype=torch.long)
    if output_mode == "classification":
        all_labels2 = torch.tensor([f.label for f in features2], dtype=torch.long)
    else:
        raise ValueError("No other `output_mode` for XNLI.")
    # print(all_input_ids.shape)
    # print(all_input_ids2.shape)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels,all_input_ids2, all_attention_mask2, all_token_type_ids2, all_labels2)
    return dataset

def load_and_cache_examples_squad(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    qa_input_dir = args.qa_data_dir if args.qa_data_dir else "."
    qa_cached_features_file = os.path.join(
        qa_input_dir,
        "qa_ver_qanli_mtl_cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.qa_max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if (os.path.exists(qa_cached_features_file) and not args.overwrite_cache and evaluate==False):
        logger.info("Loading features from cached file %s", qa_cached_features_file)
        features_and_dataset = torch.load(qa_cached_features_file)
        qa_features, qa_dataset, qa_examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating qa features from dataset file at %s", qa_input_dir)

        if not args.qa_data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            qa_examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            qa_processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                qa_examples = qa_processor.get_dev_examples(args.qa_data_dir, filename=args.predict_file)
            else:
                qa_examples = qa_processor.get_train_examples(args.qa_data_dir, filename=args.train_file)

        qa_features, qa_dataset = squad_convert_examples_to_features(
            examples=qa_examples,
            tokenizer=tokenizer,
            max_seq_length=args.qa_max_seq_length,
            doc_stride=args.qa_doc_stride,
            max_query_length=args.qa_max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", qa_cached_features_file)
            torch.save({"features": qa_features, "dataset": qa_dataset, "examples": qa_examples}, qa_cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return qa_dataset, qa_examples, qa_features
    return qa_dataset
def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--qa_save_folder",
        default=None,
        type=str,
        required=False,
        help="The location to save best qa model",
    )
    parser.add_argument(
        "--nli_save_folder",
        default=None,
        type=str,
        required=False,
        help="The location to save best qa model",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--qa_data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir for QA. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: ",
    )
    parser.add_argument(
        "--language",
        default=None,
        type=str,
        required=True,
        help="Evaluation language. Also train language if `train_language` is set to None.",
    )
    parser.add_argument(
        "--train_language", default=None, type=str, help="Train language if is different of the evaluation language."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--qa_output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written for QA.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--qa_max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--qa_per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--qa_per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--nli_logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--qa_logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    
    parser.add_argument(
        "--qa_doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks for QA, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--qa_max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--save_model_qa", action="store_true", help="Save the model with best_qa_f1 on dev set during training")
    parser.add_argument("--save_model_nli", action="store_true", help="Save the model with best_nli_acc on dev set during training")
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    args = parser.parse_args()
    if args.save_model_nli and (not os.path.exists(args.nli_save_folder)):
        os.mkdir(args.nli_save_folder)
    if args.save_model_qa and (not os.path.exists(args.qa_save_folder)):
        os.mkdir(args.qa_save_folder)
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
        print("&&& ",end=' ')
       # print(args.n_gpu)
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)

    # Prepare XNLI task
    args.task_name = "xnli"
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    # processor = processors[args.task_name](language=args.language, train_language=args.train_language)
    processor = GLUECoSNLIProcessor(language=args.language, train_language=args.train_language)
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
         # './big_corpus10_epochs/config.json',
        args.config_name if args.config_name else args.model_name_or_path,
        output_hidden_states=True
        # num_labels=num_labels,
        # finetuning_task=args.task_name,
        # cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = RobertaForQuestionAnswering.from_pretrained(
        args.model_name_or_path,
        #'./qa_bert_romanized_snap/pytorch_model.bin',
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    nli_layer = torch.nn.Sequential(torch.nn.Linear(768,2), torch.nn.LogSoftmax(dim=1))
    loss_nli_fn = torch.nn.NLLLoss()
    nli_layer.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        nli_dataset = load_and_cache_examples_nli(args, args.task_name, tokenizer, evaluate=False)
        qa_dataset = load_and_cache_examples_squad(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, nli_dataset, model,nli_layer,loss_nli_fn, tokenizer,qa_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        pred_labels,_ = evaluate(args, model, tokenizer,nli_layer,loss_nli_fn)# prefix=global_step)
        # result = dict((k, v) for k, v in result.items())
        # results.update(result)
        # with open('{}/test_predictions.txt'.format(args.output_dir), 'w') as f:
        #     f.write('\n'.join(pred_labels))

    return results


if __name__ == "__main__":
    print(os.environ["NVIDIA_VISIBLE_DEVICES"]) 
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    main()
