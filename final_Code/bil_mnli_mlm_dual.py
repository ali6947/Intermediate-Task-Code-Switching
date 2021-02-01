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

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    BertConfig,
    BertForSequenceClassification,BertForMaskedLM,XLMRobertaForMaskedLM,
    BertTokenizer,
    XLMConfig, XLMForSequenceClassification, XLMTokenizer,
    XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling, LineByLineTextDataset

)
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import xnli_output_modes as output_modes
from transformers import xnli_processors as processors

NLI_PROB=0.6
class GLUECoSNLIProcessor(processors['xnli']):
    def get_labels(self):
        return ["contradiction", "entailment"]


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "xlm-roberta": (XLMRobertaConfig, XLMRobertaForMaskedLM, XLMRobertaTokenizer),
    # "joeddav-xlm-roberta-large-xnli":()
}


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


def train(args, nli_dataset, model, nli_layer,loss_nli_fn, tokenizer,mlm_dataset):
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.mlm_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    nli_sampler = RandomSampler(nli_dataset) if args.local_rank == -1 else DistributedSampler(nli_dataset)
    mlm_sampler = RandomSampler(mlm_dataset) if args.local_rank == -1 else DistributedSampler(mlm_dataset)
    nli_dataloader = DataLoader(nli_dataset, sampler=nli_sampler, batch_size=args.train_batch_size)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    mlm_dataloader = DataLoader(mlm_dataset, collate_fn=data_collator.collate_batch, batch_size=args.mlm_batch_size)
    

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // ((len(nli_dataloader)+len(mlm_dataloader)) // args.gradient_accumulation_steps) + 1
    else:
        t_total = (len(nli_dataloader)+len(mlm_dataloader)) // args.gradient_accumulation_steps * args.num_train_epochs
        # m_total = len(mlm_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    t_total = ((2*len(nli_dataloader)+len(mlm_dataloader))// args.gradient_accumulation_steps * args.num_train_epochs)
    if args.model_type=='xlm-roberta':
        t_total*=4
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        # {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
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
    print(t_total)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", 2*len(nli_dataset)+len(mlm_dataset))
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

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    best_val_acc=0
    for _ in train_iterator:
        steps_done=0
        # nli_fin=False
        # mlm_fin=False
        drnli,nrnli,lossnli,lossmlm,iters,drmlm =0,0,0,0,0,0
        # nli_epoch_iterator = iter(nli_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        # mlm_epoch_iterator = iter(mlm_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        nli_dataloader = DataLoader(nli_dataset, sampler=nli_sampler, batch_size=args.train_batch_size)
        mlm_dataloader = DataLoader(mlm_dataset, collate_fn=data_collator.collate_batch, batch_size=args.mlm_batch_size)
        nli_epoch_iterator = iter(nli_dataloader)
        mlm_epoch_iterator = iter(mlm_dataloader)
        pbar = tqdm(total=t_total)
        nli_steps=0
        while steps_done<t_total:
            r = random.uniform(0, 1)
            if r<NLI_PROB:
                nli_steps+=1
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
                    # for step, batch in enumerate(epoch_iterator):
                    # # Skip past any already trained steps if resuming training
                    # if steps_trained_in_current_epoch > 0:
                    #     steps_trained_in_current_epoch -= 1
                    #     continue

                    model.train()
                    batch_nli = tuple(t.to(args.device) for t in batch_nli)
                    # print(batch_nli[0].shape)
                    # print(batch_nli[1].shape)
                    # inputs_nli = {"input_ids": batch_nli[0], "attention_mask": batch_nli[1]}#, "labels": batch[3]}
                    inputs_nli = {"input_ids": torch.cat((batch_nli[0],batch_nli[4]),0), "attention_mask": torch.cat((batch_nli[1],batch_nli[5]),0)}
                     #"labels": torch.cat((batch_nli[3],batch_nli[7]),0)}

                    # if args.model_type != "distilbert":
                    #     inputs_nli["token_type_ids"] = (
                    #         batch_nli[2] if args.model_type in ["bert"] else None
                    #     )  # XLM and DistilBERT don't use segment_ids
                    if args.model_type != "distilbert":
                        inputs_nli["token_type_ids"] = (
                            # batch_nli[2] if args.model_type in ["bert"] else None
                            torch.cat((batch_nli[2],batch_nli[6]),0) if args.model_type in ["bert"] else None
                        )  # XLM and DistilBERT don't use segment_ids
                    outputs_nli = model(**inputs_nli)
                    # print(len(outputs_nli[-1]))
                    hidden = outputs_nli[-1][-1]  # model outputs are always tuple in transformers (see doc)
                    # print(hidden.shape)
                    nli_pred = nli_layer(hidden[:, 0, :])
                    _,nli_labels=torch.max(nli_pred,-1)
                    
                    # nrnli+=torch.eq(nli_labels,batch_nli[3]).sum().item()
                    # drnli+=float(batch_nli[3].shape[0])
                    # loss_nli = loss_nli_fn(nli_pred, batch_nli[3])
                    nrnli+=torch.eq(nli_labels,torch.cat((batch_nli[3],batch_nli[7]),0)).sum().item()
                    drnli+=float(batch_nli[3].shape[0])+float(batch_nli[7].shape[0])
                    loss_nli = loss_nli_fn(nli_pred, torch.cat((batch_nli[3],batch_nli[7]),0))
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
    
                    # logging_loss = tr_loss

                    # if args.max_steps > 0 and global_step > args.max_steps:
                    #     epoch_iterator.close()
                    #     break
            else:
                # drnli,nrnli,lossnli,iters =0,0,0,0
                for icnt in range(args.gradient_accumulation_steps):
                    batch=next(mlm_epoch_iterator,None)
                    if batch==None:
                        # mlm_epoch_iterator.close()
                        mlm_dataloader = DataLoader(mlm_dataset, collate_fn=data_collator.collate_batch, batch_size=args.mlm_batch_size)
                        mlm_epoch_iterator = iter(mlm_dataloader)
                        batch=next(mlm_epoch_iterator,None)
                    pbar.update(1)
                    steps_done+=1
                # for step, batch in enumerate(epoch_iterator):
                #     # Skip past any already trained steps if resuming training
                # if steps_trained_in_current_epoch > 0:
                #     steps_trained_in_current_epoch -= 1
                #     continue

                    model.train()
                    mlmsz=1
                    for i in batch.keys():
                        batch[i]=batch[i].to(args.device)
                        mlmsz=batch[i].shape[0]
                    # batch = tuple(t.to(args.device) for t in batch)
                    inputs=batch
                    # inputs = {"input_ids": batch[0], "attention_mask": batch[1]}#, "labels": batch[3]}
                    # inputs = {"input_ids": torch.cat((batch[0],batch[4]),0), "attention_mask": torch.cat((batch[1],batch[5]),0), "labels": torch.cat(\
                    #         (batch[3],batch[7]),0)}

                    # if args.model_type != "distilbert":
                    #     inputs["token_type_ids"] = (
                    #         batch[2] if args.model_type in ["bert"] else None
                    #     )  # XLM and DistilBERT don't use segment_ids
                    
                    outputs = model(**inputs)

                    loss=outputs[0]

                    lossmlm+=loss.item()
                    drmlm+=mlmsz
                    iters+=1

                    if args.n_gpu > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training
                    if args.gradient_accumulation_steps > 1:
                        loss = loss / args.gradient_accumulation_steps

                    loss.backward()
                    tr_loss += loss.item()
            
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
  
            if (global_step)%args.logging_steps==0:
                results,val_acc = evaluate(args, model, tokenizer, nli_layer, loss_nli_fn)
                if val_acc>best_val_acc:
                    logger.info('new best val acc at %s',str(val_acc))
                    best_val_acc=val_acc
                    if args.save_model:
                        logger.info('saving model')
                        model.save_pretrained(args.save_folder)
                        tokenizer.save_pretrained(args.save_folder)
                        torch.save(nli_layer.state_dict(),os.path.join(args.save_folder,"nli_layer.bin"))
                        torch.save(args, os.path.join(args.save_folder, "training_args.bin"))
                else:
                    logger.info('best val acc still at %s',str(best_val_acc))
        logger.info('nli acc=%s ',str(nrnli/drnli))
        logger.info('loss nli=%s',str(lossnli/iters))
        logger.info('loss mlm=%s',str(lossmlm/drmlm))

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, nli_classifier, loss_nli_fn, prefix=""):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

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
                # eval_loss_pred = loss_nli_fn(eval_nli_pred, eval_batch[3])
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


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = GLUECoSNLIProcessor(language=args.language, train_language=args.train_language)
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_{}".format(
            "test" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
            str(args.train_language if (not evaluate and args.train_language is not None) else args.language),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache and False:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", os.path.join(args.data_dir,'english_MNLI'))
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
        # if args.local_rank in [-1, 0]:
        #     logger.info("Saving features into cached file %s", cached_features_file)
        #     torch.save(features1, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features1], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features1], dtype=torch.long)
    if args.model_type=='bert':
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features1], dtype=torch.long)
    else:
        all_token_type_ids = torch.tensor([tokenizer.create_token_type_ids_from_sequences(f.attention_mask) for f in features1], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features1], dtype=torch.long)
    else:
        raise ValueError("No other `output_mode` for XNLI.")


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
    # if args.local_rank in [-1, 0]:
    #     logger.info("Saving features into cached file %s", cached_features_file)
    #     torch.save(features2, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids2 = torch.tensor([f.input_ids for f in features2], dtype=torch.long)
    all_attention_mask2 = torch.tensor([f.attention_mask for f in features2], dtype=torch.long)
    
    if args.model_type=='bert':
        all_token_type_ids2 = torch.tensor([f.token_type_ids for f in features2], dtype=torch.long)
    else:
        all_token_type_ids2 = torch.tensor([tokenizer.create_token_type_ids_from_sequences(f.attention_mask) for f in features2], dtype=torch.long)
    if output_mode == "classification":
        all_labels2 = torch.tensor([f.label for f in features2], dtype=torch.long)
    else:
        raise ValueError("No other `output_mode` for XNLI.")

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels,all_input_ids2, all_attention_mask2, all_token_type_ids2, all_labels2)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--save_folder",
        default=None,
        type=str,
        required=False,
        help="The location to save model.",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
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
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--save_model", action="store_true", help="Whether to save current best model.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
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

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
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
    parser.add_argument("--mlm_datapath",default=None,type=str,required=True,help="Path to data for mlm")
    args = parser.parse_args()
    if args.save_model and (not os.path.exists(args.save_folder)):
        os.mkdir(args.save_folder)
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
         # './pretrainCS_bert_65k/config.json',
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
    # tokenizer = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
    model=model_class.from_pretrained(
         # './pretrainCS_bert_65k/pytorch_model.bin',
        args.model_name_or_path,
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
        nli_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        logger.info('Creating MLM features')
        print(args.mlm_datapath)
        mlm_dataset=LineByLineTextDataset(tokenizer=tokenizer, file_path=args.mlm_datapath, block_size=args.max_seq_length)
        # mlm_dataset=MaskedLMDataset(args.mlm_datapath, tokenizer,args.max_seq_length)
        global_step, tr_loss = train(args, nli_dataset, model,nli_layer,loss_nli_fn, tokenizer,mlm_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
       
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        pred_labels,_ = evaluate(args, model, tokenizer,nli_layer,loss_nli_fn)# prefix=global_step)
        

    return results


if __name__ == "__main__":
    print(os.environ["NVIDIA_VISIBLE_DEVICES"]) 
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    main()
