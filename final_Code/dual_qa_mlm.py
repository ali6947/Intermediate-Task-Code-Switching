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
    BertForSequenceClassification,BertForMaskedLM,XLMRobertaForMaskedLM,RobertaForMaskedLM,
    BertTokenizer,
    XLMConfig, XLMForSequenceClassification, XLMTokenizer,
    XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer,
    get_linear_schedule_with_warmup,AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorForLanguageModeling, LineByLineTextDataset,
    squad_convert_examples_to_features,

)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

# from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import xnli_output_modes as output_modes
from transformers import xnli_processors as processors
from transformers import modeling_utils as mutils

QA_PROB=0.75
class GLUECoSNLIProcessor(processors['xnli']):
    def get_labels(self):
        return ["contradiction", "entailment"]


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

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


def train(args, qa_dataset, model, qa_layer, tokenizer,mlm_dataset):
    """ Train the model """
    # if args.local_rank in [-1, 0]:
    #     tb_writer = SummaryWriter()
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.mlm_batch_size = args.mlm_per_gpu_train_batch_size * max(1, args.n_gpu)
    qa_sampler = RandomSampler(qa_dataset) if args.local_rank == -1 else DistributedSampler(qa_dataset)
    mlm_sampler = RandomSampler(mlm_dataset) if args.local_rank == -1 else DistributedSampler(mlm_dataset)
    qa_dataloader = DataLoader(qa_dataset, sampler=qa_sampler, batch_size=args.train_batch_size)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    mlm_dataloader = DataLoader(mlm_dataset, collate_fn=data_collator.collate_batch, batch_size=args.mlm_batch_size)
    

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // ((len(nli_dataloader)+len(mlm_dataloader)) // args.gradient_accumulation_steps) + 1
    else:
        t_total = (len(qa_dataloader)+len(mlm_dataloader)) // args.gradient_accumulation_steps * args.num_train_epochs
        # m_total = len(mlm_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    t_total = ((len(qa_dataloader)+len(mlm_dataloader))// args.gradient_accumulation_steps * args.num_train_epochs)
    if args.model_type=='xlm-roberta':
        t_total*=4
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        {"params":[p for n,p in qa_layer.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay":args.weight_decay},
        {"params":[p for n,p in qa_layer.named_parameters() if any(nd in n for nd in no_decay)],"weight_decay":0.0}
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
    logger.info("  Num examples = %d", len(qa_dataset)+len(mlm_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size+args.mlm_per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        (args.train_batch_size+args.mlm_batch_size)
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
    qa_layer.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    best_f1=0
    for _ in train_iterator:
        steps_done=0
        # nli_fin=False
        # mlm_fin=False
        drqa,nrqa,lossqa,lossmlm,iters,drmlm =0,0,0,0,0,0
        # nli_epoch_iterator = iter(nli_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        # mlm_epoch_iterator = iter(mlm_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        qa_dataloader = DataLoader(qa_dataset, sampler=qa_sampler, batch_size=args.train_batch_size)
        mlm_dataloader = DataLoader(mlm_dataset, collate_fn=data_collator.collate_batch, batch_size=args.mlm_batch_size)
        qa_epoch_iterator = iter(qa_dataloader)
        mlm_epoch_iterator = iter(mlm_dataloader)
        pbar = tqdm(total=t_total)
        qa_steps=0
        while steps_done<t_total:
            r = random.uniform(0, 1)
            if r<QA_PROB:
                qa_steps+=1
                # print(nli_steps)
                for icnt_qa in range(args.gradient_accumulation_steps):
                    batch_qa=next(qa_epoch_iterator,None)
                    if batch_qa==None:
                        # nli_epoch_iterator.close()
                        qa_dataloader = DataLoader(qa_dataset, sampler=qa_sampler, batch_size=args.train_batch_size)
                        qa_epoch_iterator = iter(qa_dataloader)
                        batch_qa=next(qa_epoch_iterator,None)
                    steps_done+=1
                    pbar.update(1)
                    # for step, batch in enumerate(epoch_iterator):
                    # # Skip past any already trained steps if resuming training
                    # if steps_trained_in_current_epoch > 0:
                    #     steps_trained_in_current_epoch -= 1
                    #     continue

                    model.train()
                    qa_layer.train()
                    batch_qa = tuple(t.to(args.device) for t in batch_qa)
                    # print(batch_nli[0].shape)
                    # print(batch_nli[1].shape)
                    # inputs_nli = {"input_ids": batch_nli[0], "attention_mask": batch_nli[1]}#, "labels": batch[3]}
                    inputs_qa = {
                        "input_ids": batch_qa[0],
                        "attention_mask": batch_qa[1],
                        "token_type_ids": batch_qa[2],
                    }
                    # inputs_qa = {"input_ids": torch.cat((batch_qa[0],batch_qa[4]),0), "attention_mask": torch.cat((batch_qa[1],batch_qa[5]),0)}
                     #"labels": torch.cat((batch_nli[3],batch_nli[7]),0)}

                    # if args.model_type != "distilbert":
                    #     inputs_nli["token_type_ids"] = (
                    #         batch_nli[2] if args.model_type in ["bert"] else None
                    #     )  # XLM and DistilBERT don't use segment_ids
                    if args.model_type != "distilbert":
                        del inputs_qa["token_type_ids"]

                    outputs_qa = model(**inputs_qa)
                    # print(len(outputs_nli[-1]))
                    train_hidden = outputs_qa[-1][-1]  # model outputs are always tuple in transformers (see doc)
                    # print(hidden.shape)
                    # inputs_squad = {
                    #     "hidden_states": hidden,
                    #     "start_positions": batch_qa[3],
                    #     "end_positions": batch_qa[4],
                    # }
                    start_positions=batch_qa[3]
                    end_positions=batch_qa[4]
                    logits = qa_layer(train_hidden)
                    train_start_logits, train_end_logits = logits.split(1, dim=-1)
                    train_start_logits = train_start_logits.squeeze(-1)
                    train_end_logits = train_end_logits.squeeze(-1)

                    if args.model_type in ["xlnet", "xlm"]:
                        inputs_squad.update({"cls_index": batch_qa[5], "p_mask": batch_qa[6]})
                        if args.version_2_with_negative:
                            inputs_squad.update({"is_impossible": batch_qa[7]})
                        if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                            inputs_squad.update(
                                {"langs": (torch.ones(hidden.shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                            )
                    total_loss = None
                    if start_positions is not None and end_positions is not None:
                        # If we are on multi-GPU, split add a dimension
                        if len(start_positions.size()) > 1:
                            start_positions = start_positions.squeeze(-1)
                        if len(end_positions.size()) > 1:
                            end_positions = end_positions.squeeze(-1)
                        # sometimes the start/end positions are outside our model inputs, we ignore these terms
                        ignored_index = train_start_logits.size(1)
                        start_positions.clamp_(0, ignored_index)
                        end_positions.clamp_(0, ignored_index)

                        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
                        start_loss = loss_fct(train_start_logits, start_positions)
                        end_loss = loss_fct(train_end_logits, end_positions)
                        total_loss = (start_loss + end_loss) / 2
                    # squad_pred = squad_head(**inputs_squad)
                    # loss_qa = squad_pred[0]
                    # loss_nli = loss_pred
                        lossqa+=total_loss.item()
                        if args.n_gpu > 1:
                            total_loss = total_loss.mean()  # mean() to average on multi-gpu parallel training
                        if args.gradient_accumulation_steps > 1:
                            total_loss / args.gradient_accumulation_steps
                        total_loss.backward()
                        tr_loss += total_loss.item()

                    iters+=1
                    # if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                qa_layer.zero_grad()
                global_step += 1
                       
    
                    # logging_loss = tr_loss

                    # if args.max_steps > 0 and global_step > args.max_steps:
                    #     epoch_iterator.close()
                    #     break
            else:
                # drnli,nrnli,lossnli,iters =0,0,0,0
                for icnt in range(args.gradient_accumulation_steps):
                    batch_mlm=next(mlm_epoch_iterator,None)
                    if batch_mlm==None:
                        # mlm_epoch_iterator.close()
                        mlm_dataloader = DataLoader(mlm_dataset, collate_fn=data_collator.collate_batch, batch_size=args.mlm_batch_size)
                        mlm_epoch_iterator = iter(mlm_dataloader)
                        batch_mlm=next(mlm_epoch_iterator,None)
                    pbar.update(1)
                    steps_done+=1

                    model.train()
                    mlmsz=1
                    for i in batch_mlm.keys():
                        batch_mlm[i]=batch_mlm[i].to(args.device)
                        mlmsz=batch_mlm[i].shape[0]
                    # batch = tuple(t.to(args.device) for t in batch)
                    inputs_mlm=batch_mlm
                    # inputs = {"input_ids": batch[0], "attention_mask": batch[1]}#, "labels": batch[3]}
                    # inputs = {"input_ids": torch.cat((batch[0],batch[4]),0), "attention_mask": torch.cat((batch[1],batch[5]),0), "labels": torch.cat(\
                    #         (batch[3],batch[7]),0)}

                    # if args.model_type != "distilbert":
                    #     inputs["token_type_ids"] = (
                    #         batch[2] if args.model_type in ["bert"] else None
                    #     )  # XLM and DistilBERT don't use segment_ids
                    
                    outputs_mlm = model(**inputs_mlm)

                    loss_mlm=outputs_mlm[0]

                    lossmlm+=loss_mlm.item()
                    drmlm+=mlmsz
                    iters+=1

                    if args.n_gpu > 1:
                        loss_mlm = loss_mlm.mean()  # mean() to average on multi-gpu parallel training
                    if args.gradient_accumulation_steps > 1:
                        loss_mlm = loss_mlm / args.gradient_accumulation_steps

                    loss_mlm.backward()

                    tr_loss += loss_mlm.item()
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
        # if args.max_steps > 0 and global_step > args.max_steps:
        #     train_iterator.close()
        #     break
            if global_step%args.logging_steps==0:
                    results = evaluate(args, model, tokenizer, qa_layer)
                    if results['f1']>best_f1:
                        best_f1=results['f1']
                        if args.save_model:
                            logger.info('new best f1 at %s, new model saved!',str(best_f1))
                            model.save_pretrained(args.save_folder) #pretrained_on_mlm_eng_squad 
                            torch.save(qa_layer.state_dict(),os.path.join(args.save_folder, "qa_layer.bin"))
                            tokenizer.save_pretrained(args.save_folder)
                            torch.save(args, os.path.join(args.save_folder, "training_args.bin"))
                        else:
                            logger.info('new best f1 at %s, no model saved',str(best_f1))
                    else:
                        logger.info('best f1 still at %s',str(best_f1))

        # logger.info('nli acc=%s ',str(nrnli/drnli))
        # logger.info('loss nli=%s',str(lossnli/iters))
        # logger.info('loss mlm=%s',str(lossmlm/drmlm))

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, qa_layer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        qa_layer.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs_val = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
                del inputs["token_type_ids"]

            example_indices = batch[3]

            val_outputs_qa = model(**inputs_val)
            # start_positions=batch_qa[3]
            # end_positions=batch_qa[4]
            val_logits = qa_layer(val_outputs_qa[-1][-1])
            val_start_logits, val_end_logits = val_logits.split(1, dim=-1)
            val_start_logits = val_start_logits.squeeze(-1)
            val_end_logits = val_end_logits.squeeze(-1)

            # hidden = outputs_qa[-1][-1]  # model outputs are always tuple in transformers (see doc)
            # print(hidden.shape)
            # inputs_squad = {"hidden_states": hidden}

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            # outputs = squad_head(**inputs_squad)


        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            # output = [to_list(output[i]) for output in outputs]
            val_start_logits_i=to_list(val_start_logits[i])
            val_end_logits_i=to_list(val_end_logits[i])
            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            # if len(output) >= 5 and False:
            if False:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                # val_start_logits, end_logits = output
                result = SquadResult(unique_id, val_start_logits_i, val_end_logits_i)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")
    else:
        output_null_log_odds_file = None

    predictions = compute_predictions_logits(
        examples,
        features,
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
    results = squad_evaluate(examples, predictions)
    print(results)
    # print(results)
    # results = {}
    return results


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache or dataset file
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(
        input_dir,
        "cached_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if (os.path.exists(cached_features_file) and not args.overwrite_cache and False) :
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)

        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
            if evaluate:
                examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
            else:
                examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt",
            threads=args.threads,
        )

        # if args.local_rank in [-1, 0]:
        #     logger.info("Saving features into cached file %s", cached_features_file)
        #     torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--save_folder",
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
    parser.add_argument(
        "--mlm_max_seq_length",
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
    parser.add_argument("--mlm_per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training for mlm.")
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
    parser.add_argument("--mlm_datapath",default=None,type=str,required=True,help="Path to data for lm")
    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument("--save_model", action="store_true", help="Save the model with best_f1 on dev sest during training")
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
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
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
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
        print("&&& ",end=' ')
        print(args.n_gpu)
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
    # args.task_name = "xnli"
    # if args.task_name not in processors:
    #     raise ValueError("Task not found: %s" % (args.task_name))
    # processor = processors[args.task_name](language=args.language, train_language=args.train_language)
    # processor = GLUECoSNLIProcessor(language=args.language, train_language=args.train_language)
    # args.output_mode = output_modes[args.task_name]
    # label_list = processor.get_labels()
    # num_labels = len(label_list)

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
    config_qa = config_class.from_pretrained(
         # './pretrainCS_bert_65k/config.json',
        args.config_name if args.config_name else args.model_name_or_path,
        # num_labels=num_labels,
        # finetuning_task=args.task_name,
        # cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # config_qa.update({'start_n_top':1,'end_n_top':1})
    # print(config_qa.to_dict())
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # tokenizer = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
    model =model_class.from_pretrained(
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
    # squad_head = mutils.SQuADHead(config=config_qa)
    # loss_nli_fn = torch.nn.NLLLoss()
    # squad_head.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    qa_layer=torch.nn.Linear(768,2)
    qa_layer.to(args.device)
    
    # Training
    if args.do_train:
        qa_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        logger.info('Creating MLM features')
        print(args.mlm_datapath)
        mlm_dataset=LineByLineTextDataset(tokenizer=tokenizer, file_path=args.mlm_datapath, block_size=args.mlm_max_seq_length)
        # mlm_dataset=MaskedLMDataset(args.mlm_datapath, tokenizer,args.max_seq_length)
        global_step, tr_loss = train(args, qa_dataset, model, qa_layer, tokenizer,mlm_dataset)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        pass
        # pred_labels,_ = evaluate(args, model, tokenizer,squad_head)# prefix=global_step)
        # result = dict((k, v) for k, v in result.items())
        # results.update(result)
        # with open('{}/test_predictions.txt'.format(args.output_dir), 'w') as f:
        #     f.write('\n'.join(pred_labels))

    return results


if __name__ == "__main__":
    print(os.environ["NVIDIA_VISIBLE_DEVICES"]) 
    print(os.environ["CUDA_VISIBLE_DEVICES"])
    main()
