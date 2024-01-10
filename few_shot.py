
from tqdm import tqdm
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer, SoftVerbalizer, AutomaticVerbalizer
from openprompt.prompts import ManualTemplate


parser = argparse.ArgumentParser("")

parser.add_argument("--model", type=str, default='bertweet')
parser.add_argument("--model_name_or_path", default='vinai/bertweet-base')
parser.add_argument("--dataset_base_path",type=str)
parser.add_argument("--mode", type=str)

parser.add_argument("--shot", type=int, default=5)
parser.add_argument("--seed", type=int, default=144)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--dataset",type=str)

parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
parser.add_argument("--max_epochs", type=int, default=5)
parser.add_argument("--kptw_lr", default=0.06, type=float)
parser.add_argument("--pred_temp", default=1.0, type=float)
parser.add_argument("--max_token_split", default=-1, type=int)
args = parser.parse_args()

import random
this_run_unicode = str(random.randint(0, 1e10))

from openprompt.utils.reproduciblity import set_seed
set_seed(args.seed)
from openprompt.plms.mlm import MLMTokenizerWrapper
from transformers import AutoModelForMaskedLM, AutoTokenizer
model_name = args.model_name_or_path
plm = AutoModelForMaskedLM.from_pretrained(model_name)
WrapperClass = MLMTokenizerWrapper
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

dataset = {}
from utils import load_and_preprocess_dataset
dataset_base_path = args.dataset_base_path
dataset_path = f"{dataset_base_path}/{args.dataset}"
dataset['train'] = load_and_preprocess_dataset(
    dataset_path=dataset_path, 
    text_file_name="train_text.txt", 
    label_file_name="train_labels.txt"
)
dataset['validation'] = load_and_preprocess_dataset(
    dataset_path=dataset_path, 
    text_file_name="val_text.txt", 
    label_file_name="val_labels.txt"
)
dataset['test'] = load_and_preprocess_dataset(
    dataset_path=dataset_path, 
    text_file_name="test_text.txt", 
    label_file_name="test_labels.txt"
)

if args.dataset == "emoji":
    class_labels = [
        "❤️",  # _red_heart_
        "😍",  # _smiling_face_with_hearteyes_
        "😂",  # _face_with_tears_of_joy_
        "💕",  # _two_hearts_
        "🔥",  # _fire_
        "😊",  # _smiling_face_with_smiling_eyes_
        "😎",  # _smiling_face_with_sunglasses_
        "✨",  # _sparkles_
        "💙",  # _blue_heart_
        "😘",  # _face_blowing_a_kiss_
        "📷",  # _camera_
        "🇺🇸", # _United_States_
        "☀️",  # _sun_
        "💜",  # _purple_heart_
        "😉",  # _winking_face_
        "💯",  # _hundred_points_
        "😁",  # _beaming_face_with_smiling_eyes_
        "🎄",  # _Christmas_tree_
        "📸",  # _camera_with_flash_
        "😜"   # _winking_face_with_tongue_
    ]
elif args.dataset == "emotion":
    class_labels = [
        "anger",    # 0
        "joy",      # 1
        "optimism", # 2
        "sadness"   # 3
    ]
elif args.dataset == "hate":
    class_labels = [
        "not-hate",
        "hate"
    ]
elif args.dataset == "irony":
    class_labels = [
        "non_irony",
        "irony"
    ]
elif args.dataset == "offensive":
    class_labels = [
        "not-offensive",
        "offensive"
    ]
elif args.dataset == "sentiment":
    class_labels = [
        "negative",
        "neutral",
        "positive"
    ]
elif args.dataset == "stance_pre":
    class_labels = [
        "none",
        "against",
        "favor"
    ]
else:
    raise NotImplementedError
cutoff=0
max_seq_l = 128
batch_s = 20
mytemplate = ManualTemplate(tokenizer=tokenizer).from_file("manual_template.txt", choice=args.template_id)
if args.verbalizer == "kpt":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff, pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(f"knowledgeable_verbalizer/{args.dataset}.txt")
elif args.verbalizer == "manual":
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"manual_verbalizer/{args.dataset}.json")
elif args.verbalizer == "soft":
    myverbalizer = SoftVerbalizer(tokenizer, model=plm, classes=class_labels).from_file(f"manual_verbalizer/{args.dataset}.json")
elif args.verbalizer == "auto":
    myverbalizer = AutomaticVerbalizer(tokenizer, classes=class_labels)
# (contextual) calibration
if args.verbalizer in ["kpt","manual"]:
    if args.calibration or args.filter != "none":
        from openprompt.data_utils.data_sampler import FewShotSampler
        support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
        dataset['support'] = support_sampler(dataset['train'], seed=args.seed)

        # for example in dataset['support']:
        #     example.label = -1 # remove the labels of support set for clarification
        support_dataloader = PromptDataLoader(dataset=dataset["support"], template=mytemplate, tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
            batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
            truncate_method="tail")


from openprompt import PromptForClassification
use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()



# HP
# if args.calibration:
if args.verbalizer in ["kpt","manual"]:
    if args.calibration or args.filter != "none":
        org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
        from contextualize_calibration import calibrate
        # calculate the calibration logits
        cc_logits = calibrate(prompt_model, support_dataloader)
        print("the calibration logits is", cc_logits)
        print("origial label words num {}".format(org_label_words_num))

    if args.calibration:
        myverbalizer.register_calibrate_logits(cc_logits.mean(dim=0))
        new_label_words_num = [len(myverbalizer.label_words[i]) for i in range(len(class_labels))]
        print("After filtering, number of label words per class: {}".format(new_label_words_num))


    from filter_method import *
    if args.filter == "tfidf_filter":
        tfidf_filter(myverbalizer, cc_logits, class_labels)
    elif args.filter == "none":
        pass
    else:
        raise NotImplementedError


    # register the logits to the verbalizer so that the verbalizer will divide the calibration probability in producing label logits
#!!    # currently, only ManualVerbalizer and KnowledgeableVerbalizer support calibration.

from openprompt.data_utils.data_sampler import FewShotSampler
if args.mode == "low":
    sample_ratio = 0.1 if len(dataset['train']) < 5000 else (0.05 if len(dataset['train']) < 20000 else 0.01)
    shot=int((len(dataset['train']) * sample_ratio) / len(class_labels))
elif args.mode == "shot":# shot = 40
    shot = args.shot
if args.mode != "full":
    print(shot)
    sampler = FewShotSampler(num_examples_per_label=shot)
    dataset['train'] = sampler(dataset['train'], seed=args.seed)


train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
    batch_size=batch_s,shuffle=True, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

validation_dataloader = PromptDataLoader(dataset=dataset["validation"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
    batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
    batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")

from utils import evaluate
from transformers import  AdamW, get_linear_schedule_with_warmup
loss_func = torch.nn.CrossEntropyLoss()


def prompt_initialize(verbalizer, prompt_model, init_dataloader):
    dataloader = init_dataloader
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Init_using_{}".format("train")):
            batch = batch.cuda()
            logits = prompt_model(batch)
        verbalizer.optimize_to_initialize()


if args.verbalizer == "soft":


    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer_grouped_parameters2 = [
        {'params': prompt_model.verbalizer.group_parameters_1, "lr":3e-5},
        {'params': prompt_model.verbalizer.group_parameters_2, "lr":3e-4},
    ]


    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    optimizer2 = AdamW(optimizer_grouped_parameters2)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    scheduler2 = get_linear_schedule_with_warmup(
        optimizer2,
        num_warmup_steps=0, num_training_steps=tot_step)

elif args.verbalizer == "auto":
    prompt_initialize(myverbalizer, prompt_model, train_dataloader)

    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None

elif args.verbalizer == "kpt":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    # optimizer_grouped_parameters2 = [
    #     {'params': , "lr":1e-1},
    # ]
    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
    optimizer2 = AdamW(prompt_model.verbalizer.parameters(), lr=args.kptw_lr)
    # print(optimizer_grouped_parameters2)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    # scheduler2 = get_linear_schedule_with_warmup(
    #     optimizer2,
    #     num_warmup_steps=0, num_training_steps=tot_step)
    scheduler2 = None

elif args.verbalizer == "manual":
    no_decay = ['bias', 'LayerNorm.weight']

    # it's always good practice to set no decay to biase and LayerNorm parameters
    optimizer_grouped_parameters1 = [
        {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    # Using different optimizer for prompt parameters and model parameters

    optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)

    tot_step = len(train_dataloader) // args.gradient_accumulation_steps * args.max_epochs
    scheduler1 = get_linear_schedule_with_warmup(
        optimizer1,
        num_warmup_steps=0, num_training_steps=tot_step)

    optimizer2 = None
    scheduler2 = None


tot_loss = 0
log_loss = 0
best_val_acc = 0
for epoch in range(args.max_epochs):
    tot_loss = 0
    prompt_model.train()
    pbar = tqdm(train_dataloader, desc="Train")
    for step, inputs in enumerate(pbar):
        if use_cuda:
            inputs = inputs.cuda()
        logits = prompt_model(inputs)
        labels = inputs['label']
        loss = loss_func(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(prompt_model.parameters(), 1.0)
        tot_loss += loss.item()
        optimizer1.step()
        scheduler1.step()
        optimizer1.zero_grad()
        if optimizer2 is not None:
            optimizer2.step()
            optimizer2.zero_grad()
        if scheduler2 is not None:
            scheduler2.step()

    val = evaluate(prompt_model, validation_dataloader, desc="Valid")

torch.save(prompt_model.state_dict(),f"{this_run_unicode}.ckpt")
prompt_model.load_state_dict(torch.load(f"{this_run_unicode}.ckpt"))
prompt_model = prompt_model.cuda()
test = evaluate(prompt_model, test_dataloader, desc="Test")





content_write = "="*20+"\n"
content_write += f"dataset {args.dataset}\t"
content_write += f"temp {args.template_id}\t"
content_write += f"seed {args.seed}\t"
content_write += f"shot {args.shot}\t"
content_write += f"verb {args.verbalizer}\t"
content_write += f"cali {args.calibration}\t"
content_write += f"filt {args.filter}\t"
content_write += f"maxsplit {args.max_token_split}\t"
content_write += f"kptw_lr {args.kptw_lr}\t"
content_write += "\n"
content_write += f"test: {test}"
content_write += "\n\n"

print(content_write)

import os
result_file = f"result/{args.mode}/{args.dataset}.txt"
directory = os.path.dirname(result_file)

# Create the directory if it does not exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Now, open the file in append mode and write the content
with open(result_file, "a") as fout:
    fout.write(content_write)