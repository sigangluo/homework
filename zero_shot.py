

from tqdm import tqdm
import torch
from openprompt.data_utils.utils import InputExample
import argparse
import numpy as np

from openprompt import PromptDataLoader
from openprompt.prompts import ManualVerbalizer, KnowledgeableVerbalizer
from openprompt.prompts import ManualTemplate


parser = argparse.ArgumentParser("")
parser.add_argument("--model", type=str, default='bertweet')
parser.add_argument("--model_name_or_path", default='vinai/bertweet-base')
parser.add_argument("--dataset_base_path",type=str)
parser.add_argument("--seed", type=int, default=144)

parser.add_argument("--plm_eval_mode", action="store_true")

parser.add_argument("--verbalizer", type=str)
parser.add_argument("--calibration", action="store_true")
parser.add_argument("--nocut", action="store_true")
parser.add_argument("--filter", default="none", type=str)
parser.add_argument("--template_id", type=int)
parser.add_argument("--max_token_split", default=-1, type=int)
parser.add_argument("--dataset",type=str)
parser.add_argument("--write_filter_record", action="store_true")
args = parser.parse_args()

from openprompt.utils.reproduciblity import set_seed
set_seed(args.seed)
if args.model == "bertweet":
    from openprompt.plms.mlm import MLMTokenizerWrapper
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    model_name = args.model_name_or_path
    plm = AutoModelForMaskedLM.from_pretrained(model_name)
    WrapperClass = MLMTokenizerWrapper
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
else:
    from openprompt.plms import load_plm
    plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

dataset = {}
from utils import load_and_preprocess_dataset,evaluate
dataset_base_path = args.dataset_base_path
dataset_path = f"{dataset_base_path}/{args.dataset}"
dataset['train'] = load_and_preprocess_dataset(
    dataset_path=dataset_path, 
    text_file_name="train_text.txt", 
    label_file_name="train_labels.txt"
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
print(mytemplate)

if args.verbalizer == "kpt":
    myverbalizer = KnowledgeableVerbalizer(tokenizer, classes=class_labels, candidate_frac=cutoff, pred_temp=args.pred_temp, max_token_split=args.max_token_split).from_file(f"knowledgeable_verbalizer/{args.dataset}.txt")
elif args.verbalizer == "manual":
    myverbalizer = ManualVerbalizer(tokenizer, classes=class_labels).from_file(f"manual_verbalizer/{args.dataset}.json")
elif args.verbalizer == "soft":
    raise NotImplementedError
elif args.verbalizer == "auto":
    raise NotImplementedError

# (contextual) calibration
if args.calibration:
    from openprompt.data_utils.data_sampler import FewShotSampler
    support_sampler = FewShotSampler(num_examples_total=200, also_sample_dev=False)
    dataset['support'] = support_sampler(dataset['train'], seed=args.seed)

    for example in dataset['support']:
        example.label = -1 # remove the labels of support set for clarification
    support_dataloader = PromptDataLoader(dataset=dataset["support"], template=mytemplate, tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
        batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
        truncate_method="tail")


from openprompt import PromptForClassification
use_cuda = True
prompt_model = PromptForClassification(plm=plm,template=mytemplate, verbalizer=myverbalizer, freeze_plm=False, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()


myrecord = ""
# HP
if args.calibration:
    org_label_words_num = [len(prompt_model.verbalizer.label_words[i]) for i in range(len(class_labels))]
    from contextualize_calibration import calibrate
    # calculate the calibration logits
    cc_logits = calibrate(prompt_model, support_dataloader)
    print("the calibration logits is", cc_logits)
    myrecord += "Phase 1 {}\n".format(org_label_words_num)

    myverbalizer.register_calibrate_logits(cc_logits.mean(dim=0))
    new_label_words_num = [len(myverbalizer.label_words[i]) for i in range(len(class_labels))]
    myrecord += "Phase 2 {}\n".format(new_label_words_num)


    from filter_method import *
    if args.filter == "tfidf_filter":
        record = tfidf_filter(myverbalizer, cc_logits, class_labels)
        myrecord += record
    elif args.filter == "none":
        pass
    else:
        raise NotImplementedError


    # register the logits to the verbalizer so that the verbalizer will divide the calibration probability in producing label logits
    # currently, only ManualVerbalizer and KnowledgeableVerbalizer support calibration.
print(myrecord)

# zero-shot test
test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer,
    tokenizer_wrapper_class=WrapperClass, max_seq_length=max_seq_l, decoder_max_length=3,
    batch_size=batch_s,shuffle=False, teacher_forcing=False, predict_eos_token=False,
    truncate_method="tail")
evaluate(prompt_model, test_dataloader, desc="Test")


  # roughly ~0.853 when using template 0
