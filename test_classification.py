seed=144
from openprompt.utils.reproduciblity import set_seed
set_seed(seed)
dataset_type = "emoji"
batch_size = 20
num_class_map = {"emoji":20, "emotion":4, "hate":2, "irony":2, "offensive":2, "sentiment":3, "stance_pre":3}
class_num = num_class_map[dataset_type]
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset_base_path = "/mnt/nvm_data/guest24/sigangluo/tweeteval/datasets"
dataset_path = f"{dataset_base_path}/{dataset_type}"
from utils import load_and_preprocess_dataset_2
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base')
train_all_dataset = load_and_preprocess_dataset_2(
    dataset_path=dataset_path, 
    text_file_name="train_text.txt", 
    label_file_name="train_labels.txt",
    tokenizer=tokenizer
)
# sample_ratio = 0.1 if len(train_all_dataset) < 5000 else (0.05 if len(train_all_dataset) < 20000 else 0.01)
# shot=int((len(train_all_dataset) * sample_ratio) / class_num)
shot = 40
import random
def sample_from_dataset(dataset, shot):
    # Grouping the dataset by labels
    categorized_data = {}
    for item in dataset:
        label = item['label']
        if label not in categorized_data:
            categorized_data[label] = []
        categorized_data[label].append(item)

    # Sampling 'shot' items from each category
    sampled_data = []
    for label, items in categorized_data.items():
        sampled_data.extend(random.sample(items, min(shot, len(items))))

    return sampled_data
train_dataset = sample_from_dataset(train_all_dataset,shot)
# train_dataset = train_all_dataset
print(len(train_dataset))
# from sklearn.model_selection import train_test_split
# # 提取标签作为层次抽样的依据
# labels = [item['label'] for item in train_all_dataset]
# # 使用 stratify 参数进行层次抽样
# _, train_dataset = train_test_split(train_all_dataset, test_size=sample_ratio, stratify=labels)
# print(len(train_dataset))
test_dataset = load_and_preprocess_dataset_2(
    dataset_path=dataset_path, 
    text_file_name="test_text.txt", 
    label_file_name="test_labels.txt",
    tokenizer=tokenizer
)
valid_dataset = load_and_preprocess_dataset_2(
    dataset_path=dataset_path, 
    text_file_name="val_text.txt", 
    label_file_name="val_labels.txt",
    tokenizer=tokenizer
)
print(train_dataset[0])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
import torch.nn as nn
from transformers import AutoModelForSequenceClassification
class ClassificationModel(nn.Module):
    def __init__(self, num_classes, frozen=False):
        super(ClassificationModel, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained('vinai/bertweet-base')
        self.model.num_labels = num_classes
        self.model.classifier.out_proj = nn.Linear(768, num_classes, bias=True)
        if frozen:
            for param in self.model.roberta.parameters():
                param.requires_grad = False

    def forward(self, x, mask):
        out = self.model(x, mask)
        return out['logits']
    
model_name = "vinai/bertweet-base"
model = ClassificationModel(class_num)
model.to(device)
loss_func = torch.nn.CrossEntropyLoss()
no_decay = ['bias', 'LayerNorm.weight']
# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters1 = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
from transformers import  AdamW, get_linear_schedule_with_warmup
optimizer1 = AdamW(optimizer_grouped_parameters1, lr=3e-5)
gradient_accumulation_steps = 1
max_epochs = 5
tot_step = len(train_loader) // gradient_accumulation_steps * max_epochs
scheduler1 = get_linear_schedule_with_warmup(
    optimizer1,
    num_warmup_steps=0, num_training_steps=tot_step)

tot_loss = 0
log_loss = 0
best_val_acc = 0

from utils import evaluate_2
from tqdm import tqdm
for epoch in range(max_epochs):
    tot_loss = 0
    model.train()
    pbar = tqdm(train_loader)
    for step, inputs in enumerate(pbar):
        input_ids, attention_mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)
        logits = model(input_ids, attention_mask)
        labels = inputs['label'].to(device)
        loss = loss_func(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        tot_loss += loss.item()
        optimizer1.step()
        scheduler1.step()
        optimizer1.zero_grad()

    val_acc = evaluate_2(model, val_loader, desc="Valid",device = device)

torch.save(model.state_dict(),"test.ckpt")
model.load_state_dict(torch.load("test.ckpt"))
model.to(device)
test_acc = evaluate_2(model, test_loader, desc="Test",device = device)

