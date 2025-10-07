import json
import re
import string
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct" 
DATA_PATH = "data/generate_data_for_rl_hotpot.json"  
OUTPUT_DIR = "./decomposition_model_ppo"    

def load_and_format_data(file_path):
    with open(file_path, 'r') as f:
        raw_data = json.load(f)
    
    formatted_data = []
    for item in raw_data:
        subtasks = item['subquestions']

        formatted_data.append({
            "question": item["question"],
            "subquestions": subtasks,
            "answer": item["answer"],
            "final_answer": item["final_answer"]
        })
    
    return formatted_data
raw_data = load_and_format_data(DATA_PATH)[:1500]

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def remove_prompt(text):
        return text.replace("answer:", " ")

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(remove_prompt(lower(s)))))
def f1_score(prediction, ground_truth_list):
    normalized_prediction = normalize_answer(prediction)  
    max_f1 = 0
    max_precision = 0
    max_recall = 0
    
    # for ground_truth in [ground_truth_list]:
    for ground_truth in ground_truth_list:
        normalized_ground_truth = normalize_answer(ground_truth)
        
        ZERO_METRIC = (0, 0, 0)
        
        if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            continue
        if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
            continue
            
        prediction_tokens = normalized_prediction.split()
        ground_truth_tokens = normalized_ground_truth.split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            continue
            
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        # print(f1)
        
        if f1 > max_f1:
            max_f1 = f1
            max_precision = precision
            max_recall = recall
    return max_f1
scores = []
for example in raw_data:
    scores.append(f1_score(example['final_answer'], [example['answer']]))

for i, item in enumerate(raw_data):
    item['score'] = float(scores[i]) 

    print(item['score'])


train_dataset = Dataset.from_list(raw_data)
split_datasets = train_dataset.train_test_split(test_size=0.2, shuffle=True, seed=42)
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]


rm_model_name = "microsoft/deberta-v3-base" 
# rm_model_name = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(rm_model_name)
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    texts = [f"Question: {q}\nDecomposition: {d}" for q, d in zip(examples['question'], examples['subquestions'])]
    model_inputs = tokenizer(texts, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = examples["score"] 
    return model_inputs

tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(preprocess_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    rm_model_name,
    num_labels=1, 
    problem_type="regression" 
)
model.config.pad_token_id = tokenizer.pad_token_id


training_args = TrainingArguments(
    output_dir="./reward_model_hotpot",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=50,
    logging_dir="./logs",
    eval_strategy="epoch", 
    save_strategy="epoch",
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_eval_dataset,
)

trainer.train()

model.save_pretrained("./trained_reward_model")
tokenizer.save_pretrained("./trained_reward_model")