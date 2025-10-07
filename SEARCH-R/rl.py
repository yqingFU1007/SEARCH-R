import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import re
import string
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import respond_to_batch
from peft import PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer, util
from llm_utils import custom_llm_llama_rl
import tempfile

sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()

def load_and_format_data(file_path):
    with open(file_path, 'r') as f:
        raw_data = json.load(f)
    
    formatted_data = []
    for item in raw_data:
        subtasks = item['subquestions']

        formatted_data.append({
            "question": item["question"],
            "subtasks": subtasks,
            "answer": item["answer"],
            "final_answer": item["final_answer"]
        })
    
    return formatted_data

def augment_data(example, tokenizer, aug_factor=3):
    augmented_examples = []

    base_format = format_training_example(example)
    augmented_examples.append(base_format)

    question = example["question"]
    rephrases = [
        f"Can you tell me {question.lower().replace('?', '')}?",
        f"I'm wondering {question.lower().replace('?', '')}?",
        f"Please explain {question.lower().replace('?', '')}?"
    ]
    
    for rephrase in rephrases[:aug_factor]:
        new_example = example.copy()
        new_example["question"] = rephrase
        augmented_examples.append(format_training_example(new_example))
    
  
    subtasks = example["subtasks"]
    
    
    if len(subtasks) > 0:
        new_subtasks = []
        for subq in subtasks:
           
            variants = [
                f"Determine {subq.lower().replace('?', '')}",
                f"Find out {subq.lower().replace('?', '')}",
                f"Identify {subq.lower().replace('?', '')}"
            ]
            new_subtasks.append(random.choice(variants))
        
        new_example = example.copy()
        new_example["subtasks"] = new_subtasks
        augmented_examples.append(format_training_example(new_example))
    
    return augmented_examples

def format_training_example(example):
    prompt = (
        """### System:\n 
        You are a helpful AI assistant that helps break down questions into minimal necessary sub-questions.
        Guidelines:
        1. Only break down the question if it requires finding and connecting multiple distinct pieces of information
        2. Each sub-question should target a specific, essential piece of information
        3. Avoid generating redundant or overlapping sub-questions
        4. For questions about impact/significance, focus on:
        - What was the thing/event
        - What was its impact/significance
        5. For comparison questions between two items (A vs B):
        - First identify the specific attribute being compared for each item
        - Then ask about that attribute for each item separately
        - For complex comparisons, add a final question to compare the findings
        6.**Logical Progression**:
        Sub-questions should have clear relationships, such as:
        - **Parallel**: Independent sub-questions that both contribute to answering the original question.
        
        Example:
        Original: "What are the causes and consequences of climate change on global ecosystems?"
        One of the sub-question list: ["What are the main causes of climate change?", "What are the major consequences of climate change on global ecosystems?"]
        - **Sequential**: Sub-questions that build upon each other step-by-step.
        Example:
        Original: "What university, founded in 1890, is known for its groundbreaking work in economics?"
        One of the sub-question list: ["Which universities were founded in 1890?", "Which of these universities is known for its groundbreaking work in economics?"]
        - **Comparative**: Questions that compare attributes between items.
        Example 1:
        Original: "Which film has the director who was born earlier, The Secret Invasion or The House Of The Seven Hawks?"
        One of the sub-question list: ["Who directed The Secret Invasion and when was this director born?", "Who directed The House Of The Seven Hawks and when was this director born?"]
        Example 2:
        Original: "Do both films The Reincarnation Of Golden Lotus and I'll Get By (Film) have directors from the same country?"
        One of the sub-question list: ["Who directed The Reincarnation Of Golden Lotus and which country is he/she from?", "Who directed I'll Get By (Film) and which country is he/she from?"]

        7. Keep the total length of a sub-questions list minimal (usually 2 at most)
        8. Output should be only a list of subquestions, no other explaination.

        Output format should be a JSON array of sub-questions. For example:
        Original: "Were the wireless earbuds Apple introduced in 2016 revolutionary for the market?"
        Output: 
        ["What wireless earbuds did Apple introduce in 2016?","How did these earbuds impact the wireless earbud market?"]
        

        # Remember: Each sub-question and must be necessary and distinct.Each sub-question list must be distinct. Do not create redundant questions. For comparison questions, focus on gathering the specific information needed for the comparison in the most efficient way.\n"""
        "### User:\n"
        f"Original: {example['question']}\n"
        "### Assistant:\n"
        "Output:\n"
    )

    subtask_content = []
    for i, subq in enumerate(example["subtasks"]):
        subtask_content.append(subq)
        # subtask_content += f"{subq.strip()}\n"
    # subtask_content += f"Final Answer: {example['final_answer']}"
    # subtask_content = f'{{"subquestions":{subtask_content}}}'
    # print(example)
    return {
        "question": example['question'],
        "prompt": prompt,
        "completion": f'{subtask_content}',
        "final_answer": example['final_answer'],
        "expected_answer": example['answer']
    }

def sim_calc(question, sqs):
    question_embedding = sim_model.encode(question, show_progress_bar=False).reshape(1, -1)  
    sum_embs = sim_model.encode(sqs, batch_size=2048, show_progress_bar=False)
    similarity = util.cos_sim(sum_embs, question_embedding).flatten()
    return similarity

def compute_reward(generated_texts, questions):
    rewards = []
    
    for i, generated_text in enumerate(generated_texts):
        # try:
        #     match = re.search(r'\[.*\]', generated_text)
        #     if match:
        #         subquestions_str = match.group(0)
        #         subquestions = json.loads(subquestions_str)
        #         subquestions.append(questions[i])
        #     input_text = f"Question: {questions[i]}\nDecomposition: {subquestions}"
        # except:
        #     print("Parse error!")
        input_text = f"Question: {questions[i]}\nDecomposition: {generated_text}"
        result = reward_pipe(input_text, truncation=True, max_length=512)
        rewards.append(result[0]['score']) 
    
    return torch.tensor(rewards, dtype=torch.float32)

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

class PPOTuningTrainer:
    def __init__(self, model, tokenizer, processed_train_data, original_train_data, args):
        self.model = model
        self.tokenizer = tokenizer
        self.processed_train_data = processed_train_data 
        self.original_train_data = original_train_data
        self.args = args

        self.ppo_config = PPOConfig(
            model_name=args.model_name,
            learning_rate=args.learning_rate,
            batch_size=args.ppo_batch_size,
            mini_batch_size=args.ppo_mini_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optimize_cuda_cache=True,
            early_stopping=args.early_stopping,
            target_kl=args.target_kl,
            ppo_epochs=args.ppo_epochs,
            seed=args.seed,
        )

        
        
        self.ppo_config.remove_unused_columns = False        

        self.ppo_trainer = PPOTrainer(
            config=self.ppo_config,
            model=model,
            ref_model=None, 
            tokenizer=tokenizer,
            dataset=processed_train_data,
        )
    
    def train(self):
        generation_kwargs = {
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "max_new_tokens": 256, 
        }
        
        for epoch in range(self.args.num_train_epochs):
            for batch in tqdm(self.ppo_trainer.dataloader):
                # print(batch["idx"])
                indices = batch["idx"].tolist()
                
                questions = [self.original_train_data[i]["questions"] for i in indices]
                prompts = [self.original_train_data[i]["prompts"] for i in indices]
                final_answers = [self.original_train_data[i]["final_answer"] for i in indices]
                expected_answers = [self.original_train_data[i]["expected_answer"] for i in indices]
                expected_rewards = []
                response_texts = []
                query_tensors = []
                response_tensors = []
                for i, prompt in enumerate(prompts):
                    # query = self.tokenizer.encode(questions[i], return_tensors="pt").cuda()
                    response, inputs, output = custom_llm_llama_rl(self.tokenizer, self.model, prompt)
                    print(f'Question: {questions[i]}')
                    print(f'Response: {response}')
                    expected_reward = f1_score(final_answers[i], [expected_answers[i]])
                    expected_rewards.append(expected_reward)
                    response_texts.append(response)
                    response_tensors.append(output.squeeze())
                    query_tensors.append(inputs["input_ids"].squeeze())
                
              
                rewards = compute_reward(response_texts, questions).cuda()
                print(f'Rewards: {rewards}')
                # print(f'Expected Rewards: {expected_rewards}')
                # print(f'final_answers: {final_answers}')
                # print(f'expected_answers: {expected_answers}')
                
                scores_list = [reward.unsqueeze(0) for reward in rewards]
                # scores_list = [torch.tensor(e) for e in expected_rewards]
                stats = self.ppo_trainer.step(query_tensors, response_tensors, scores_list)
                
               
                self.ppo_trainer.log_stats(stats, batch, rewards)
                # self.ppo_trainer.log_stats(stats, batch, expected_rewards)

            # if epoch % self.args.save_steps == 0:
            #     self.model.save_pretrained(f"{self.args.output_dir}/ppo_epoch_{epoch}")

def preprocess_function(examples):
    inputs = tokenizer(
        examples["prompts"], 
        truncation=True,
        max_length=512,  
        padding="max_length",
        return_tensors="pt"
    )

    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
    }

if __name__ == "__main__":

    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct" 
    DATA_PATH = "data/generate_data_for_rl.json"  
    OUTPUT_DIR = "./decomposition_model_ppo"    

    rm_tokenizer = AutoTokenizer.from_pretrained("./trained_reward_model")
    rm_model = AutoModelForSequenceClassification.from_pretrained("./trained_reward_model")
    reward_pipe = pipeline("text-classification", 
                          model=rm_model, 
                          tokenizer=rm_tokenizer, 
                          device=0 if torch.cuda.is_available() else -1,
                          function_to_apply="none") 
    raw_data = load_and_format_data(DATA_PATH)[1500:]
    # train_raw, val_raw = train_test_split(raw_data, test_size=0.15, random_state=42)

    llama_model_path = "meta-llama/Llama-3.1-8B-Instruct"
    adapter_path = "decomposition_model_300/best_model"
    tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        llama_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).cuda()
    llama_lora = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = llama_lora.merge_and_unload()

    processed_data = []
    for i, ex in enumerate(raw_data):
        processed_data.append(format_training_example(ex))

    ppo_train_data = []
    for example in processed_data:
        ppo_train_data.append({
            "questions": example["question"],
            "prompts": example["prompt"],
            "completion": example["completion"],
            "final_answer": example['final_answer'],
            "expected_answer": example['expected_answer']
        })

    ppo_train_data = Dataset.from_list(ppo_train_data)
    ppo_train_data_processed = ppo_train_data.map(
        preprocess_function,
        batched=True, 
    )
    
    ppo_train_data_processed = ppo_train_data_processed.add_column("idx", list(range(len(ppo_train_data_processed))))
    ppo_train_data_processed.set_format("torch", columns=["input_ids", "attention_mask", "idx"])
    ppo_model = AutoModelForCausalLMWithValueHead(merged_model)
    ppo_model.is_peft_model = False

    ppo_args = type('Args', (), {})()
    ppo_args.model_name = MODEL_NAME
    ppo_args.learning_rate = 1.41e-5
    ppo_args.ppo_batch_size = 2
    ppo_args.ppo_mini_batch_size = 2
    ppo_args.gradient_accumulation_steps = 1
    ppo_args.early_stopping = True
    ppo_args.target_kl = 0.1
    ppo_args.ppo_epochs = 3
    ppo_args.seed = 42
    ppo_args.num_train_epochs = 3
    ppo_args.output_dir = OUTPUT_DIR
    ppo_args.save_steps = 100

    print("Starting PPO training...")
    ppo_trainer = PPOTuningTrainer(ppo_model, tokenizer, ppo_train_data_processed, ppo_train_data, ppo_args)
    ppo_trainer.train()

    ppo_model.save_pretrained(f"{OUTPUT_DIR}/final_ppo")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_ppo")
    
