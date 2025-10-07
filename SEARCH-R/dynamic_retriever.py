import argparse
import os
import json
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI
import spacy
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import pipeline, AutoTokenizer
import torch
import string
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain 
import faiss
from flair.models import SequenceTagger
from flair.data import Sentence
from collections import Counter
from llm_utils import custom_llm, custom_llm_llama
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Tuple, Set
import numpy as np
from peft import PeftModel, PeftConfig
import ast

nlp_trf = spacy.load("en_core_web_trf")
tagger = SequenceTagger.load("flair/ner-english-large")

# from retrieval import context_length_total
from utils import (
    load_dataset,
    append_result,
    DATASETS,
    logger
)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run ChainRAG algorithm on specified dataset")

    parser.add_argument(
        "--dataset",
        type=str,
        default="musique",
        choices=list(DATASETS.keys()),
        help="Name of dataset to use"
    )

    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Disable cache (default: cache enabled)"
    )

    parser.add_argument(
        "--llm",
        default="gpt-mini-4o",
        help=""
    )

    return parser.parse_args()


def is_single_word(element):
    return len(element.split()) == 1

# Function to get the number of words in an element
def get_word_count(element):
    return len(element.split())

def extract_between_brackets(text):
    pattern = r'\[(.*?)\]'
    raw_arrays = re.findall(pattern, text)
    print(f'raw_arrays: {raw_arrays}')
    nested_arrays = [
        [item.strip() for item in raw_arrays.split(',')] 
        for arr in raw_arrays
    ]
    return nested_arrays

def extract_json(text):
    pattern = r'({.*?})'
    result = re.findall(pattern, text, re.DOTALL)[0]
    return result



def generate_one_line_summaries(chunks, batch_size=32):
   

    device = 0 if torch.cuda.is_available() else -1
    multilingual = False
    for c in chunks:
        if has_non_english_except_punct(c):
            multilingual = True
            break
    
    # if multilingual:
    #     model_name = "csebuetnlp/mT5_multilingual_XLSum" 
    # else:
    model_name = "facebook/bart-large-cnn"  
    # model_name = "uer/t5-base-chinese-summary" 
    print(f'multilingual: {multilingual}')
    summarizer = pipeline(
        "summarization",
        model=model_name,
        device=device,
        truncation=True
    )
    tokenizer = summarizer.tokenizer
    
    summaries = []
    short_texts = [] 

    for i, text in enumerate(chunks):
        if get_word_count(text) < 10:
            summaries.append(text) 
            short_texts.append(i)

    process_texts = [text for i, text in enumerate(chunks) if i not in short_texts]


    for i in range(0, len(process_texts), batch_size):
        batch = process_texts[i:i+batch_size]
        
        results = summarizer(
            batch,
            max_length=40,    
            min_length=5,      
            do_sample=False,    # 不使用采样
            num_beams=4,       
            early_stopping=True 
        )

        batch_summaries = [res['summary_text'] for res in results]
        summaries.extend(batch_summaries)

    final_summaries = []
    process_idx = 0
    for i in range(len(chunks)):
        if i in short_texts:
            final_summaries.append(summaries[short_texts.index(i)])
        else:
            final_summaries.append(summaries[len(short_texts) + process_idx])
            process_idx += 1
    
    return final_summaries

def get_sub_questions(sentences):
    filtered_elements = []
    for i in range(1, len(sentences)):
        if is_single_word(sentences[i-1]) and get_word_count(sentences[i]) > 3:
            filtered_elements.append(sentences[i])
    sorted_elements = sorted(filtered_elements, key=get_word_count)
    results = []
    for j in range(0, len(sorted_elements)-1):
        if sorted_elements[j] not in sorted_elements[j+1]:
            continue
        sub = f'What is {sorted_elements[j]}'
        results.append(sub)
    results.append(item['question'])
    return results

def get_chunks(contexts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=50,
        separators=[r"\n\n", r"\n", r".", r"!", r"?", r"\.\s+", r"\n\s*•", r',', r" "]  # 优先按段落/句子切分
    )
    chunks = splitter.split_text(contexts)
   
    return chunks

# def get_chunks_and_summary(contexts):
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=50,
#         separators=["\n\n", "\n", ".", "!", "?", "\.\s+", "\n\s*•", ',', " "]  # 优先按段落/句子切分
#     )
#     chunks = splitter.split_text(contexts)
#     # docs = splitter.create_documents([context])
        
#     summary = generate_one_line_summaries(chunks)
#     return chunks, summary
# def get_summary(contexts):
#     chunks = contexts  
#     summary = generate_one_line_summaries(chunks)
#     return chunks, summary

def remove_passage_prefix_regex(text):
    # 匹配'Passage'后跟数字和冒号的模式
    pattern = r'^Passage \d+:\s*'
    return re.sub(pattern, '', text)

def extract_entities(tagger, text):
    entities = []
    for c in tqdm(text):
        tagger.predict(c)
        ent = set()
        for entity in c.get_spans('ner'):
            ent.add(remove_passage_prefix_regex(entity.text).strip())
        entities.append(list(ent))
    return entities

def trf_statistics(nlp_trf, summary):
    word_count_results = []
    for s in summary:
        trf = nlp_trf(s)
        all_words = []
        for token in trf:
            subtree = [child.text for child in token.subtree]
            lower = [word.lower() for word in subtree]
            for word in subtree:
                all_words.append(word)
        word_count = dict(Counter(all_words).most_common())
        word_count_results.append(word_count)
    return word_count_results

def get_max_word_frequency(entities_list, word_frequency):

    freq = []
    for wf in word_frequency:
        freq.append({word.lower(): count for word, count in wf.items()})
    result = []

    for i,entities in enumerate(entities_list):
        entity_word_freq = []
        result_unit = {}
        for entity in entities:
            # print(entity)
            words = [word.lower() for word in entity.split()]
            for word in words:
                if word in freq[i]:
                    entity_word_freq.append(freq[i][word])
            max_freq = max(entity_word_freq) if entity_word_freq else 0
            result_unit[entity] = max_freq
        sorted_result = dict(sorted(result_unit.items(), key=lambda item: item[1], reverse=True))
        result.append(sorted_result)
    
    return result

def retrieval(question, entities_list):
    mrr = []
    for entities in entities_list:
        score = 0.0
        for i, entity in enumerate(entities):
            if entity in question:
                score += 1.0/(i+1)
        mrr.append(score)

    indices = np.where(np.array(mrr) > 0.08)[0]
    mrr_top = {}
    for s in indices:
        mrr_top[int(s)] = mrr[s]
    return mrr_top.values(), mrr_top

def sim_calc(model, question, summaries):
    question_embedding = model.encode([question], show_progress_bar=False).reshape(1, -1)
    # print(len(summaries))
    sum_embs = [model.encode(s, batch_size=2048, show_progress_bar=False) for s in summaries]
    similarities = util.cos_sim(sum_embs, question_embedding).flatten().tolist()
    sorted_indices = np.argsort(similarities)[::-1][:10]
    sim_top = {}
    for s in sorted_indices:
        sim_top[int(s)] = similarities[s]
    return similarities, sim_top
def extract_subquestions(input_str):
    match = re.search(r'\[(.*?)\]', input_str, re.DOTALL)
    if not match:
        return []

    inner_content = match.group(1).strip()
    
    if inner_content.startswith('{') and inner_content.endswith('}'):
        inner_content = inner_content[1:-1].strip()
    
    subquestions = []
    for item in inner_content.split(','):
        cleaned = item.strip().strip('"').strip()
        if cleaned:
            if cleaned.endswith('"'):
                cleaned = cleaned[:-1]
            subquestions.append(cleaned)
    
    return subquestions
def force_answer(sub_question: str, context_sentences: List[str]) -> str:
    # print("Begin answer...")
    prompt = """Based on the given context, you must provide an answer with fewest words to the question.
        Only give me the answer and do not output any other words.
        
        Context: {context}
        Question: {question}
        
        Provide your best possible answer:"""
        # if len(sub_question) < 2:
        #     return ""
    prompt = prompt.format(
                context=" ".join(context_sentences),
                question=sub_question
            )
    # print(prompt)
    try:
        # response = custom_llm_llama(tokenizer, llama_model, prompt)
        response = custom_llm(prompt)
        return response.strip()
            
    except Exception as e:
        print(f"Error forcing answer: {str(e)}")
        return "Unable to provide an answer due to error"
            
def retrieval_and_answer(query, entities_freq, chunks):
    answers = []
    sub_questions = decompose_question(query)
    sub_questions_modified = []
    print(f'sub_questions: {sub_questions}')
    for i,sq in enumerate(sub_questions):
        modified_sub_q = sq
        if i > 0:
            prev_ans = answers[:i]
            should_replace = custom_llm(f"""
                            Previous question: {sub_questions[:i]}
                            Previous answers: {prev_ans}
                            Current question: {sq}
                            Does the current question refer to the answer of the previous questions? Answer yes or no.
                            """).strip().lower()
            print(f'should_replace: {should_replace}')
            if "yes" in should_replace:
                modified_sub_q = custom_llm(f"""
                            Rewrite the following question to be self-contained by replacing pronouns or references with the actual entities they refer to.
                            Previous question: {sub_questions[:i]}
                            Previous answers: {prev_ans}
                            Current question: {sq}
                            Rewritten question:
                            """).strip()
        
        print(f'Current ques: {modified_sub_q}')
        
        sub_questions_modified.append(modified_sub_q)
        scores, mrr_top = retrieval(modified_sub_q, entities_freq)
            
        sim, sim_top = sim_calc(model, modified_sub_q, chunks)
            # print(f'sim: {sim}')
        mrr_index = list(mrr_top.keys())
        sim_index = list(sim_top.keys())[:8]
            # sim_index = []
        index = list(set(mrr_index).union(sim_index))
    
        context = [chunks[idx] for idx in index]
    
        current_answer = force_answer(modified_sub_q, context)
        print(f'Answer: {current_answer}')
        answers.append(current_answer)
        
    return answers, sub_questions, sub_questions_modified


def decompose_question(question: str) -> List[str]:

    system_prompt = """You are a helpful AI assistant that helps break down questions into minimal necessary sub-questions. 
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
        ["What wireless earbuds did Apple introduce in 2016?", "How did these earbuds impact the wireless earbud market?"]
        

        # Remember: Each sub-question and must be necessary and distinct.Each sub-question list must be distinct. Do not create redundant questions. For comparison questions, focus on gathering the specific information needed for the comparison in the most efficient way."""

        
    full_prompt = f"{system_prompt}\n\nQuestion: {question}\n\nBreak down this question into minimal necessary sub-questions:"
        

    response = custom_llm_llama(tokenizer, merged_model, full_prompt)
    # response = custom_llm(full_prompt)
    print(f'response: {response}')
    sub_questions = extract_subquestions(response)
    # print(f'----------sub_questions {len(sub_questions)}:{sub_questions}')
    if len(sub_questions) != 0:
        # print('1----------')
        sub_questions.append(question)
        return sub_questions
    else:
        # print('2----------')
        sub_questions_split = response.split(',')
        sub_questions_split.append(question)
        return sub_questions_split

            
        


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def has_non_english_except_punct(text):
    punctuations = set(string.punctuation)
    text_without_punct = ''.join([ch for ch in text if ch not in punctuations])
    non_english_pattern = re.compile(r'[^a-zA-Z]')
    
  
    return bool(non_english_pattern.search(text_without_punct))
    
if __name__ == "__main__":
    args = parse_args()
    # output_dir = Path("output") /"llama-3.1" /args.dataset
    output_dir = Path("output")/"ppo"/args.dataset
    # output_dir = Path("question_kmeans")
    output_dir.mkdir(parents=True, exist_ok=True)
    # output_file = output_dir/"musique_train"/"subsentences_kmeans_2_k4_gpt.json"
    llama_model_path = "meta-llama/Llama-3.1-8B-Instruct"
    # adapter_path = "decomposition_model_300/best_model"
    tokenizer = AutoTokenizer.from_pretrained(llama_model_path)
    # base_model = AutoModelForCausalLM.from_pretrained(
    #     llama_model_path,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto"
    # ).cuda()
    llama_model = AutoModelForCausalLM.from_pretrained(
        llama_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).cuda()
    # llama_lora = PeftModel.from_pretrained(base_model, adapter_path)
    # merged_model = llama_lora.merge_and_unload()
    ppo_model_path = "decomposition_model_ppo/final_ppo"
    merged_model = AutoModelForCausalLM.from_pretrained(
        ppo_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).cuda()
    
    output_file= output_dir/"retrieval_doc_tmp_0.08.json"
    
    
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').cuda()
    questions = []
    output_results = []

    # data = load_data("data/musique/data/musique_train.json")
    data = load_dataset(args.dataset)
    for idx, item in enumerate(tqdm(data, desc="Processing questions", unit="question")):
        query = item['question']
        expected_answer = item["expected_answer"]
        questions.append(query)
        trf = nlp_trf(query)
        sentences = []
        contexts = item['context']
        all_words = []
        for token in trf:
            subtree = [child.text for child in token.subtree]
            sentence = " ".join(subtree)
            sentences.append(sentence)
       
        # contexts = []
        # for p in item['paragraphs']:
        #     contexts.append(f"{p['title']}:{p['paragraph_text']}")
        
        # chunks, summary = get_chunks_and_summary(contexts)
       
        chunks = get_chunks(contexts)

        word_count = trf_statistics(nlp_trf, chunks)

        chunks_sentence = [Sentence(x) for x in chunks]
        entities = extract_entities(tagger, chunks_sentence)


        entities_freq = get_max_word_frequency(entities, word_count)

 
        answer_list = []
        context_list = []
        answers, sub_questions, sub_questions_modified = retrieval_and_answer(query, entities_freq, chunks)
        answer_list.append(answers)
        print(f'------answers: {answers}')
       
        output = {
            "question": query,
            "expected_answer": expected_answer,
            "sub_questions": sub_questions,
            "sub_questions_modified": sub_questions_modified,
            "answers": answer_list,
        }
        output_results.append(output)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_results, f, ensure_ascii=False,indent=4)