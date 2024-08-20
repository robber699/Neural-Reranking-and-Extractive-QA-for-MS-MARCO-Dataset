import sys
import pandas as pd
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import json
import re 
from core_metrics import compute_exact, compute_f1

model_name = "deepset/roberta-base-squad2"
pipeline("question-answering", model=model_name, tokenizer=model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load data files
file_path_answer = "Part-3/msmarco-fira-21.qrels.qa-answers.tsv"
file_path_tuples = "Part-3/msmarco-fira-21.qrels.qa-tuples.tsv"

# Function to load and parse the data
def load_data(file_path, is_tuples=False):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            fields = re.split(r'\t+', line.strip())
            if is_tuples:
                if len(fields) >= 5:
                    queryid = fields[0]
                    documentid = fields[1]
                    relevance_grade = fields[2]
                    query_text = fields[3]
                    document_text = fields[4]
                    text_selection = '\t'.join(fields[5:])
                    data.append({
                        "queryid": queryid,
                        "documentid": documentid,
                        "relevance_grade": relevance_grade,
                        "query_text": query_text,
                        "document_text": document_text,
                        "text_selection": text_selection
                    })
            else:
                if len(fields) >= 4:
                    queryid = fields[0]
                    documentid = fields[1]
                    relevance_grade = fields[2]
                    text_selection = ' '.join(fields[3:])
                    text_selection_values = text_selection.split('\t') if text_selection else []
                    first_text_selection = text_selection_values[0] if text_selection_values else ""
                    data.append({
                        "queryid": queryid,
                        "documentid": documentid,
                        "relevance_grade": relevance_grade,
                        "text_selection": first_text_selection
                    })
    return pd.DataFrame(data)

# Load answer and tuple data
answer_data = load_data(file_path_answer)
triple_data = load_data(file_path_tuples, is_tuples=True)

# Ensure relevance_grade is numeric and handle missing values
triple_data['relevance_grade'] = pd.to_numeric(triple_data['relevance_grade'], errors='coerce')
triple_data.dropna(subset=['relevance_grade'], inplace=True)

# Group by queryid and keep the most relevant document for each query
triple_data = triple_data.loc[triple_data.groupby('queryid')['relevance_grade'].idxmax()]
print("group_by done")


def tokenize_qa_pairs(query, passage):
    # Check if either query or passage is NaN
    if pd.isnull(query) or pd.isnull(passage):
        return None
    
    inputs = tokenizer.encode_plus(query, passage, add_special_tokens=True, return_tensors="pt")
    return inputs

def extract_answers(data, model):
    results = []

    for entry in data.itertuples(index=False):
        query_id, document_id, relevance_grade, query_text, document_text = entry[:5]

        # Skip processing if document_text is NaN
        if pd.isnull(document_text):
            continue

        inputs = tokenize_qa_pairs(query_text, document_text)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Unpack the outputs tuple
        start_logits = outputs[0]
        end_logits = outputs[1]

        answer_start = torch.argmax(start_logits, dim=1).item()

        beam_scores = outputs[1]
        beam_size = 5  # Number of candidates to consider
        topk_scores, topk_indices = torch.topk(beam_scores, k=beam_size)

        
        # Choose answer_end_idx based on the criteria
        for idx in range(beam_size):
            candidate_idx = topk_indices[0, idx].item()
            if candidate_idx > answer_start:
                answer_end = candidate_idx
                break
        else:
            # If no candidate greater than answer_start_idx, default to the highest score
            answer_end = topk_indices[0, 0].item()

        # Handle cases where answer_start or answer_end exceed the length of input_ids
        if answer_start >= len(input_ids[0]):
            continue
        if answer_end >= len(input_ids[0]):
            continue

        answer = tokenizer.decode(input_ids[0, answer_start:answer_end+1])

        result = {
            "queryid": query_id,
            "documentid": document_id,
            "relevance-grade": relevance_grade,
            "answer": answer,
            "score": float(start_logits[0, answer_start]) + float(end_logits[0, answer_end])
        }
        
        results.append(result)

    return pd.DataFrame(results)


def evaluate(predictions, ground_truth):
    exact_scores = []
    f1_scores = []

    for _, pred in predictions.iterrows():
        query_id = pred["queryid"]
        document_id = pred["documentid"]
        pred_answer = pred["answer"]

        #corresponding ground truth answer
        gt_entry = ground_truth[(ground_truth["queryid"] == query_id) & (ground_truth["documentid"] == document_id)]
        if gt_entry.empty:
            continue

        gt_answer = gt_entry.iloc[0]["text_selection"]

        #exact match and F1 scores
        exact_scores.append(compute_exact(gt_answer, pred_answer))
        f1_scores.append(compute_f1(gt_answer, pred_answer))

    exact_match = sum(exact_scores) / len(exact_scores) if exact_scores else 0
    f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0

    return exact_match, f1_score


if __name__ == '__main__':
    #print("Loading ranked result data...")
    #data = load_ranked_data(input_file)
    #print("Ranked result data loaded successfully.", data)
    #data=gold_tuples
    #print(data)

    #print("Extracting answers...")
    results = extract_answers(triple_data, model)
    print("answer done")

    metrics=evaluate(results, answer_data)
    print("Metrics(exact_match, f1_score)")
    print(metrics)

    
    # Print the first three instances
    for idx, row in results.head(3).iterrows():
        query_id = row['queryid']
        document_id = row['documentid']
        query_text = triple_data[triple_data['queryid'] == query_id].iloc[0]['query_text']
        document_text = triple_data[triple_data['queryid'] == query_id].iloc[0]['document_text']
        text_selection = answer_data[(answer_data['queryid'] == query_id) & (answer_data['documentid'] == document_id)]['text_selection'].values[0]
        answer = row['answer']
        
        print(f"Query ID: {query_id}")
        print(f"Document ID: {document_id}")
        print(f"Query: {query_text}")
        print(f"Text Selection: {text_selection}")
        print(f"Document Text: {document_text}")
        print(f"Computed Answer: {answer}")
        print("\n" + "="*50 + "\n")
