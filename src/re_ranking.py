from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.dataloader import PyTorchDataLoader

prepare_environment(Params({})) # sets the seeds to be fixed

import torch
import torch.optim as optim
import torch.nn as nn
#####
from torch.optim import Adam
from torch.nn import MarginRankingLoss

from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder

from data_loading import IrTripleDatasetReader, IrLabeledTupleDatasetReader
from model_knrm import KNRM
from model_tk import TK

from core_metrics import calculate_metrics_plain

import os
#from tqdm import tqdm
import sys
from tqdm import tqdm

#####
from itertools import islice
####

from allennlp.nn.util import move_to_device
from tools import load_qrels_to_dict


# change paths to your data directory
config = {
    "vocab_directory": "Part-2/allen_vocab_lower_10",
    "pre_trained_embedding": "Part-2/glove.42B.300d.txt",
    "model": "tk",#knrm #tk
    "train_data": "Part-2/triples.train.tsv",
    "validation_data": "Part-2/msmarco_tuples.validation.tsv",
    "test_data":"Part-2/msmarco_tuples.test.tsv",
    "qrels_path":"Part-2/msmarco_qrels.txt"
}

#
# data loading
#
vocab = Vocabulary.from_files(config["vocab_directory"])
# Check if embeddings have been previously created and saved
print("okay")
embeddings_path = "Part-2/tokens_imported.pth"
if os.path.exists(embeddings_path):
    # Load existing embeddings
    tokens_embedder = Embedding(vocab=vocab, embedding_dim=300, trainable=True, padding_index=0)
    tokens_embedder.load_state_dict(torch.load(embeddings_path))
    print("Embeddings loaded from files.")
else:
    # Create new embeddings
    tokens_embedder = Embedding(vocab=vocab,
                               pretrained_file=config["pre_trained_embedding"],
                               embedding_dim=300,
                               trainable=True,
                               padding_index=0)
    torch.save(tokens_embedder.state_dict(), embeddings_path)
    print("New embeddings created and saved.")
print("Embeddings are ready.")
word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})

print("Tokens and word embedders are ready.")

##########################
# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Move model to device

# recommended default params for the models (but you may change them if you want)
if config["model"] == "knrm":
    model = KNRM(word_embedder, n_kernels=11, device=device)#11
elif config["model"] == "tk":
    model = TK(word_embedder, n_kernels=11, n_layers = 2, n_tf_dim = 300, n_tf_heads = 10)
else:
    raise ValueError("Unknown model type")

# Move model to device
model.to(device)
###
# todo optimizer, loss
###
# optimizer = optim.Adam(model.parameters(), lr=0.009)
optimizer = Adam(model.parameters(), lr=0.001)
# optimizer = Adam(model.parameters(), lr=0.015)
loss_function = MarginRankingLoss(margin=1.0, reduction='mean') # .to(device)
###



print('Model',config["model"],'total parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
print('Network:', model)

#
# train
#
training_batch_size = 128
_triple_reader = IrTripleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
_triple_reader = _triple_reader.read(config["train_data"])
_triple_reader.index_with(vocab)
train_loader = PyTorchDataLoader(_triple_reader, batch_size=training_batch_size, pin_memory=True)
validation_batch_size = 128
val_reader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
val_dataset = val_reader.read(config["validation_data"])
val_dataset.index_with(vocab)
val_loader = PyTorchDataLoader(val_dataset, batch_size=training_batch_size)

print("Before training all good :)")

# Training function
# Early stopping parameters
best_val_metric = float('-inf')
patience = 3
patience_counter = 0

start_epoch = 0
# Load checkpoint
checkpoint_path = "model-0-.pt"

if os.path.exists(checkpoint_path):
    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_function.load_state_dict(checkpoint['loss'])
    start_epoch = checkpoint['epoch']

# Main training and validation loop
num_epochs = 5
for epoch in range(start_epoch, num_epochs, 1):

    # Train
    model.train()
    label = torch.ones(training_batch_size, device=device) # .to(device)
    total_loss = 0.0
    batch_count = 0  # Initialize batch counter


    train_progress_bar = enumerate(Tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False))
    for i, batch in train_progress_bar:

        batch["query_tokens"]["tokens"]["tokens"]=batch["query_tokens"]["tokens"]["tokens"].to(device)
        batch["doc_pos_tokens"]["tokens"]["tokens"]=batch["doc_pos_tokens"]["tokens"]["tokens"].to(device)
        batch["doc_neg_tokens"]["tokens"]["tokens"]=batch["doc_neg_tokens"]["tokens"]["tokens"].to(device)

        optimizer.zero_grad()

        pos_scores = model.forward(query=batch["query_tokens"], document=batch["doc_pos_tokens"])
        neg_scores = model.forward(query=batch["query_tokens"], document=batch["doc_neg_tokens"])
        
        # Ensure consistent batch sizes
        if pos_scores.size(0) != neg_scores.size(0):
            print(f"Skipping batch due to size mismatch: pos_scores.size(0)={pos_scores.size(0)}, neg_scores.size(0)={neg_scores.size(0)}")
            continue
    
        # Ensure the label size matches the pos_scores and neg_scores
        if label.size(0) != pos_scores.size(0):
            print(f"Skipping batch due to size mismatch: label.size(0)={label.size(0)}, pos_scores.size(0)={pos_scores.size(0)}")
            continue

        #target = torch.ones_like(pos_scores)
        loss = loss_function(pos_scores, neg_scores, label)#target?

        #optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        batch_count += 1  # Increment batch counter

    avg_loss = total_loss / batch_count
    print("##########################################################################################")
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    print("##########################################################################################")

    # save check point
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_function.state_dict(),
    }, f"model-{epoch + 1}-.pt")

    # Validate
    model.eval()
    ranked_results = {}
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating", leave=False):
            # Print structure/shape and data types before moving to device
            #print("Keys in batch:")
            #print(batch.keys())

            # Move batch to device
            batch["query_tokens"]["tokens"]["tokens"]=batch["query_tokens"]["tokens"]["tokens"].to(device)
            batch["doc_tokens"]["tokens"]["tokens"]=batch["doc_tokens"]["tokens"]["tokens"].to(device)

            # Print data types after moving to device
            query_ids = batch["query_id"]
            query = batch["query_tokens"]
            document = batch["doc_tokens"]
            doc_ids = batch["doc_id"]  # Assuming doc_id is directly accessible from batch#############

            for idx, qid in enumerate(query_ids):
                single_query_tokens = {key: {subkey: val[idx].unsqueeze(0) for subkey, val in query['tokens'].items()} for key in query.keys()}
                single_doc_tokens = {key: {subkey: val[idx].unsqueeze(0) for subkey, val in document['tokens'].items()} for key in document.keys()}
                
                # Ensure query_tokens and doc_tokens are tensors
                assert isinstance(single_query_tokens['tokens']['tokens'], torch.Tensor), f"Query tokens are not tensors"
                assert isinstance(single_doc_tokens['tokens']['tokens'], torch.Tensor), f"Doc tokens are not tensors"

                scores = model(single_query_tokens, single_doc_tokens)
                #print(f"Scores shape: {scores.shape}")
                if qid not in ranked_results:
                    ranked_results[qid] = []
                
                # Extract docID directly from batch["doc_id"]
                doc_id = doc_ids[idx]

                #ranked_results[qid].append(scores.squeeze().cpu().tolist())
                ranked_results[qid].append((doc_id, scores.cpu().item())) #.sum()    before imem
                
    print("okayyyy validationnnn")
    # Sort the results for each query ID based on the scores in descending order
    for qid in ranked_results:
        ranked_results[qid] = sorted(ranked_results[qid], key=lambda x: x[1], reverse=True)
        ranked_results[qid] = [doc[0] for doc in ranked_results[qid]]  # Keep only the document IDs, discard the scores
    '''
    print("okayyyy validationnnn")
    print("######################ranked results######################")
    print("ranked_results", ranked_results)
    print("###################### END    ranked results######################")
    '''
    # Calculate validation metrics
    qrels = load_qrels_to_dict(config["qrels_path"])
    '''
    print("###################### qrels ######################")
    print(qrels)
    print("###################### END    qrels######################")
    '''
    val_metrics = calculate_metrics_plain(ranked_results, qrels)
    
    print("##########################################################################################")
    print("Validation Metrics:")
    for metric, value in val_metrics.items():
        print(f"{metric}: {value}")
    print("##########################################################################################")
    # Determine early stopping
    current_val_metric = val_metrics['MRR@10']  # Adjust this metric based on your needs
    if current_val_metric > best_val_metric:
        best_val_metric = current_val_metric
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
        print("New best model saved.")
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered.")
        break

# Optionally save the final trained model
#torch.save(model.state_dict(), "trained_model.pth")

