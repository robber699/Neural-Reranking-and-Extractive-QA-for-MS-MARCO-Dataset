from allennlp.common import Params, Tqdm
from allennlp.common.util import prepare_environment
from allennlp.data.dataloader import PyTorchDataLoader
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from data_loading import IrTripleDatasetReader, IrLabeledTupleDatasetReader
from model_knrm import KNRM
from model_tk import TK
from core_metrics import calculate_metrics_plain
import torch
import os
from tqdm import tqdm
from allennlp.nn.util import move_to_device
from tools import load_qrels_to_dict
import json


# change paths to your data directory
config = {
    "vocab_directory": "Part-2/allen_vocab_lower_10",
    "pre_trained_embedding": "Part-2/glove.42B.300d.txt",
    "model": "tk",  # "knrm" or "tk"
    "test_data": "Part-2/fira-22.tuples.tsv",
    "qrels_path": "Part-1/fira-22.baseline-qrels.tsv",   #fira qrels relevance for comaprison-> Part-1/fira-22.baseline-qrels.tsv or  Part-1/part1-qrels.txt
    "model_path": "best_model_tk.pth" #best_model_tk.pth or best_model.pth
}

# Load vocabulary
vocab = Vocabulary.from_files(config["vocab_directory"])

# Load embeddings
embeddings_path = "Part-2/tokens_imported.pth"
if os.path.exists(embeddings_path):
    # Load existing embeddings
    tokens_embedder = Embedding(vocab=vocab, embedding_dim=300, trainable=True, padding_index=0)
    tokens_embedder.load_state_dict(torch.load(embeddings_path))
    #print("Embeddings loaded from files.")
else:
    # Create new embeddings
    tokens_embedder = Embedding(vocab=vocab,
                               pretrained_file=config["pre_trained_embedding"],
                               embedding_dim=300,
                               trainable=True,
                               padding_index=0)
    torch.save(tokens_embedder.state_dict(), embeddings_path)
    #print("New embeddings created and saved.")
#print("Embeddings are ready.")
word_embedder = BasicTextFieldEmbedder({"tokens": tokens_embedder})

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#print("okay device")

# Initialize the model
if config["model"] == "knrm":
    model = KNRM(word_embedder, n_kernels=11, device=device)  # Adjust parameters as needed
elif config["model"] == "tk":
    model = TK(word_embedder, n_kernels=2, n_layers = 2, n_tf_dim = 300, n_tf_heads = 30)#n_tf_dim = 300  # Adjust parameters as needed
else:
    raise ValueError("Unknown model type")

# Load the trained model state
#model.load_state_dict(torch.load(config["model_path"]))
model.load_state_dict(torch.load(config["model_path"], map_location=device))
model.to(device)

# Load test data
test_batch_size = 1024
test_reader = IrLabeledTupleDatasetReader(lazy=True, max_doc_length=180, max_query_length=30)
test_dataset = test_reader.read(config["test_data"])
test_dataset.index_with(vocab)
test_loader = PyTorchDataLoader(test_dataset, batch_size=test_batch_size)
#print("Test data loaded.")

# Validate
model.eval()
ranked_results = {}
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing", leave=False):
        batch["query_tokens"]["tokens"]["tokens"] = batch["query_tokens"]["tokens"]["tokens"].to(device)
        batch["doc_tokens"]["tokens"]["tokens"] = batch["doc_tokens"]["tokens"]["tokens"].to(device)

        query_ids = batch["query_id"]
        query = batch["query_tokens"]
        document = batch["doc_tokens"]
        doc_ids = batch["doc_id"]

        for idx, qid in enumerate(query_ids):
            single_query_tokens = {key: {subkey: val[idx].unsqueeze(0) for subkey, val in query['tokens'].items()} for key in query.keys()}
            single_doc_tokens = {key: {subkey: val[idx].unsqueeze(0) for subkey, val in document['tokens'].items()} for key in document.keys()}

            assert isinstance(single_query_tokens['tokens']['tokens'], torch.Tensor), "Query tokens are not tensors"
            assert isinstance(single_doc_tokens['tokens']['tokens'], torch.Tensor), "Doc tokens are not tensors"

            scores = model(single_query_tokens, single_doc_tokens)
            if qid not in ranked_results:
                ranked_results[qid] = []

            doc_id = doc_ids[idx]
            ranked_results[qid].append((doc_id, scores.cpu().item()))

for qid in ranked_results:
    ranked_results[qid] = sorted(ranked_results[qid], key=lambda x: x[1], reverse=True)
    ranked_results[qid] = [doc[0] for doc in ranked_results[qid]]

# Calculate test metrics
qrels = load_qrels_to_dict(config["qrels_path"])
test_metrics = calculate_metrics_plain(ranked_results, qrels)
print("##########################################################################################")
print("Test Metrics FIRA - :")
for metric, value in test_metrics.items():
    print(f"{metric}: {value}")
print("##########################################################################################")
with open("best_ranked_results_fira.json", 'w') as f:
    json.dump(ranked_results, f, indent=4)