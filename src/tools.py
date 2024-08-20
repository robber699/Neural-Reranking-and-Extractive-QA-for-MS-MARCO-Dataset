def load_qrels_to_dict(qrels_path):
    qrels = {}
    with open(qrels_path, 'r') as file:
        for line in file:
            qid, _, docid, rel = line.strip().split()
            if qid not in qrels:
                qrels[qid] = {}
            qrels[qid][docid] = int(rel)  # Add the document ID and its relevance score
    return qrels