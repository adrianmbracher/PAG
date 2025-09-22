import gzip
import json
import pickle

QUERIES_PATH = "./limit/limit-custom/queries.jsonl"
TRAINING_QUERIES_PATH = "./limit/limit-custom/queries-train.jsonl"
CORPUS_PATH = "./limit/limit-custom/corpus.jsonl"
QRELS_PATH = "./limit/limit-custom/qrels.jsonl"

training_queries = 800

if __name__ == "__main__":
    # Convert limit queries to msmarco tsv format
    with open(QUERIES_PATH, "r") as infile, open("./limit_formatted/limit/queries/raw.tsv", "w") as outfile:
        for line in infile:
            data = json.loads(line)
            outfile.write(f"{data['_id']}\t{data['text']}\n")

    with open(TRAINING_QUERIES_PATH, "r") as infile, open("./limit_formatted/limit/train_queries/raw.tsv", "w") as outfile:
        for line in infile:
            data = json.loads(line)
            outfile.write(f"{data['_id']}\t{data['text']}\n")

    # Convert limit corpus to msmarco tsv format
    with open(CORPUS_PATH, "r") as infile, open("./limit_formatted/limit/corpus/raw.tsv", "w") as outfile:
        for line in infile:
            data = json.loads(line)
            outfile.write(f"{data['_id']}\t{data['text']}\n")

    # Convert limit qrels to msmarco tsv format
    with open(QRELS_PATH, "r") as infile, open("./limit_formatted/limit/queries/qrel.json", "w") as outfile:
        qrels = {}
        for line in infile:
            data = json.loads(line)
            if data["query-id"] not in qrels:
                qrels[data["query-id"]] = {}
            qrels[data["query-id"]][data["corpus-id"]] = data["score"]

        # create json file with query_id and corpus_id:score mapping
        json.dump(qrels, outfile)



    # create teacher scores
    from sentence_transformers import CrossEncoder

    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
    with open(TRAINING_QUERIES_PATH, "r") as queriesfile, open(CORPUS_PATH, "r") as corpusfile:
        qid_to_rerank = {}
        docids = []
        texts = []
        for cline in corpusfile:
            centry = json.loads(cline)
            docids.append(centry["_id"])
            texts.append(centry["text"])
        for qline in queriesfile:
            qentry = json.loads(qline)
            print(qentry)
            qid = qentry["_id"]
            qid_to_rerank[qentry["_id"]] = {}
            prediction_inputs = [(qentry["text"], text) for text in texts]
            scores = model.predict(prediction_inputs, show_progress_bar=True)
            qid_to_rerank[qid] = {docid: float(score) for docid, score in zip(docids, scores)}
        with gzip.open("./limit_formatted/limit/hard_negatives_scores/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz", "wb") as outfile:
            pickle.dump(qid_to_rerank, outfile)