import gzip
import json
import pickle

CORPUS_PATH = "./limit/limit/corpus.jsonl"

training_queries = 800

if __name__ == "__main__":
    # Convert limit queries to msmarco tsv format
    with open(CORPUS_PATH, "r") as infile, open("./limit_formatted/limit_seal/corpus/raw.tsv", "w") as outfile:
        for line in infile:
            data = json.loads(line)
            outfile.write(f"{data['_id']}\t{data['_id']}\t{data['text']}\n")