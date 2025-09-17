import json


if __name__ == "__main__":
    # Convert limit queries to msmarco tsv format
    with open("./limit/limit/queries.jsonl", "r") as infile, open("./limit_formatted/limit/queries/raw.tsv", "x") as outfile:
        for line in infile:
            data = json.loads(line)
            outfile.write(f"{data['_id']}\t{data['text']}\n")

    # Convert limit corpus to msmarco tsv format
    with open("./limit/limit/corpus.jsonl", "r") as infile, open("./limit_formatted/limit/corpus/raw.tsv", "x") as outfile:
        for line in infile:
            data = json.loads(line)
            outfile.write(f"{data['_id']}\t{data['text']}\n")

    # Convert limit qrels to msmarco tsv format
    with open("./limit/limit/qrels.jsonl", "r") as infile, open("./limit_formatted/limit/queries/qrel.json", "x") as outfile:
        qrels = {}
        for line in infile:
            data = json.loads(line)
            if data["query-id"] not in qrels:
                qrels[data["query-id"]] = {}
            qrels[data["query-id"]][data["corpus-id"]] = data["score"]

        # create json file with query_id and corpus_id:score mapping
        json.dump(qrels, outfile)