"""Try this 1: find a person's documents by meaning, by name, or both (make try-hybrid)

Runs the five person queries from test_person_search.py three ways over the 22
sample documents and counts, in each query's top 3, the documents that belong
to the person asked about:

  vector only    cosine similarity of the embeddings, as the pipeline searches
  name match     documents whose text contains the name in the query, ordered by vector score
  hybrid         vector score + WEIGHT x the share of the query's name words found in the text

The bank statement file holds a page for every customer, so it counts as right
for all five. Change WEIGHT or GENERIC_WORDS and run it again.
"""

import re

import numpy as np

import common

WEIGHT = 0.5
# Words in the queries that describe the documents, not the person
GENERIC_WORDS = {"kyc", "documents", "identification", "personal", "information",
                 "financial", "records", "tax"}


def name_words(query):
    words = [w.lower().removesuffix("'s").strip("'") for w in query.split()]
    return [w for w in words if w and w not in GENERIC_WORDS]


def name_share(words, text):
    lower = text.lower()
    return sum(bool(re.search(rf"\b{re.escape(w)}\b", lower)) for w in words) / len(words)


def main():
    docs = common.documents()
    model = common.model()
    doc_vecs = model.encode([d[2][:8000] for d in docs], normalize_embeddings=True)
    query_vecs = model.encode(common.PERSON_QUERIES, normalize_embeddings=True)

    totals = {"vector only": 0, "name match": 0, "hybrid": 0}
    print(f"{'Query':42} {'vector only':>11} {'name match':>10} {'hybrid':>6}   (right person in top 3)")
    for query, person, qv in zip(common.PERSON_QUERIES, common.QUERY_PERSON, query_vecs):
        vector = doc_vecs @ qv
        words = name_words(query)
        share = np.array([name_share(words, d[2]) for d in docs])
        rankings = {
            "vector only": np.argsort(-vector, kind="stable"),
            "name match": np.lexsort((-vector, -(share > 0).astype(int))),
            "hybrid": np.argsort(-(vector + WEIGHT * share), kind="stable"),
        }
        row = []
        for method, order in rankings.items():
            hits = sum(person in docs[i][3] for i in order[:3])
            totals[method] += hits
            row.append(hits)
        print(f"{query:42} {row[0]:>9}/3 {row[1]:>8}/3 {row[2]:>4}/3")
    n = 3 * len(common.PERSON_QUERIES)
    print(f"{'All five queries':42} {totals['vector only']:>8}/{n} {totals['name match']:>7}/{n} "
          f"{totals['hybrid']:>3}/{n}")


if __name__ == "__main__":
    main()
