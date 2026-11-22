"""Try this 2: how much of each document does the embedding read? (make try-chunks)

all-MiniLM-L6-v2 reads at most 256 tokens of a text and ignores the rest, and
create_embeddings.py sends each document as one text. This counts the tokens in
each document, then embeds the documents two ways and runs the pipeline's
queries on both:

  whole document   one vector per document, as the pipeline does (first 256 tokens)
  chunks           one vector per piece of CHUNK_SIZES tokens; a document scores as its best chunk

It reports, for the five person queries, the documents in the top 3 that belong
to the person, and for the four document-type queries, the top 3 results of the
right type. Change CHUNK_SIZES and run it again.
"""

import numpy as np

import common

CHUNK_SIZES = [256, 128, 64]


def chunks(tokenizer, text, size):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return [tokenizer.decode(ids[i:i + size]) for i in range(0, max(len(ids), 1), size)], len(ids)


def score(doc_vecs, query_vecs, docs):
    """doc_vecs: list (per document) of arrays of chunk vectors."""
    person = types = 0
    for qv, who in zip(query_vecs["person"], common.QUERY_PERSON):
        best = np.array([float((v @ qv).max()) for v in doc_vecs])
        person += sum(who in docs[i][3] for i in np.argsort(-best, kind="stable")[:3])
    for qv, (_, category) in zip(query_vecs["type"], common.TYPE_QUERIES):
        best = np.array([float((v @ qv).max()) for v in doc_vecs])
        types += sum(docs[i][1] == category for i in np.argsort(-best, kind="stable")[:3])
    return person, types


def main():
    docs = common.documents()
    model = common.model()
    tok = model.tokenizer
    limit = model.max_seq_length

    lengths = np.array([chunks(tok, d[2][:8000], limit)[1] for d in docs])
    over = lengths > limit
    print(f"{len(docs)} documents, {lengths.min()} to {lengths.max()} tokens each; the model reads {limit}.")
    print(f"{over.sum()} documents are longer; it ignores {1 - np.minimum(lengths, limit).sum() / lengths.sum():.0%} "
          f"of all tokens.\n")

    query_vecs = {"person": model.encode(common.PERSON_QUERIES, normalize_embeddings=True),
                  "type": model.encode([q for q, _ in common.TYPE_QUERIES], normalize_embeddings=True)}
    whole = [model.encode([d[2][:8000]], normalize_embeddings=True) for d in docs]
    runs = [("whole document", whole)]
    for size in CHUNK_SIZES:
        pieces = [chunks(tok, d[2][:8000], size)[0] for d in docs]
        runs.append((f"chunks of {size} tokens", [model.encode(p, normalize_embeddings=True) for p in pieces]))

    print(f"{'Embedding':28} {'vectors':>7} {'person queries':>14} {'type queries':>12}")
    for label, vecs in runs:
        person, types = score(vecs, query_vecs, docs)
        n = sum(len(v) for v in vecs)
        print(f"{label:28} {n:7} {person:>11}/15 {types:>9}/12")


if __name__ == "__main__":
    main()
