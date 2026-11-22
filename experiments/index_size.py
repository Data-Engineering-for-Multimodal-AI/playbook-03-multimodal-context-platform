"""Try this 3: when does an exact search get too slow? (make try-index-size)

The local index (LocalIndex in src/kyc.py) scores the query against every
vector, sorts all the scores and builds a Python list of every hit before it
keeps the top results. That is exact and fine for 22 documents. This fills a
LocalIndex with random unit vectors of the local model's size (384), times 50
top-10 queries at each size, then the same search with argpartition, which
finds the top 10 without sorting everything.

Pass other sizes with make try-index-size SIZES="1000 1000000". A million
vectors need about 1.5 GB of memory.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kyc import LocalIndex  # noqa: E402

DIM = 384
SIZES = [int(s) for s in sys.argv[1:]] or [22, 10_000, 100_000, 300_000]
QUERIES = 50
K = 10


def unit(rows, rng):
    m = rng.standard_normal((rows, DIM), dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def local_index(matrix):
    index = LocalIndex.__new__(LocalIndex)  # skip loading a file
    index.ids = [str(i) for i in range(len(matrix))]
    index.matrix = matrix
    index.categories = [""] * len(matrix)
    return index


def top_k(matrix, q):
    scores = matrix @ q
    part = np.argpartition(-scores, K - 1)[:K] if len(scores) > K else np.arange(len(scores))
    return part[np.argsort(-scores[part], kind="stable")]


def per_query_ms(fn, queries):
    start = time.perf_counter()
    for q in queries:
        fn(q)
    return 1000 * (time.perf_counter() - start) / len(queries)


def main():
    rng = np.random.default_rng(0)
    queries = unit(QUERIES, rng)
    print(f"{'vectors':>9} {'memory':>8} {'LocalIndex (sort all)':>22} {'top 10 only':>11}")
    for n in SIZES:
        matrix = unit(n, rng)
        index = local_index(matrix)
        for q in queries[:5]:
            exact = [int(i) for i, _ in index.find_neighbors(q, K)]
            assert exact == [int(i) for i in top_k(matrix, q)], "top_k disagrees with LocalIndex"
        a = per_query_ms(lambda q: index.find_neighbors(q, K), queries)
        b = per_query_ms(lambda q: top_k(matrix, q), queries)
        print(f"{n:9,} {matrix.nbytes / 1e6:6.1f} MB {a:19.2f} ms {b:8.2f} ms")
        del matrix


if __name__ == "__main__":
    main()
