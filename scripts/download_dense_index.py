from __future__ import annotations

import argparse

from app.rag.huggingface import HF_INDEX_NAME, HF_INDEX_REPO, download_published_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the published Kuphaldt FAISS index from Hugging Face.")
    parser.add_argument("--output", default=f"data/indexes/{HF_INDEX_NAME}")
    parser.add_argument("--repo", default=HF_INDEX_REPO)
    parser.add_argument("--index-name", default=HF_INDEX_NAME)
    args = parser.parse_args()
    output = download_published_index(args.output, args.repo, args.index_name)
    print(f"Downloaded published dense index to {output}")


if __name__ == "__main__":
    main()
