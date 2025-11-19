import os
import re
import argparse
import logging
import multiprocessing
import transformers
from tqdm import tqdm
from typing import List, Any, Dict
from transformers import AutoTokenizer

tokenizer = None

def clean_text(text: str) -> str:
    """
    Clean and normalize text for TSV format.
    
    Args:
        text (str): Raw text content to be cleaned.
    
    Returns:
        str: Cleaned text with normalized whitespace.
    """
    text = re.sub(r'[\n\r\t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def save_tsv(
    ids: List[Any],
    texts: List[str],
    output_file: str,
) -> None:
    """
    Save (id, text) pairs as TSV file.
    
    Args:
        ids (List[Any]): List of document or query IDs.
        texts (List[str]): List of document or query texts.
        output_file (str): Final output file path. e.g., '/path/to/output/queries_zho_title.tsv'

    Returns:
        None
    """
    with open(output_file, "w", encoding="utf-8") as f:
        for id, text in zip(ids, texts):
            f.write(f"{id}\t{text}\n")


def save_relevance_trec(
    relass: Dict[Any, List[Any]],
    output_file: str
) -> None:
    """
    Save relevance judgments in TREC format (qrel file).
    
    Args:
        relass (Dict[Any, List[Any]]): Mapping from query_id to list of relevant doc_ids or (docid, rel) tuples.
        output_file (str): Final output file path. e.g., '/path/to/output/qrels_zho.tsv'

    Returns:
        None
    
    TREC Format:
        Each line: query_id\t0\tdoc_id\trelevance_score
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for qid, doc_ids in relass.items():
            for item in doc_ids:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    docid, rel = item
                else:
                    docid, rel = item, 1
                f.write(f"{qid}\t0\t{docid}\t{rel}\n")

def init_worker(tokenizer_name):
    """
    Initialize worker process with HuggingFace tokenizer for multiprocessing.
    Sets up global tokenizer variable for efficient reuse across document processing.
    
    Args:
        tokenizer_name (str): Name of the HuggingFace tokenizer to load. If None, uses whitespace tokenization.
    """
    global tokenizer
    transformers.logging.set_verbosity_error()
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    else:
        tokenizer = None

def process_page(args):
    """
    Split a single document into overlapping passages using global tokenizer.
    Uses HuggingFace tokenizer if available, otherwise falls back to whitespace tokenization.
    
    Args:
        args (tuple): Contains (nwords, overlap, docid, text) where:
            - nwords (int): Maximum number of tokens per passage
            - overlap (int): Number of overlapping tokens between passages
            - docid (str): Document identifier
            - text (str): Document text content
    
    Returns:
        list[tuple]: List of (passage_id, passage_text) tuples where passage_id format is "{docid}_{idx}"
    """
    nwords, overlap, docid, text = args
    global tokenizer

    if tokenizer is None:
        words = text.split()
        n_token = len(words)
        if n_token <= overlap:
            passages = [words]
        else:
            passages = [words[offset:offset + nwords] for offset in range(0, n_token, nwords - overlap)]
        passages = [' '.join(psg) for psg in passages]
    else:
        offsets = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        n_token = len(offsets)
        
        if n_token <= overlap:
            passages = [text]
        else:
            step = nwords - overlap
            passages = []
            for start in range(0, n_token, step):
                end = min(start + nwords, n_token)
                passages.append(text[offsets[start][0]:offsets[end-1][1]])
    return [(f"{docid}_{idx}", passage) for idx, passage in enumerate(passages)]


def split_doc_to_passage(
    input_tsv: str,
    passage_file: str,
    mapping_file: str,
    passage_length: int = 180,
    overlap: int = 90,
    tokenizer: str = 'bert-base-multilingual-cased',
    nthreads: int = None
) -> None:
    """
    Split documents into overlapping passages with sequential indices.
    
    Reads a TSV file containing documents and splits each document into overlapping passages. 
    Outputs two files: passages with sequential indices and a mapping file linking indices to original passage IDs.
    
    Args:
        input_tsv (str): Path to input TSV file with format: docid\ttext
        passage_file (str): Path to output passage TSV file with format: index\tpassage
        mapping_file (str): Path to output mapping TSV file with format: index\toriginal_id
        passage_length (int, optional): Maximum tokens per passage. Defaults to 180.
        overlap (int, optional): Overlapping tokens between passages. Defaults to 90.
        tokenizer_name (str, optional): HuggingFace tokenizer name. Defaults to 'bert-base-multilingual-cased'.
        nthreads (int, optional): Number of worker processes. Defaults to os.cpu_count() // 2.
    
    Returns:
        None
    """

    if nthreads is None:
        nthreads = os.cpu_count() // 2
    transformers.logging.set_verbosity_error()

    # Count total documents without loading all into memory
    num_docs = sum(1 for _ in open(input_tsv, encoding="utf-8"))
    
    logging.info(f"Splitting {num_docs} documents into passages with {nthreads} workers...")
    passage_idx = 0
    
    # Process documents in streaming fashion with larger buffers
    with open(input_tsv, encoding="utf-8", buffering=16*1024*1024) as doc_file, \
         open(passage_file, "w", encoding="utf-8", buffering=16*1024*1024) as passage_out, \
         open(mapping_file, "w", encoding="utf-8", buffering=16*1024*1024) as mapping_out:
        
        with multiprocessing.Pool(nthreads, initializer=init_worker, initargs=(tokenizer,)) as pool:
            def doc_generator():
                for line in doc_file:
                    doc_id, doc_text = line.rstrip('\n').split('\t', 1)
                    yield (passage_length, overlap, doc_id.strip(), doc_text.strip())
            
            # Use imap_unordered for better performance, then buffer writes
            for out in tqdm(pool.imap(process_page, doc_generator(), chunksize=50), 
                          total=num_docs, desc="Splitting docs to passages"):
                for original_pid, passage in out:
                    passage_out.write(f"{passage_idx}\t{passage}\n")
                    mapping_out.write(f"{passage_idx}\t{original_pid}\n")
                    passage_idx += 1
    
    logging.info(f"Completed: {num_docs} documents → {passage_idx} passages")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split document TSV into passages and save as TSV.")
    parser.add_argument('--input-tsv', type=str, required=True, help='Input document TSV file (<docid>\t<text>)')
    parser.add_argument('--output-tsv', type=str, required=True, help='Output passage TSV file (<index>\t<passage>)')
    parser.add_argument('--mapping-tsv', type=str, required=True, help='Output mapping TSV file (<index>\t<original_id>)')
    parser.add_argument('--passage-length', type=int, default=180, help='Passage length (in tokens)')
    parser.add_argument('--overlap', type=int, default=90, help='Overlap between passages')
    parser.add_argument('--tokenizer', type=str, default='bert-base-multilingual-uncased', help='HuggingFace tokenizer name')
    parser.add_argument('--nthreads', type=int, default=None, help='Number of parallel workers (default: os.cpu_count())')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    split_doc_to_passage(
        input_tsv=args.input_tsv,
        output_tsv=args.output_tsv,
        mapping_tsv=args.mapping_tsv,
        passage_length=args.passage_length,
        overlap=args.overlap,
        tokenizer_name=args.tokenizer,
        nthreads=args.nthreads
    )