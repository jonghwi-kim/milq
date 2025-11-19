import os
import argparse
import json
import logging
import re
from multiprocessing import Pool, cpu_count
from typing import Optional, Dict, Set, List, Union
from tqdm import tqdm
from .utils import save_tsv, split_doc_to_passage

def load_qrels(
    qrels_path: str, 
    lang: str
) -> Set[str]:
    """
    Load qrels file and extract unique query IDs with relevance > 0.
    
    Args:
        qrels_path (str): Path to qrels file in TREC format.
        lang (str): Language code (ISO 639-3) for logging purposes. e.g., 'zho', 'fas', 'rus'
    
    Returns:
        set: Set of unique query IDs that have at least one relevant document.
    """
    logging.info(f"[QRELS] Loading qrels for {lang}: {qrels_path}")
    
    query_ids = set()
    with open(qrels_path, 'r', encoding="utf-8") as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            if relevance != '0':
                query_ids.add(query_id)

    logging.info(f"[QRELS] Loaded {len(query_ids)} query IDs for {lang}")
    return query_ids

def preprocess_topics_to_queries(
    topics_file: str,
    output_dir: str,
    qrels_by_lang: Optional[Dict[str, Set[str]]] = None
) -> None:
    """
    Convert topics JSONL to language-specific query TSV files with qrels filtering.
    
    Args:
        topics_file (str): Path to input topics JSONL file.
        output_dir (str): Output directory path. e.g., '/path/to/output'
        qrels_by_lang (dict, optional): Dictionary mapping language codes to sets of valid query IDs. 
                                       Only topics with IDs in these sets will be included.

    Returns:
        None
    """

    logging.info(f"[QUERY] Processing topics file: {topics_file}")

    # Load topics and parse data
    topics_by_lang = {}
    with open(topics_file, "r", encoding="utf-8") as input_file:
        for line in input_file:
            data = json.loads(line.strip())
            topic_id = data.get("topic_id", "")
            
            for topic in data.get("topics", []):
                lang = topic.get("lang", "")
                if lang not in topics_by_lang:
                    topics_by_lang[lang] = {}
                
                title = topic.get("topic_title", "").strip()
                description = topic.get("topic_description", "").strip()
                
                topics_by_lang[lang][topic_id] = {
                    'title': title,
                    'desc': description,
                    'concat': f"{title} {description}"
                }

    # Process each target language (zho, fas, rus)
    for target_lang in ['zho', 'fas', 'rus']:
        try:
            logging.info(f"[QUERY] Processing {target_lang} topics...")
            lang_dir = os.path.join(output_dir, target_lang)
            os.makedirs(lang_dir, exist_ok=True)

            for query_type in ['title', 'desc', 'concat']:
                native_file = os.path.join(lang_dir, f"{target_lang}_queries_{query_type}.tsv")
                eng_file = os.path.join(lang_dir, f"eng_queries_{query_type}.tsv")
                
                # 1) Check if both query files already exist
                if os.path.exists(native_file) and os.path.exists(eng_file):
                    logging.info(f"[QUERY] Query files already exist for {target_lang} {query_type}, skipping.")
                    continue
                
                # 2) Filter topics by qrels
                valid_query_ids = qrels_by_lang[target_lang]

                # 3) Process native language queries
                if not os.path.exists(native_file):
                    topic_ids = []
                    topic_texts = []
                    for topic_id, topic_data in topics_by_lang[target_lang].items():
                        if topic_id in valid_query_ids:
                            topic_ids.append(topic_id)
                            topic_texts.append(topic_data[query_type])
                    
                    # 4) Save native language queries as TSV files
                    save_tsv(topic_ids, topic_texts, native_file)
                    
                # 5) Process English queries
                if not os.path.exists(eng_file):
                    eng_topic_ids = []
                    eng_topic_texts = []
                    for topic_id, topic_data in topics_by_lang['eng'].items():
                        if topic_id in valid_query_ids:
                            eng_topic_ids.append(topic_id)
                            eng_topic_texts.append(topic_data[query_type])
                    
                    # 6) Save English queries as TSV files
                    save_tsv(eng_topic_ids, eng_topic_texts, eng_file)

        except Exception as e:
            logging.error(f"[QUERY] Failed processing {target_lang}: {e}")
            continue

def process_jsonl_line(line: str) -> Optional[str]:
    """
    Process a single JSONL line and convert to TSV format for parallel processing.
    
    Args:
        line (str): Single line from JSONL file containing JSON document.
    
    Returns:
        str: TSV-formatted line "doc_id\ttext" if valid document, None if invalid.
    """
    line = line.strip()
    if not line:
        return None
    
    try:
        data = json.loads(line)
        doc_id = str(data.get("id", "")).strip()
        if not doc_id:
            return None
        
        title = str(data.get("title", ""))
        text = str(data.get("text", ""))
        full_text = f"{title} {text}".strip()
        
        full_text = re.sub(r'[\n\r\t]+', ' ', full_text)
        full_text = re.sub(r'\s+', ' ', full_text).strip()
        
        return f"{doc_id}\t{full_text}" if full_text else None
    except (json.JSONDecodeError, KeyError):
        return None

def preprocess_doc_collection(
    docs_files: List[str],
    output_dir: str,
    split_passages: bool = False,
    passage_length: int = 180,
    overlap: int = 90,
    tokenizer: str = 'bert-base-multilingual-uncased',
    num_processes: Optional[int] = cpu_count() // 2
) -> None:
    """
    Process NeuCLIR docs.jsonl files to TSV format and save. Optionally split documents into passages.

    Args:
        docs_files (List[str]): List of paths to input JSONL files containing documents.
        output_dir (str): Base output directory path.
        split_passages (bool, optional): Whether to split documents into passages. Defaults to False.
        passage_length (int, optional): Maximum tokens per passage when splitting. Defaults to 180.
        overlap (int, optional): Overlapping tokens between passages. Defaults to 90.
        tokenizer (str, optional): HuggingFace tokenizer for passage splitting. 
                                       Defaults to 'bert-base-multilingual-uncased'.
        num_processes (Optional[int], optional): Number of parallel processes. Defaults to cpu_count() // 2.

    Returns:
        None
    """
    
    for docs_file in docs_files:
        try:
            if not os.path.exists(docs_file):
                logging.error(f"[DOC] File not found: {docs_file}")
                continue
            
            language_code = os.path.basename(os.path.dirname(docs_file))
            lang_dir = os.path.join(output_dir, language_code)
            os.makedirs(lang_dir, exist_ok=True)
            logging.info(f"[DOC] Processing {language_code}: {docs_file}")
            
            if split_passages:
                # 1) Define output file paths with tokenizer parameters
                tokenizer_name = tokenizer.replace("/", "_")
                passages_file = os.path.join(lang_dir, f"{language_code}_passages_{tokenizer_name}_{passage_length}_{overlap}.tsv")
                mapping_file = os.path.join(lang_dir, f"{language_code}_mapping_{tokenizer_name}_{passage_length}_{overlap}.tsv")

                # 2) Skip if passage files already exist
                if os.path.exists(passages_file) and os.path.exists(mapping_file):
                    logging.info(f"[DOC] Passage files already exist for {language_code}, skipping.")
                    continue

                # 3) Create temporary document TSV file (or reuse if exists)
                temp_docs_file = os.path.join(lang_dir, f"{language_code}_docs_temp.tsv")
                if os.path.exists(temp_docs_file):
                    logging.info(f"[DOC] Reusing existing temporary document file: {temp_docs_file}")
                else:
                    _convert_jsonl_to_tsv_parallel(docs_file, temp_docs_file, num_processes or cpu_count() // 2)
                    logging.info(f"[DOC] Created temporary document file: {temp_docs_file}")

                # 4) Split documents into passages with sequential IDs and save passage as TSV
                logging.info(f"[DOC] Splitting documents into passages (length={passage_length}, overlap={overlap})...")
                split_doc_to_passage(
                    input_tsv=temp_docs_file,
                    passage_file=passages_file,
                    mapping_file=mapping_file,
                    passage_length=passage_length,
                    overlap=overlap,
                    tokenizer=tokenizer,
                    nthreads=num_processes
                )

                # 5) Clean up temporary file
                if os.path.exists(temp_docs_file):
                    os.remove(temp_docs_file)
                    logging.info(f"[DOC] Cleaned up temporary file")                
            else:
                # 1) Standard document processing without passage splitting
                docs_file_output = os.path.join(lang_dir, f"{language_code}_docs.tsv")

                # 2) Skip if document file already exists
                if os.path.exists(docs_file_output):
                    logging.info(f"[DOC] Document file already exists for {language_code}, skipping.")
                    continue
                
                # 3) Save documents as TSV file
                _convert_jsonl_to_tsv_parallel(docs_file, docs_file_output, num_processes or cpu_count() // 2)
                logging.info(f"[DOC] Saved documents for {language_code}")
                
        except Exception as e:
            logging.error(f"[DOC] Failed processing {docs_file}: {e}")

def _convert_jsonl_to_tsv_parallel(
    docs_jsonl_file: str,
    output_file: str,
    num_processes: int
) -> None:
    """
    Convert JSONL to TSV using multiprocessing with streaming for memory efficiency.
    Processes large JSONL files in chunks to avoid loading the entire file into memory,
    which prevents memory overflow on large datasets (e.g., NeuCLIR with millions of documents).
    
    Args:
        docs_jsonl_file (str): Path to input JSONL file.
        output_file (str): Path to output TSV file.
        num_processes (int): Number of parallel worker processes.
    
    Returns:
        None
    """
    # 1) Create output directory if needed
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 2) Count total lines for progress bar without loading all into memory
    total_lines = sum(1 for _ in open(docs_jsonl_file, 'r', encoding='utf-8'))
    
    processed_doc_count = 0
    chunk_size = 1_000_000  # Process 1M lines at a time to balance memory usage and performance
    
    # 3) Process file in streaming fashion with large I/O buffers (8MB) for efficiency
    with Pool(processes=num_processes) as pool, \
         open(docs_jsonl_file, "r", encoding="utf-8", buffering=8*1024*1024) as input_file, \
         open(output_file, "w", encoding="utf-8", buffering=8*1024*1024) as output_handle:
        
        pbar = tqdm(total=total_lines, desc="Processing docs", unit='docs', unit_scale=True)
        
        while True:
            # 4) Read lines in chunks to avoid loading entire file into memory
            lines = []
            for _ in range(chunk_size):
                line = input_file.readline()
                if not line:
                    break
                lines.append(line)
            
            if not lines:
                break
            
            # 5) Process chunk with multiprocessing pool
            for result in pool.imap(process_jsonl_line, lines, chunksize=100):
                if result:
                    output_handle.write(result + "\n")
                    processed_doc_count += 1
                pbar.update(1)
        
        pbar.close()
    
    logging.info(f"[DOC] Processed {processed_doc_count:,} documents to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess NeuCLIR dataset and save as TSV files.")
    parser.add_argument('--input-dir', required=True, help='Input directory path')
    parser.add_argument('--output-dir', required=True, help='Output directory path')
    parser.add_argument('--topics-file', type=str, default="topics.0720.utf8.jsonl", help='Topics JSONL file to process')
    parser.add_argument('--qrels-file', nargs='+', required=True, help='List of paths to qrels files.')
    parser.add_argument('--docs-file', nargs='+', required=True, help='List of paths to docs.jsonl files.')
    parser.add_argument('--split-passages', action='store_true', help='Split documents into passages')
    parser.add_argument('--passage-length', type=int, default=180, help='Passage length (in tokens)')
    parser.add_argument('--overlap', type=int, default=90, help='Overlap between passages')
    parser.add_argument('--tokenizer', type=str, default='bert-base-multilingual-uncased', help='HuggingFace tokenizer name')
    parser.add_argument('--num-processes', type=int, default=None, help='Number of processes for parallel processing')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    args = parser.parse_args()
    
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format='%(asctime)s %(levelname)s %(message)s')
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Load qrels for each language (zho, fas, rus)
    if args.qrels_file:
        logging.info("[MAIN] Starting qrels processing...")
        qrels_by_lang = {}
        for qrels_file in args.qrels_file:
            lang_code = os.path.basename(qrels_file)[-3:] # Extract language code from original filename (last 3 characters)
            qrels_by_lang[lang_code] = load_qrels(qrels_file, lang_code)
        logging.info(f"[MAIN] Loaded qrels for {len(qrels_by_lang)} languages.")
    
    # Process topics to generate query TSV files
    if args.topics_file:
        logging.info("[MAIN] Starting topics processing...")
        topics_file_path = os.path.join(args.input_dir, args.topics_file)
        preprocess_topics_to_queries(topics_file_path, args.output_dir, qrels_by_lang)
        logging.info(f"[MAIN] Topics processing completed.")

    # Process documents to generate document/passage TSV files
    if args.docs_file:
        logging.info("[MAIN] Starting document processing...")
        preprocess_doc_collection(
            docs_files=args.docs_file,
            output_dir=args.output_dir,
            split_passages=args.split_passages,
            passage_length=args.passage_length,
            overlap=args.overlap,
            tokenizer=args.tokenizer,
            num_processes=args.num_processes
        )
        logging.info("[MAIN] Document processing completed.")

    logging.info("[MAIN] All NeuCLIR preprocessing completed!")

if __name__ == "__main__":
    main()
