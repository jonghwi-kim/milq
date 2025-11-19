import os
import argparse
import logging
from collections import defaultdict
from typing import List, Optional
from .clef_paths import ALL_LANGUAGES, DEFAULT_OUTPUT_DIR
from .clef_dataloader import load_documents, load_relevance_assessments, load_queries
from .utils import save_tsv, save_relevance_trec, split_doc_to_passage


def process_qrels(
    rel_langs: List[str],
    rel_years: List[str],
    output_dir: str,
    save: bool = True
) -> dict:
    """
    Process CLEF relevance assessments (qrels) for specified languages and years.
    
    Args:
        rel_langs (List[str]): Language codes (ISO 639-1) for relevance output. e.g., ['en', 'de', 'fr']
        rel_years (List[str]): Years for relevance assessment processing. e.g., ['2000', '2001', '2002']
        output_dir (str): Output directory path. e.g., '/path/to/output'
        save (bool, optional): Whether to save qrels files to disk. Defaults to True.
    
    Returns:
        dict: Relevance assessments indexed by (language, year).
              Example: {
                ('en', '2000'): {1: ['LA123', 'LA456'], 2: ['LA789'], ...}
              }
    """
    all_languages = [short for short, _ in ALL_LANGUAGES]
    qrels_dir = os.path.join(output_dir, "qrels")
    os.makedirs(qrels_dir, exist_ok=True)
    
    results = {}
    for lang in rel_langs:
        if lang not in all_languages:
            logging.warning(f"[QRELS] Language '{lang}' is not available. Skipping.")
            continue
            
        for year in rel_years:
            try:
                # 1) Check if qrels file already exists
                qrels_file = os.path.join(qrels_dir, f"{lang}_{year}_qrel.tsv")
                if save and os.path.exists(qrels_file):
                    logging.info(f"[QRELS] Qrels file already exists for {lang}-{year}, skipping.")
                    continue
                
                # 2) Load relevance assessments
                logging.info(f"[QRELS] Loading relevance assessments for {lang}-{year}...")
                relass = load_relevance_assessments(lang, year=year)
                results[(lang, year)] = relass
                
                # 3) Save qrels file if requested
                if save:
                    save_relevance_trec(relass, os.path.join(qrels_dir, f"{lang}_{year}_qrel.tsv"))
                    logging.info(f"[QRELS] Saved relevance assessments for {lang}-{year}")
                    
            except Exception as e:
                logging.error(f"[QRELS] Failed {lang}-{year}: {e}")

    return results

def process_queries(
    query_langs: List[str],
    query_years: List[str],
    output_dir: str,
    save: bool = True
) -> dict:
    """
    Process CLEF topics (queries) for specified languages and years.
    
    Args:
        query_langs (List[str]): Language codes (ISO 639-1) for query output. e.g., ['de', 'fr', 'fi']
        query_years (List[str]): Years for query processing. e.g., ['2000', '2001', '2002']
        output_dir (str): Output directory path. e.g., '/path/to/output'
        save (bool, optional): Whether to save query files to disk. Defaults to True.
    
    Returns:
        dict: Query data indexed by (language, year).
              Example: {
                ('de', '2000'): {'ids': [1, 2, ...], 'titles': ['title1', 'title2', ...], 'descs': ['desc1', 'desc2', ...], 'concat': ['title1 desc1', 'title2 desc2', ...]},
              }
    """
    all_languages = [short for short, _ in ALL_LANGUAGES]
    queries_dir = os.path.join(output_dir, "queries")
    os.makedirs(queries_dir, exist_ok=True)
    
    results = {}
    for lang in query_langs:
        if lang not in all_languages:
            logging.warning(f"[QUERY] Language '{lang}' is not available. Skipping.")
            continue
            
        for year in query_years:
            try:
                # 1) Check if all query files already exist
                title_file = os.path.join(queries_dir, f"{lang}_{year}_queries_title.tsv")
                desc_file = os.path.join(queries_dir, f"{lang}_{year}_queries_desc.tsv")
                concat_file = os.path.join(queries_dir, f"{lang}_{year}_queries_concat.tsv")
                
                if save and all(os.path.exists(f) for f in [title_file, desc_file, concat_file]):
                    logging.info(f"[QUERY] Query files already exist for {lang}-{year}, skipping.")
                    continue
                
                # 2) Load queries from CLEF sources
                logging.info(f"[QUERY] Loading queries for {lang}-{year}...")
                query_ids, queries = load_queries(language=lang, year=year)
                
                # 3) Extract different query fields
                titles = [query['title'] for query in queries]
                descs = [query['desc'] for query in queries]
                concat = [query['title'] + ' ' + query['desc'] for query in queries]
                
                results[(lang, year)] = {
                    "ids": query_ids, "title": titles, "desc": descs, "concat": concat
                }
                
                # 4) Save query files if requested
                if save:
                    save_tsv(query_ids, titles, os.path.join(queries_dir, f"{lang}_{year}_queries_title.tsv"))
                    save_tsv(query_ids, descs, os.path.join(queries_dir, f"{lang}_{year}_queries_desc.tsv"))
                    save_tsv(query_ids, concat, os.path.join(queries_dir, f"{lang}_{year}_queries_concat.tsv"))
                    logging.info(f"[QUERY] Saved {len(query_ids)} queries for {lang}-{year}")
                    
            except Exception as e:
                logging.error(f"[QUERY] Failed {lang}-{year}: {e}")

    return results

def aggregate_qrels_queries(
    results: dict,
    output_dir: str,
    query_languages: list
) -> None:
    """
    Filter and aggregate CLEF qrels & queries (Bonab et al. Training Effective Neural CLIR by Bridging the Translation Gap (SIGIR 2020)).
    Keeps only queries with at least one relevant LA Times doc.
    
    Args:
        results (dict): Output from preprocessing containing qrels and queries.
        output_dir (str): Output directory path. e.g., '/path/to/output'
        query_languages (List[str]): Query language codes to aggregate. e.g., ['de', 'fr', 'fi']
    
    Returns:
        None
    """
    years = ["2000", "2001", "2002", "2003"]

    # 1. Filter qrels: only LA Times docs, only C001-C200
    filtered_qrels = defaultdict(set)
    for year in years:
        rel_assess = results["qrels"].get(("en", year))
        assert rel_assess is not None, f"Missing English qrels for year {year}"
        for query_id, doc_ids in rel_assess.items():
            qid = int(query_id)
            assert 1 <= qid <= 200, f"Query id {qid} out of C001-C200 range"
            for doc_id in doc_ids:
                if str(doc_id).startswith("LA"):
                    filtered_qrels[qid].add(str(doc_id))

    queries_dir = os.path.join(output_dir, "queries")
    qrels_dir = os.path.join(output_dir, "qrels")

    # 2. For each query language, aggregate and save queries
    for lang in query_languages:
        # 1) Merge queries for all years into a single dict per type
        title, desc, concat = {}, {}, {}
        for year in years:
            query_data = results["queries"].get((lang, year))
            if query_data is None:
                logging.warning(f"[AGG] Missing queries for {lang}-{year}, skipping")
                continue
            for idx, query_id in enumerate(query_data["ids"]):
                qid = int(query_id)
                assert 1 <= qid <= 200, f"Query id {qid} out of C001-C200 range"
                title[qid] = query_data["title"][idx]
                desc[qid] = query_data["desc"][idx]
                concat[qid] = query_data["concat"][idx]
        # 2) Filter queries with at least one relevant LA Times doc
        filtered_qids = sorted(
            qid for qid in concat
            if filtered_qrels[qid]
        )
        # 3) Save queries (title, desc, concat) using save_tsv
        save_tsv(filtered_qids, [title[qid] for qid in filtered_qids], os.path.join(queries_dir, f"{lang}_2000-2003_queries_title.tsv"))
        save_tsv(filtered_qids, [desc[qid] for qid in filtered_qids], os.path.join(queries_dir, f"{lang}_2000-2003_queries_desc.tsv"))
        save_tsv(filtered_qids, [concat[qid] for qid in filtered_qids], os.path.join(queries_dir, f"{lang}_2000-2003_queries_concat.tsv"))

    # 3. Save English qrels using save_relevance_trec
    filtered_qrels_dict = {str(qid): sorted(list(filtered_qrels[qid])) for qid in filtered_qids}
    save_relevance_trec(filtered_qrels_dict, os.path.join(qrels_dir, "en_2000-2003_qrels.tsv"))
    logging.info(f"[AGG] Aggregated {len(filtered_qids)} queries with relevant LA Times docs")

def process_documents(
    doc_langs: List[str],
    doc_years: List[str],
    output_dir: str,
    split_passages: bool = False,
    passage_length: int = 180,
    overlap: int = 90,
    tokenizer: str = 'bert-base-multilingual-uncased',
    num_processes: Optional[int] = None
) -> None:
    """
    Process CLEF documents sources for specified languages and years and save as TSV files. Optionally split documents into passages.

    Args:
        doc_langs (List[str]): Language codes (ISO 639-1) for document output. e.g., ['en', 'de']
        doc_years (List[str]): Years for document processing. e.g., ['2000', '2001']
        output_dir (str): Base output directory where docs subdirectory will be created.
        split_passages (bool, optional): Whether to split documents into passages. Defaults to False.
        passage_length (int, optional): Maximum tokens per passage when splitting. Defaults to 180.
        overlap (int, optional): Overlapping tokens between passages. Defaults to 90.
        tokenizer (str, optional): HuggingFace tokenizer name for passage splitting. Defaults to 'bert-base-multilingual-uncased'.
        num_processes (Optional[int], optional): Number of parallel processes for passage splitting. Defaults to None.
    
    Returns:
        None
    """
    all_languages = [short for short, _ in ALL_LANGUAGES]
    docs_dir = os.path.join(output_dir, "docs")
    os.makedirs(docs_dir, exist_ok=True)
    
    for lang in doc_langs:
        if lang not in all_languages:
            logging.warning(f"[DOC] Language '{lang}' is not available. Skipping.")
            # continue
            
        for year in doc_years:
            try:
                # 1) Load documents from CLEF sources
                logging.info(f"[DOC] Loading documents for {lang}-{year}...")
                doc_ids, documents = load_documents(language=lang, year=year)
                
                if split_passages:
                    # 2) Define output file paths with tokenizer parameters
                    tokenizer_name = tokenizer.replace("/", "_")
                    passages_file = os.path.join(docs_dir, f"{lang}_{year}_passages_{tokenizer_name}_{passage_length}_{overlap}.tsv")
                    mapping_file = os.path.join(docs_dir, f"{lang}_{year}_mapping_{tokenizer_name}_{passage_length}_{overlap}.tsv")
                    
                    # 3) Skip if passage files already exist
                    if os.path.exists(passages_file) and os.path.exists(mapping_file):
                        logging.info(f"[DOC] Passage files already exist for {lang}-{year}, skipping.")
                        continue
                    
                    # 4) Create temporary document TSV file (or reuse if exists)
                    temp_docs_file = os.path.join(docs_dir, f"{lang}_{year}_docs_temp.tsv")
                    if os.path.exists(temp_docs_file):
                        logging.info(f"[DOC] Reusing existing temporary document file: {temp_docs_file}")
                    else:
                        save_tsv(doc_ids, documents, temp_docs_file)
                        logging.info(f"[DOC] Created temporary document file: {len(doc_ids)} documents")
                    
                    # 5) Split documents into passages with sequential IDs and save passage as TSV
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
                    
                    # 6) Clean up temporary file
                    if os.path.exists(temp_docs_file):
                        # os.remove(temp_docs_file)
                        logging.info(f"[DOC] Cleaned up temporary file")
                else:
                    # 2) Standard document processing without passage splitting
                    docs_file = os.path.join(docs_dir, f"{lang}_{year}_docs.tsv")
                    
                    # 3) Skip if document file already exists
                    if os.path.exists(docs_file):
                        logging.info(f"[DOC] Document file already exists for {lang}-{year}, skipping.")
                        continue
                    
                    # 4) Save documents as TSV file
                    save_tsv(doc_ids, documents, os.path.join(docs_dir, f"{lang}_{year}_docs.tsv"))
                    logging.info(f"[DOC] Saved {len(doc_ids)} documents for {lang}-{year}")
                
            except Exception as e:
                logging.error(f"[DOC] Failed {lang}-{year}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess CLEF dataset and save as TSV files.")
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--doc-langs', nargs='+', help='List of languages for document output (e.g. en)')
    parser.add_argument('--doc-years', nargs='+', help='List of years for document output (e.g. 2000)')
    parser.add_argument('--rel-langs', nargs='+', help='List of languages for relevance output (e.g. en)')
    parser.add_argument('--rel-years', nargs='+', help='List of years for relevance output (e.g. 2000 2001 2002 2003)')
    parser.add_argument('--query-langs', nargs='+', help='List of languages for query/topic output (e.g. de fr fi sw so)')
    parser.add_argument('--query-years', nargs='+', help='List of years for query/topic output (e.g. 2000 2001 2002 2003)')
    parser.add_argument('--aggregate-queries', action='store_true', help='Aggregate queries and qrels across years')
    parser.add_argument('--split-passages', action='store_true', help='Split documents into passages')
    parser.add_argument('--passage-length', type=int, default=180, help='Passage length (in tokens)')
    parser.add_argument('--overlap', type=int, default=90, help='Overlap between passages')
    parser.add_argument('--tokenizer', type=str, default='bert-base-multilingual-uncased', help='HuggingFace tokenizer name')
    parser.add_argument('--num-processes', type=int, help='Number of processes for parallel processing')
    parser.add_argument('--log-level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR)')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format='%(asctime)s %(levelname)s %(message)s')

    results = {"qrels": {}, "queries": {}}
    
    # Process relevance assessments if specified
    if args.rel_langs and args.rel_years:
        logging.info("[MAIN] Starting qrels processing...")
        qrels_results = process_qrels(
            rel_langs=args.rel_langs,
            rel_years=args.rel_years,
            output_dir=args.output_dir,
            save=not args.aggregate_queries
        )
        results["qrels"] = qrels_results
        logging.info(f"[MAIN] Qrels processing completed.")
    
    # Process queries if specified
    if args.query_langs and args.query_years:
        logging.info("[MAIN] Starting query processing...")
        query_results = process_queries(
            query_langs=args.query_langs,
            query_years=args.query_years,
            output_dir=args.output_dir,
            save=not args.aggregate_queries
        )
        results["queries"] = query_results
        logging.info(f"[MAIN] Query processing completed.")
    
    # Aggregate queries and qrels if requested
    if args.aggregate_queries:
        logging.info("[MAIN] Starting aggregation...")
        if not (args.query_langs and args.query_years and args.rel_langs and args.rel_years):
            logging.warning("[MAIN] Aggregation requires both query and qrels data. Skipping aggregation.")
        else:
            aggregate_qrels_queries(results, args.output_dir, args.query_langs)

    # Process documents if specified
    if args.doc_langs and args.doc_years:
        logging.info("[MAIN] Starting document processing...")
        process_documents(
            doc_langs=args.doc_langs,
            doc_years=args.doc_years,
            output_dir=args.output_dir,
            split_passages=args.split_passages,
            passage_length=args.passage_length,
            overlap=args.overlap,
            tokenizer=args.tokenizer,
            num_processes=args.num_processes
        )
        logging.info("[MAIN] Document processing completed.")

    logging.info("[MAIN] All CLEF preprocessing completed!")

if __name__ == "__main__":
    main()