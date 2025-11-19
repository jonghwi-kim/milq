
import os
import gzip
import codecs
import urllib
from tqdm import tqdm

from typing import Union
from lxml import html as etree
from .clef_paths import get_lang2pair, PATH_BASE_QUERIES, PATH_BASE_EVAL, CLEF_LOWRES_DIR, all_paths
from typing import Tuple, List


def _decode_xml(path, encoding):
    """
    Decode CLEF SGML files, transparently handling gzip-compressed inputs.
    """
    is_gzip = path.endswith(".gz")
    if is_gzip:
        # gzip.open supports text mode with encoding arguments
        if encoding == "plain":
            open_kwargs = {"mode": "rt"}
        else:
            open_kwargs = {"mode": "rt", "encoding": encoding, "errors": "ignore"}
        with gzip.open(path, **open_kwargs) as f:
            xml = f.read()
    else:
        if encoding == "plain":
            with open(path, "r") as f:
                xml = f.read()
        else:
            with codecs.open(path, encoding=encoding, errors="ignore") as f:
                xml = f.read()
    
    xml = xml.replace("<BODY>", "<BODYY>").replace("</BODY>", "</BODYY>")
    xml = xml.replace("<HEAD>", "<HEADD>").replace("</HEAD>", "</HEADD>")
    xml = "<root>" + xml + "</root>"
    xml = xml.replace("<IZV>", "").replace("</IZV>", "")  # for russian corpus
    return etree.fromstring(xml)


def load_relevance_assessments(language: str, year: Union[str, list], load_non_relevant_docs=False):
    if type(year) == str:
        assert year in ["2000", "2001", "2002", "2003"]
    else:
        for y in year:
            assert y in ["2000", "2001", "2002", "2003"]
    doc_lang_short, doc_lang_full = get_lang2pair(language)
    if type(year) == str:
        year = [year]

    positive_list = {}
    negative_list = {}
    for y in year:
        # Special handling for English 2000 and 2001
        if doc_lang_short.lower() == "en":
            if y == "2000":
                path = os.path.join(PATH_BASE_EVAL, f"{y}rels", "biling_qrels")
            elif y == "2001":
                path = os.path.join(PATH_BASE_EVAL, f"{y}rels", "qrels_bilingual")
            else:
                path = os.path.join(PATH_BASE_EVAL, f"{y}rels", "qrels_" + doc_lang_full)
        else:
            path = os.path.join(PATH_BASE_EVAL, f"{y}rels", "qrels_" + doc_lang_full)

        if not os.path.exists(path):
            # Try fallback to short code upper
            path = os.path.join(PATH_BASE_EVAL, f"{y}rels", "qrels_" + doc_lang_short.upper())
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f.readlines():
                tokens = line.rstrip("\n").split(" ")
                # check if document is relevant for query
                relevance = int(tokens[len(tokens) - 1])
                if relevance != 0:
                    query_id = int(tokens[0].strip())
                    document_id = tokens[2].strip()
                    if query_id not in positive_list:
                        relevant_docs = [document_id]
                    else:
                        relevant_docs = positive_list[query_id]
                        relevant_docs.append(document_id)
                    positive_list[query_id] = relevant_docs

                if load_non_relevant_docs and relevance == 0:
                    query_id = int(tokens[0].strip())
                    document_id = tokens[2].strip()
                    if query_id not in negative_list:
                        non_relevant_docs = [document_id]
                    else:
                        non_relevant_docs = negative_list[query_id]
                        non_relevant_docs.append(document_id)
                    negative_list[query_id] = non_relevant_docs

    result = negative_list if load_non_relevant_docs else positive_list
    return result

_docs_cache = {}
def load_documents(language: str, year:Union[list, str], limit_documents:int=None, only_body:bool=False, **kwargs):
    """
    Walks through document files and extracts content.
    :param language: corpus language
    :param year: year of CLEF campaign (2001-2003)
    :param limit_documents: load only top k documents
    :param only_body: used for masked language modelling corpus
    :return:
    """
    def _load_file(file_path, extractor, encoding, limit=None, only_body=False, **kwargs):
        tree = _decode_xml(file_path, encoding)
        docs = []
        ids = []
        rerank_set = kwargs.get("rerank_corpus", None)
        doc_list = list(tree)
        for i, doc in enumerate(tqdm(doc_list, desc=f"Parsing {os.path.basename(file_path)}", leave=False)):
            if len(docs) == limit:
                break
            document_id, full_text = extractor(doc, only_body=only_body)
            if document_id is not None and full_text is not None:
                if rerank_set and document_id not in rerank_set:
                    continue
                ids.append(document_id)
                docs.append(full_text)
        return ids, docs
    
    doc_lang_iso, doc_lang_full = get_lang2pair(language)
    key = f"{doc_lang_iso}_{year}_{limit_documents}"
    
    # Always load corpus if we do re-ranking.
    do_reranking = bool(kwargs.get("rerank_corpus", False))
    corpus_is_in_memory = key in _docs_cache
    
    if not corpus_is_in_memory or do_reranking:
        doc_dirs = all_paths[doc_lang_iso][year]
        documents = []
        doc_ids = []
        limit_reached = False
        encoding = 'UTF-8' if doc_lang_iso == "ru" or doc_lang_full == "russian" else 'ISO-8859-1'
        for doc_dir, extractor in tqdm(doc_dirs, desc=f"Doc dirs for {doc_lang_iso}-{year}"):
            if not limit_reached:
                files = [file for file in next(os.walk(doc_dir))[2] if not file.endswith(".dtd") and not file.startswith("README")]
                for file in tqdm(files, desc=f"Files in {os.path.basename(doc_dir)}", leave=False):
                    tmp_doc_ids, tmp_documents = _load_file(
                        file_path = os.path.join(doc_dir, file),
                        encoding = encoding,
                        extractor=extractor,
                        limit=limit_documents,
                        only_body=only_body, **kwargs
                    )
                    documents.extend(tmp_documents)
                    doc_ids.extend(tmp_doc_ids)
                    if limit_documents and len(documents) >= limit_documents:
                        limit_reached = True
                        break
        if not do_reranking:
            _docs_cache[key] = (doc_ids, documents)
    else:
        doc_ids, documents = _docs_cache[key]
    
    return doc_ids, documents


def load_queries(language, year: Union[list, str], limit=None, encoding=None, include_desc=True) -> Tuple[List, List]:
    language = language.lower()
    
    if language == "sw" or language == "swahili":
        return _load_swahili_queries(year)
    
    if language == "so" or language == "somali":
        return _load_somali_queries(year)
    
    if language == "tr" or language == "turkish":
        return _load_turkish_queries(year)
    
    if language == "ug" or language == "uyghur":
        return _load_uyghur_queries(year)
    
    if language == "kg" or language == "kyrgyz":
        return _load_kyrgyz_queries(year)
    
    if year == "2000":
        return _load_clef2000_queries(language)
    
    queries = []
    ids = []
    lang_iso, lang_full = get_lang2pair(language)
    
    if type(year) == str:
        year = [year]
      
    for y in year:
        if lang_iso == "fi" and y == "2002":
            filename = "Top-fi50-02.txt"
        else:
            filename = "Top-" + lang_iso + y[-2:] + ".txt"
        path = os.path.join(PATH_BASE_QUERIES, f"topics{y}", filename)

        if not encoding:
            encoding = 'UTF-8' if lang_iso == "ru" or lang_full == "russian" in path else 'ISO-8859-1'

        tree = _decode_xml(path, encoding)
        tag_title = lang_iso + '-title'
        tag_desc = lang_iso + '-desc'
        # tag_narr = language_tag + '-narr' # excluding narrations field

        for i, topic in enumerate(list(tree)):
            _id = topic.findtext('num').strip() # e.g. 'C041'
            _id = int(_id[1:]) # e.g. 41
            title = topic.findtext(tag_title).strip().replace("\n", "").replace("\r", "")
            if include_desc:
                desc = topic.findtext(tag_desc).strip().replace("\n", "").replace("\r", "")
            else:
                desc = ''
            queries.append({'title': title, 'desc': desc})
            ids.append(_id)
            if i == limit:
                break
        
    # newly added:
    # queries = [q.replace("\n", " ").replace("\r", " ") for q in queries]
    
    return ids, queries


def load_clef(query_lang: str, doc_lang: str, year: Union[str, list] = "2003", limit_documents=None, **kwargs):
    # year = str(year)
    # assert year in ["2001", "2002", "2003", "all"]
    
    query_lang_iso, query_lang_full = get_lang2pair(query_lang)
    
    # Load relevance assessments
    relass = load_relevance_assessments(doc_lang, year=year)
    
    # Load queries
    query_ids, queries = load_queries(
        language=query_lang_iso,
        year=year,
        limit=None,
        encoding="plain" if query_lang_iso == "ru" else None,
        include_desc=kwargs.get("include_desc", True)
    )
    
    # Load documents
    doc_ids, documents = load_documents(language=doc_lang, year=year, limit_documents=limit_documents, **kwargs)
    return doc_ids, documents, query_ids, queries, relass


def load_clef_rerank(dlang: str, qlang: str, rerank_dir: str, topk: int, include_desc: bool=True, year: Union[int, str]="2003"):
    """
    For efficiency, we only load a document if it's within the top-k of any query.
    :param dlang: ISO-code of document language
    :param qlang: ISO-code of query language
    :param rerank_dir: directory containing one file for each query (filename = query-id), content: list of document-ids (=pre-ranking).
    :param topk: cut-off, number of documents to re-rank for each query
    :param include_desc: clef description 
    :param year: clef year
    :return: document ids, documents, queries, query ids, relevance assessments, list of top-k document ids for each query (=input to be re-ranked)
    """

    qid2topk_rerank, rerank_corpus = load_ranking_files(rerank_dir, topk, year)
    
    doc_ids, documents, query_ids, queries, relass = load_clef(
        query_lang=qlang,
        doc_lang=dlang,
        rerank_corpus=rerank_corpus,
        include_desc=include_desc,
        year=year)
    return doc_ids, documents, queries, query_ids, relass, qid2topk_rerank


def load_ranking_files(rerank_dir, topk=-1, year=None):
    qid2topk_rerank = {}
    rerank_corpus = set()

    if year == "2001":
        qid_begin = 0
        qid_end = 90
    elif year == "2002":
        qid_begin = 91
        qid_end = 140
    elif year == "2003":
        qid_begin = 141
        qid_end = 200
    else:
        # for hc4
        qid_begin = 0
        qid_end = float('inf')

    paths = [elem for elem in os.listdir(rerank_dir) if qid_begin <= int(elem.split(".")[0]) <= qid_end]
    paths = sorted([e for e in paths], key=lambda elem: int(elem.split(".")[0]))

    for p in paths:
        qid = int(p.split("/")[-1][:-4])
        rerank_doc_ids = []
        with open(os.path.join(rerank_dir, p), "r") as f:
            for i, did_score in enumerate(f):
                did = did_score.split("\t")[0]
                did = did.strip()
                rerank_doc_ids.append(did)
                if i < topk:
                    rerank_corpus.add(did)
        qid2topk_rerank[qid] = rerank_doc_ids
    return qid2topk_rerank, rerank_corpus


def _load_clef2000_queries(language):
    short2lang = {"en": "E", "fr": "F", "fi": "FI", "de": "G", "it": "I", "sw": "SW", "es": "SP"}
    long2lang = {"english": "E", "french": "F", "finnish": "FI", "german": "G", "italian": "I", "swedish": "SW", "spanish": "SP"}
    if language in short2lang:
        lang = short2lang[language]
    elif language in long2lang:
        lang = long2lang[language]
    else:
        raise NotImplementedError
    
    encoding = 'UTF-8' if lang == "ru" or lang == "russian" else 'ISO-8859-1'
    with codecs.open(os.path.join(PATH_BASE_QUERIES, "topics2000", f"TOP-{lang}.txt"), encoding=encoding) as f:
        lines = f.readlines()
    
    topics = []
    topic = {}
    lines = iter(lines)
    for line in lines:
        line = line.strip()
        if line == "":
            continue

        elif line.startswith("<num>"):
            assert 'num' not in topic
            num_val = line.replace("<num>", "").strip()
            if num_val == "":
                topic['num'] = next(lines).strip()
            else:
                topic['num'] = num_val

        elif line.startswith(f"<{lang}-title>") or line.startswith(f"<{lang}-desc>") or line.startswith(f"<{lang}-narr>"):
            field_name = None
            if line.startswith(f"<{lang}-title>"):
                field_name = 'title'
                tag_marker = f"<{lang}-title>"
            elif line.startswith(f"<{lang}-desc>"):
                field_name = 'desc'
                tag_marker = f"<{lang}-desc>"
            elif line.startswith(f"<{lang}-narr>"):
                field_name = 'narr'
                tag_marker = f"<{lang}-narr>"

            content = line.replace(tag_marker, "").strip()
            field_lines = [content] if content else []
            for next_line in lines:
                next_line = next_line.strip()
                if not next_line or next_line.startswith("<"):
                    if next_line and next_line.startswith("<"):
                        lines = [next_line] + list(lines)
                    break
                field_lines.append(next_line)
            assert field_name not in topic
            topic[field_name] = "".join(field_lines)

        else:
            if line.strip() in ["<top>", "</top>"]:
                if topic:
                    assert all([tag in topic for tag in ['num', 'title', 'desc']])
                    topics.append(topic)
                topic = {}

    if topic: topics.append(topic)

    queries = []
    qids = []
    for topic in topics:
        qids.append(int(topic['num'][1:]))
        queries.append({'title': topic['title'], 'desc': topic.get('desc','')})
    return qids, queries

# TODO: move UG, TR, KG into separate folder, away from long_paper folder to rlitschk folder
def _load_lowres_queries(filepath: str, year: Union[str, list]):
    if not os.path.exists(filepath):
        url = "https://ciir.cs.umass.edu/downloads/ictir19_simulate_low_resource/ictir19_simulate_low_resource.zip"
        raise FileNotFoundError(f"Please download\n\n{url}\n\nand set CLEF_LOWRES_DIR in paths.py")
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    qids, _queries = [], []
    is_valid_query_id = []

    # CLEF query ids
    if year == "2000":
        is_valid_query_id_2000 = lambda query_id: 0 <= query_id <= 40
        is_valid_query_id.append(is_valid_query_id_2000)
    elif year == "2001":
        is_valid_query_id_2001 = lambda query_id: 41 <= query_id <= 90
        is_valid_query_id.append(is_valid_query_id_2001)
    elif year == "2002":
        is_valid_query_id_2002 = lambda query_id: 91 <= query_id <= 140
        is_valid_query_id.append(is_valid_query_id_2002)
    else:
        assert year == "2003"
        is_valid_query_id_2003 = lambda query_id: 141 <= query_id <= 200
        is_valid_query_id.append(is_valid_query_id_2003)

    while lines:
        qid_line = lines.pop(0)
        qid = qid_line.split("query_id:")[-1].strip()
        qid = int(qid[1:])
        title_line = lines.pop(0)
        title = title_line.split("title:")[-1].strip()
        desc_line = lines.pop(0)
        desc = desc_line.split("description:")[-1].strip()
        while lines and lines[0] != "":
            desc_line = lines.pop(0)
            desc_p2 = desc_line.split("description:")[-1]
            desc = f"{desc} {desc_p2}"
        while lines and lines[0] == "":
            lines.pop(0)
        if any(fn(qid) for fn in is_valid_query_id):
            qids.append(qid)
            _queries.append({'title': title, 'desc': desc})
    return qids, _queries

def _load_swahili_queries(year: str):
    return _load_lowres_queries(
        os.path.join(CLEF_LOWRES_DIR, "clef-en-2000-2003-wo-narrative-Day-2_SWAHILI.txt"),
        year
    )


def _load_somali_queries(year: str):
    return _load_lowres_queries(
        os.path.join(CLEF_LOWRES_DIR, "clef-en-2000-2003-wo-narrative-Day-2_SOMALI.txt"),
        year
    )


def _load_kyrgyz_queries(year: str):
    return _load_lowres_queries(
        os.path.join(CLEF_LOWRES_DIR, "clef-en-2000-2003_KYRGYZ.txt"),
        year
    )


def _load_turkish_queries(year: str):
    return _load_lowres_queries(
        os.path.join(CLEF_LOWRES_DIR, "clef-en-2000-2003_TURKISH.txt"),
        year
    )


def _load_uyghur_queries(year: str):
    return _load_lowres_queries(
        os.path.join(CLEF_LOWRES_DIR, "clef-en-2000-2003_UYGHUR.txt"),
        year
    )
