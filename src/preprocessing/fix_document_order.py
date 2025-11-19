from pathlib import Path
import json
import gzip
import argparse
from hashlib import md5

from tqdm.auto import tqdm

def hash_doc(e):
    return md5( (e['title'].strip() + e['text'].strip()).encode('utf-8') ).hexdigest()

def main(args):
    ordered_ids = []
    hashs = {}
    for id_file in args.id_file:
        with (gzip.open(id_file, 'rt') if id_file.suffix == '.gz' else open(id_file)) as fp:
            for line in tqdm(fp, desc=f'Reading {id_file}'):
                line = json.loads(line)
                ordered_ids.append(line['id'])
                if args.check_hash:
                    hashs[ line['id'] ] = line['md5']

    # getting doc_id and offset mapping
    docs_pos = {}
    doc_fp = open(args.raw_download_file)
    with tqdm(total=len(ordered_ids), desc='Reading downloaded file') as pbar:
        doc_offset = 0
        while True:
            line = doc_fp.readline()
            if not line:
                break
            line = json.loads(line)
            if args.check_hash and hash_doc(line) != hashs[ line['id'] ]:
                print(f"Doc {line['id']} hash mismatch -- should be {hashs[line['id']]} but got {hash_doc(line)}")
            doc_id = line['id']
            docs_pos[doc_id] = doc_offset
            doc_offset = doc_fp.tell()
            pbar.update()
    
    # assert len(ordered_ids) == len(docs_pos), \
    #        f"Downloaded {len(docs_pos)} unique documents but id file(s) have {len(docs_pos)} unique ids."

    missing_ids = set(ordered_ids) - docs_pos.keys()
    assert len(missing_ids) == 0, \
            f"There are {len(missing_ids)} documents specified in {args.id_file} that are not in {args.raw_download_file}"

    output_file = args.raw_download_file.with_name(f"{args.raw_download_file.name}.sorted")
    
    with open(output_file, 'w') as fw:
        for doc_id in tqdm(ordered_ids, desc="Writing sorted docuements"):
            doc_fp.seek(docs_pos[doc_id])
            fw.write( doc_fp.readline() )
        
    doc_fp.close()

    # moving files
    print("Backing up the original file...")
    args.raw_download_file.rename( args.raw_download_file.with_name(f"{args.raw_download_file.name}.bak") )
    output_file.rename(args.raw_download_file)
    print("Done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_download_file', type=Path, help='file containing the documents for a particular lang', required=True)
    parser.add_argument('--id_file', type=Path, help='file containing ids for the language', nargs='+', required=True)
    parser.add_argument('--check_hash', action='store_true', default=False, help="Validate document hashes during download.")

    args = parser.parse_args()
    
    # reorder the id_files
    if len(args.id_file) > 1:
        args.id_file = sorted(args.id_file, key=lambda x: int(x.name.split(".")[1]))
    
    main(args)