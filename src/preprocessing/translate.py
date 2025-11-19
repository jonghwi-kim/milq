"""Translate MSMARCO query shards with an NLLB model."""

import argparse
import csv
import os
from typing import List, Tuple

import ftfy
import nltk
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


nltk.download("punkt", quiet=True)


class MSMarco(Dataset):
    """Loads query IDs and sentences from an MSMARCO `query.tsv` file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.collection = self._load_tsv()

    def __len__(self) -> int:
        return len(self.collection)

    def __getitem__(self, idx: int) -> Tuple[str, str]:
        return self.collection[idx]

    def _load_tsv(self) -> List[Tuple[str, str]]:
        """Reads the TSV file and returns (query_id, sentence) pairs."""
        collection: List[Tuple[str, str]] = []

        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as tsv_file:
            reader = csv.reader(tsv_file, delimiter="\t")
            for line in tqdm(reader, desc="Reading query.tsv"):
                if len(line) < 2:
                    continue

                query_id = line[0].strip()
                query_text = ftfy.fix_text(line[1].strip())
                sentences = nltk.tokenize.sent_tokenize(query_text)

                for sent in sentences:
                    if sent:
                        collection.append((query_id, sent))
        return collection


def parse_args() -> argparse.Namespace:
    """Builds the CLI used to configure translation jobs."""
    parser = argparse.ArgumentParser(description="Translate MSMARCO query.tsv entries with NLLB-200.")
    parser.add_argument(
        "--model_name_or_path",
        default="facebook/nllb-200-distilled-600M",
        type=str,
        help="NLLB-200 model checkpoint.",
    )
    parser.add_argument("--source_language", required=True, type=str, help="Source language code (e.g., eng_Latn).")
    parser.add_argument("--target_language", required=True, type=str, help="Target language code (e.g., kor_Hang).")
    parser.add_argument("--input_file", required=True, type=str, help="Path to MSMARCO query.tsv file.")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory to store translated queries.")
    parser.add_argument("--batch_size", default=128, type=int, help="Batch size per GPU for translation.")
    parser.add_argument("--num_workers", default=4, type=int, help="Number of DataLoader workers.")
    parser.add_argument("--num_beams", default=5, type=int, help="Beam search width.")
    parser.add_argument("--max_seq_length", default=128, type=int, help="Maximum input/output sequence length.")
    return parser.parse_args()


def translate(tsv_dataset: MSMarco, args: argparse.Namespace) -> None:
    """Streams the dataset through the translation model and writes TSV output."""
    
    # Detect if running under torchrun (checks for RANK environment variable)
    is_distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    
    if is_distributed:
        # Initialize distributed process group
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.src_lang = args.source_language
    tokenizer.tgt_lang = args.target_language

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
    )
    model.to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    model.eval()

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(args.target_language)
    if forced_bos_token_id == tokenizer.unk_token_id:
        raise ValueError(f"Target language {args.target_language} not supported by the selected tokenizer.")

    # Use DistributedSampler for multi-GPU
    sampler = DistributedSampler(tsv_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None
    
    dataloader = DataLoader(
        tsv_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False if is_distributed else False,
        sampler=sampler,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Rank-specific output file for distributed processing
    if is_distributed:
        output_path = os.path.join(args.output_dir, f"{args.target_language}_queries_rank{rank}.tsv")
    else:
        output_path = os.path.join(args.output_dir, f"{args.target_language}_queries.tsv")

    with open(output_path, "w", encoding="utf-8") as writer:
        desc = f"Translating (GPU {rank}/{world_size})" if is_distributed else "Translating queries"
        for batch in tqdm(dataloader, desc=desc, disable=rank != 0):
            doc_ids, collection = batch

            tokenized = tokenizer(
                list(collection),
                padding=True,
                truncation=True,
                max_length=args.max_seq_length,
                return_tensors="pt",
            )
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            with torch.no_grad(), torch.cuda.amp.autocast():
                # Get the actual model for generation (unwrap DDP if needed)
                model_to_generate = model.module if is_distributed else model
                translated = model_to_generate.generate(
                    **tokenized,
                    forced_bos_token_id=forced_bos_token_id,
                    num_beams=args.num_beams,
                    max_length=args.max_seq_length,
                    do_sample=False,
                    use_cache=True,
                )

            decoded = tokenizer.batch_decode(translated, skip_special_tokens=True)
            for query_id, translated_text in zip(doc_ids, decoded):
                writer.write(f"{query_id}\t{translated_text.strip()}\n")

    # Wait for all processes to finish writing
    if is_distributed:
        dist.barrier()
        if rank == 0:
            print(f"\nAll {world_size} GPUs finished translation. Merging results...")
    
    # Merge results on rank 0
    if rank == 0:
        if is_distributed and world_size > 1:
            final_output = os.path.join(args.output_dir, f"{args.target_language}_queries.tsv")
            with open(final_output, "w", encoding="utf-8") as outfile:
                for r in range(world_size):
                    rank_file = os.path.join(args.output_dir, f"{args.target_language}_queries_rank{r}.tsv")
                    if os.path.exists(rank_file):
                        with open(rank_file, "r", encoding="utf-8") as infile:
                            outfile.write(infile.read())
                        os.remove(rank_file)
            print(f"✓ Merged translations saved to {final_output}")
        else:
            print(f"✓ Translations saved to {output_path}")
    
    # Cleanup distributed process group
    if is_distributed:
        dist.destroy_process_group()


def main() -> None:
    """Entrypoint used by `python -m src.preprocessing.translate`."""
    args = parse_args()
    dataset = MSMarco(file_path=args.input_file)
    translate(dataset, args)


if __name__ == "__main__":
    main()
