import argparse
import csv
import json
import os
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import nltk
from nltk.tokenize import TreebankWordDetokenizer, TreebankWordTokenizer
from tqdm import tqdm


nltk.download("punkt", quiet=True)


Token = str
Lexicon = Dict[str, List[str]]


class LexiconSwitch:
    """Applies random lexicon-based replacements to tokenized text."""

    def __init__(
        self,
        lexicon: Lexicon,
        probability: float,
        rng: random.Random,
    ) -> None:
        self.lexicon = lexicon
        self.probability = probability
        self.rng = rng
        self.tokenizer = TreebankWordTokenizer()
        self.detokenizer = TreebankWordDetokenizer()

    def __call__(self, text: str) -> Tuple[str, int]:
        """Returns the switched sentence plus the number of replacements applied."""
        tokens = self.tokenizer.tokenize(text)
        if not tokens:
            return text, 0

        switched_tokens: List[Token] = []
        switch_count = 0

        for token in tokens:
            key = token.lower()
            candidates = self.lexicon.get(key)

            if candidates and self.rng.random() < self.probability:
                replacement = self._pick_candidate(token, candidates)
                switched_tokens.extend(replacement)
                switch_count += 1
            else:
                switched_tokens.append(token)

        if switch_count == 0:
            return text, 0

        return self.detokenizer.detokenize(switched_tokens), switch_count

    def _pick_candidate(self, source_token: str, candidates: Sequence[str]) -> List[str]:
        """Samples a replacement string and tokenizes it to align with Treebank boundaries."""
        raw_replacement = self.rng.choice(candidates)
        replacement_tokens = self.tokenizer.tokenize(raw_replacement)
        if not replacement_tokens:
            return [source_token]

        first = self._match_case(replacement_tokens[0], source_token)
        rest = replacement_tokens[1:]
        return [first, *rest]

    @staticmethod
    def _match_case(replacement: str, original: str) -> str:
        if original.isupper():
            return replacement.upper()
        if original[:1].isupper():
            return replacement.capitalize()
        return replacement


def load_lexicon(path: str) -> Lexicon:
    """Loads a lexicon JSON mapping source tokens to lists of candidate translations."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Lexicon file not found: {path}")

    with open(path, "r", encoding="utf-8") as lexicon_file:
        data = json.load(lexicon_file)

    normalized: Lexicon = {}
    for key, value in data.items():
        key_lower = key.lower().strip()
        if not key_lower:
            continue
        values = value if isinstance(value, list) else [value]
        cleaned = [str(entry).strip() for entry in values if str(entry).strip()]
        if cleaned:
            normalized[key_lower] = cleaned
    return normalized


def iter_rows(path: str) -> Iterable[List[str]]:
    """Yields each TSV row as a list of columns."""
    with open(path, "r", encoding="utf-8", errors="ignore") as input_file:
        reader = csv.reader(input_file, delimiter="\t")
        for row in reader:
            if row:
                yield row


def parse_args() -> argparse.Namespace:
    """Defines the CLI surface for the code-switch job."""
    parser = argparse.ArgumentParser(description="Apply lexicon-based code switching to TSV files.")
    parser.add_argument("--input_file", required=True, type=str, help="TSV file to process.")
    parser.add_argument("--output_file", required=True, type=str, help="Destination TSV file.")
    parser.add_argument("--lexicon_name", required=True, type=str, help="Lexicon filename (e.g., eng_swh.json).")
    parser.add_argument("--lexicon_dir", default="data/lexicon", type=str, help="Lexicon directory.")
    parser.add_argument("--probability", default=0.5, type=float, help="Token switch probability.")
    parser.add_argument("--seed", default=13, type=int, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for lexicon-driven code switching."""
    args = parse_args()
    rng = random.Random(args.seed)
    lexicon_path = os.path.abspath(os.path.join(args.lexicon_dir, args.lexicon_name))
    lexicon = load_lexicon(lexicon_path)

    switcher = LexiconSwitch(lexicon=lexicon, probability=args.probability, rng=rng)

    total_rows = 0
    switched_rows = 0
    total_switches = 0

    with open(args.output_file, "w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t")
        for row in tqdm(iter_rows(args.input_file), desc="Code-switching"):
            total_rows += 1
            if len(row) < 2:
                writer.writerow(row)
                continue

            text_index = len(row) - 1
            switched_text, num_switches = switcher(row[text_index])
            row[text_index] = switched_text
            writer.writerow(row)
            
            if num_switches > 0:
                switched_rows += 1
                total_switches += num_switches

    switch_rate = (switched_rows / total_rows * 100) if total_rows else 0.0
    avg_per_switched = (total_switches / switched_rows) if switched_rows else 0.0
    print(f"{total_rows:,} queries → {switched_rows:,} code-switched queries({switch_rate:.1f}%), avg {avg_per_switched:.1f} words/query")


if __name__ == "__main__":
    main()
