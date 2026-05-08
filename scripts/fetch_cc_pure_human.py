"""
Harvest pure_human Common Crawl rows using the validator's CCDataset.
Append-safe: re-running adds to existing output and dedups against it.
"""
import csv, hashlib, json, sys, argparse, random
from pathlib import Path

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

from detection.validator.cc_dataset import CCDataset, get_2023_dumps

ap = argparse.ArgumentParser()
ap.add_argument('--output', default='data/MainData/by_source/_new_pure_human_cc_9k.csv')
ap.add_argument('--target', type=int, default=9000,
                help='STOP when total rows in output >= target (across reruns)')
ap.add_argument('--seed',   type=int, default=42)
ap.add_argument('--num-segments', type=int, default=1)
ap.add_argument('--cache-dir', default='cc_net/cache')
ap.add_argument('--tmp-dir',   default='cc_net/tmp_segments')
args = ap.parse_args()

random.seed(args.seed)
MIN_WORDS, MAX_WORDS = 35, 350
OUTPUT = Path(args.output); OUTPUT.parent.mkdir(parents=True, exist_ok=True)
Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
Path(args.tmp_dir).mkdir(parents=True, exist_ok=True)

# Dedup set: (1) hashes of existing output rows, (2) external hash file
def text_hash(t):
    return hashlib.md5((t or '').encode('utf-8')).hexdigest()

def build_dedup():
    seen = set()
    extra = Path('data/MainData/by_source/_dedup_hashes.txt')
    if extra.exists():
        for line in open(extra):
            seen.add(line.strip())
        print(f'  loaded {len(seen):,} hashes from dedup file')
    if OUTPUT.exists():
        n_existing = 0
        with open(OUTPUT, 'r', encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                seen.add(text_hash(r.get('text') or ''))
                n_existing += 1
        print(f'  loaded {n_existing:,} hashes from existing output')
    return seen

seen = build_dedup()

# Count how many we already have toward target
existing_in_output = 0
if OUTPUT.exists():
    with open(OUTPUT, 'r', encoding='utf-8', newline='') as f:
        existing_in_output = sum(1 for _ in csv.DictReader(f))
print(f'Existing output rows: {existing_in_output:,} / target {args.target:,}')

if existing_in_output >= args.target:
    print('Already at target. Nothing to do.')
    sys.exit(0)

print(f'Init CCDataset (num_segments={args.num_segments})...')
dataset = CCDataset(
    dumps=get_2023_dumps(),
    num_segments=args.num_segments,
    lang_model=Path('cc_net/bin/lid.bin'),
    lm_dir=Path('cc_net/data/lm_sp/'),
    lang_whitelist=['en'],
    lang_threshold=0.5,
    min_len=300,
    cache_dir=Path(args.cache_dir),
    tmp_dir=Path(args.tmp_dir),
)

fields = ['text', 'segmentation_labels', 'data_source',
          'sample_type', 'model_name', 'n_words', 'augmented']

new_file = not OUTPUT.exists()
n_new = n_seen = n_dup = n_short = n_trunc = 0
with open(OUTPUT, 'a', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    if new_file:
        writer.writeheader()
    for doc in dataset:
        n_seen += 1
        text = (doc.get('raw_content') or '').strip()
        if not text: continue
        words = text.split()
        if len(words) < MIN_WORDS: n_short += 1; continue
        if len(words) > MAX_WORDS:
            start = random.randint(0, len(words) - MAX_WORDS)
            words = words[start:start + MAX_WORDS]
            n_trunc += 1
        new_text = ' '.join(words)
        h = text_hash(new_text)
        if h in seen: n_dup += 1; continue
        seen.add(h)
        writer.writerow({
            'text': new_text,
            'segmentation_labels': json.dumps([0] * len(words)),
            'data_source': 'common_crawl',
            'sample_type': 'pure_human',
            'model_name': 'none',
            'n_words': str(len(words)),
            'augmented': 'cc_dump_pure_human',
        })
        n_new += 1
        f.flush()  # so we don't lose data if killed mid-iteration
        total = existing_in_output + n_new
        if n_new % 50 == 0:
            print(f'  +{n_new:,} new ({n_seen:,} seen, {n_dup:,} dups, '
                  f'{n_short:,} short, {n_trunc:,} truncated). '
                  f'Total: {total:,}/{args.target:,}')
        if total >= args.target: break

total = existing_in_output + n_new
print(f'\nDone. Added {n_new:,} new rows. Total: {total:,}/{args.target:,}')
print(f'  Streamed: {n_seen:,}, Dups: {n_dup:,}, Short: {n_short:,}, Truncated: {n_trunc:,}')
