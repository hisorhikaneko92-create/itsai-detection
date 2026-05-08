import csv, hashlib, sys
from pathlib import Path

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

sources = [
    'data/MainData/by_source/train_pile_with_adv.csv',
    'data/MainData/by_source/train_common_crawl_with_adv.csv',
]
dest = Path('data/MainData/by_source/_dedup_hashes.txt')

n = 0
with open(dest, 'w', encoding='utf-8') as out:
    for src in sources:
        if not Path(src).exists():
            print(f'SKIP {src}: not found'); continue
        with open(src, 'r', encoding='utf-8', newline='') as f:
            for r in csv.DictReader(f):
                h = hashlib.md5((r.get('text') or '').encode('utf-8')).hexdigest()
                out.write(h + '\n')
                n += 1
        print(f'  +{src}')
print(f'Wrote {n:,} hashes to {dest}')
