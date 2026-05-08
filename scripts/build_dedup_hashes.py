import csv, hashlib, sys
csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

src  = 'data/MainData/by_source/train_pile_with_adv.csv'
dest = 'data/MainData/by_source/_dedup_hashes.txt'

n = 0
with open(src, 'r', encoding='utf-8', newline='') as f, \
     open(dest, 'w', encoding='utf-8') as out:
    for r in csv.DictReader(f):
        h = hashlib.md5((r.get('text') or '').encode('utf-8')).hexdigest()
        out.write(h + '\n')
        n += 1
print(f'Wrote {n:,} hashes to {dest}')
