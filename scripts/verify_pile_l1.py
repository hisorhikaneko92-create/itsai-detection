# scripts/verify_pile_l1.py
from rbloom import Bloom
import json, glob, re, xxhash
bf = Bloom.load("indexes/pile_l1_5gram.bloom")
norm = lambda t: re.sub(r'[^a-z0-9 ]', ' ', t.lower()).split()
def hit_ratio(text, n=5):
    w = norm(text)
    if len(w) < n: return 0.0
    h = sum(1 for i in range(len(w)-n+1)
            if xxhash.xxh3_64_intdigest(' '.join(w[i:i+n]).encode()) in bf)
    return h / (len(w) - n + 1)

for fp in glob.glob('neurons/validator_logs/raw/*.json')[-50:]:
    d = json.load(open(fp))
    for t in d.get('texts', []):
        print(f"{t['hash']}  hit_ratio={hit_ratio(t['full_text']):.2f}")
