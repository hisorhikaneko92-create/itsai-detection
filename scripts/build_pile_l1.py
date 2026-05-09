# scripts/build_pile_l1.py
from datasets import load_dataset
from rbloom import Bloom
from tqdm import tqdm
import re, xxhash, json, signal, sys
from pathlib import Path

OUT_BLOOM = Path("indexes/pile_l1_5gram.bloom")
OUT_STATE = Path("indexes/pile_l1.state.json")
EXPECTED  = 30_000_000_000   # 30 B unique 5-grams
FPR       = 1e-6              # ~75 GB filter
OUT_BLOOM.parent.mkdir(exist_ok=True)

state = json.loads(OUT_STATE.read_text()) if OUT_STATE.exists() else {"docs": 0}
bf = Bloom.load(str(OUT_BLOOM)) if OUT_BLOOM.exists() else Bloom(EXPECTED, FPR)

def normalize(t): return re.sub(r'[^a-z0-9 ]', ' ', t.lower()).split()
def grams5(w):
    for i in range(len(w) - 4):
        yield xxhash.xxh3_64_intdigest(' '.join(w[i:i+5]).encode())

def save(*_):
    bf.save(str(OUT_BLOOM)); OUT_STATE.write_text(json.dumps(state)); sys.exit(0)
signal.signal(signal.SIGINT, save)

ds = load_dataset("monology/pile-uncopyrighted", streaming=True)['train'].skip(state["docs"])
for doc in tqdm(ds, initial=state["docs"]):
    for h in grams5(normalize(doc['text'])): bf.add(h)
    state["docs"] += 1
    if state["docs"] % 100_000 == 0:
        bf.save(str(OUT_BLOOM)); OUT_STATE.write_text(json.dumps(state))
bf.save(str(OUT_BLOOM))
