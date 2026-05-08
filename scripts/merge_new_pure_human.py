import pandas as pd, csv, sys, os, shutil
csv.field_size_limit(sys.maxsize if os.name != 'nt' else 2**31 - 1)

src  = 'data/MainData/by_source/_new_pure_human_9k.csv'
work = 'data/MainData/by_source/train_pile_with_adv.csv'
bak  = 'data/MainData/by_source/train_pile_with_adv.pre_pure_human_add.bak.csv'

new = pd.read_csv(src)
print(f'New rows fetched: {len(new):,}')
print(f'  sample_type: {sorted(new["sample_type"].unique().tolist())}')
print(f'  data_source: {sorted(new["data_source"].unique().tolist())}')

new['model_name'] = 'none'

df = pd.read_csv(work, engine='python', on_bad_lines='warn')
work_texts = set(df['text'].astype(str))
new_clean = new[~new['text'].astype(str).isin(work_texts)].copy()
print(f'After final dedup: {len(new_clean):,}')

new_clean = new_clean[df.columns.tolist()]

if not os.path.exists(bak):
    shutil.copy(work, bak)
    print(f'Backup -> {bak}')

out = pd.concat([df, new_clean], ignore_index=True)
out.to_csv(work, index=False)
print(f'Wrote {work}: {len(df):,} -> {len(out):,}  (added {len(new_clean):,})')
print()
print('=== sample_type after merge ===')
print(out['sample_type'].value_counts().to_string())
print()
print('=== data_source after merge ===')
print(out['data_source'].value_counts().to_string())
print()
print('=== model_name (top 5) ===')
print(out['model_name'].value_counts().head(5).to_string())
print()
print(f'NaN model_name: {out["model_name"].isna().sum()}')
