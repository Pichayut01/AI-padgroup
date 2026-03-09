import pandas as pd
from transformers import AutoTokenizer

print("Loading train.csv...")
df_train = pd.read_csv('train.csv')
print("Columns:", df_train.columns.tolist())
star_col = df_train.columns[1]
print("Unique stars:", df_train[star_col].unique())
print("Star rating counts:")
print(df_train[star_col].value_counts())

def map_stars_to_label(star):
    if pd.isna(star): return -1
    try:
        star = int(star)
    except:
        return -1
    if star <= 2: return 0  # 0 = Negative
    elif star == 3: return 1 # 1 = Neutral
    else: return 2          # 2 = Positive

df_train['label'] = df_train[star_col].apply(map_stars_to_label)
print("Label assignments:")
print(df_train['label'].value_counts())

text_col = df_train.columns[0]
print("Number of missing texts:", df_train[text_col].isna().sum())

print("Loading tokenizer to test...")
tokenizer = AutoTokenizer.from_pretrained("airesearch/wangchanberta-base-att-spm-uncased", use_fast=False)
sample_text = df_train[text_col].dropna().iloc[0]
out = tokenizer(sample_text)
print("Tokenized sample length:", len(out['input_ids']))
