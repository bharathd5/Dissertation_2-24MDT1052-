from pipeline_data import load_and_prepare
from config import DATA_FILES

df, feat_cols = load_and_prepare(DATA_FILES['3min'])
print('Features:', len(feat_cols))
print('Rows:', len(df))
print(df.label_primary.value_counts())