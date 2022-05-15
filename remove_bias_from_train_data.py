import pandas as pd
from gensim.models import LdaModel

df = pd.read_csv('with_topic_train_dataset_copy.csv', header=None, encoding='utf8', sep='\t',
                names=['gender', 'text', 'tokens', 'text_bow', 'topic'])

topic2cnt = {}
topics = range(0, 15)

for topic in topics:
    val_cnts = df[df.topic == topic].gender.value_counts()
    topic2cnt[topic] = min(val_cnts[0], val_cnts[1])


df_nonbiased = pd.DataFrame()

for topic in topics:
    min_text_cnt = topic2cnt[topic]
    male = df[(df.gender == 1) & (df.topic == topic)].sample(min_text_cnt)
    female = df[(df.gender == 0) & (df.topic == topic)].sample(min_text_cnt)
    df_nonbiased = pd.concat([df_nonbiased, male, female], ignore_index=True)

df_nonbiased.to_csv('nonbiased_dataset.csv', sep='\t', index=False, header=None, encoding='utf8')
