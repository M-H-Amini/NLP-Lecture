import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datasets import load_dataset
from tqdm import tqdm

def readTweetDataset(n_sample=500):
    dataset = load_dataset("SetFit/tweet_sentiment_extraction")
    N = len(dataset['train']['text'])
    df_dict = {'text': [], 'label': []}
    df_dict['label'] = dataset['train']['label']
    labels_count = np.unique(df_dict['label'], return_counts=True)
    print(labels_count)
    df_dict['text'] = dataset['train']['text']
    df = pd.DataFrame(df_dict)
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace('[^\w\s]', '')

    vocab = Counter()
    for index, row in df.iterrows():
        vocab.update(row['text'].split())
    
    df = df.groupby('label').apply(lambda x: x.sample(n=n_sample, replace=True)).reset_index(drop=True)

    X, y = [], []
    for index, row in tqdm(df.iterrows()):
        X.append(makeVocabVector(vocab, row['text']))
        y.append(row['label'])
    X = np.array(X)
    y = np.array(y)
    return X, y, df

def makeVocabVector(vocab, text):
    text_cnt = vocab.copy()
    text_cnt.clear()
    text_cnt.update(text.split())
    return [text_cnt[word] for word in sorted(vocab.keys())]

if __name__ == '__main__':
    X, y, df = readTweetDataset()
    # X = df[['cnt_0', 'cnt_1', 'cnt_2']].values
    # y = df['label'].values.astype(np.int8)
    # print(X.shape, y.shape)
    # ##  Visualization...
    # sns.set(font_scale=1.5)
    # plt.figure(figsize=(8, 6))
    # plt.scatter(X[y==0, 0], X[y==0, 2], cmap='coolwarm', s=20, label='Negative', c='red')
    # # plt.scatter(X[y==1, 0], X[y==1, 1], cmap='coolwarm', s=20, label='Neutral', c='blue')
    # plt.scatter(X[y==2, 0], X[y==2, 2], cmap='coolwarm', s=20, label='Positive', c='green')
    # plt.xlabel('Negative Count')
    # plt.ylabel('Positive Count')
    # plt.title('Negative vs Positive')
    # # plt.xlim([0, 0.3])
    # # plt.ylim([0, 0.4])
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig('tweet_visualization.png', layout='tight')
    # plt.show()
