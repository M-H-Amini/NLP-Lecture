import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def readSpamDataset(csv_file='spam.csv'):
    ##  Read the csv file...
    df = pd.read_csv('spam.csv', encoding='latin-1', usecols=[0, 1], names=['label', 'text'], skiprows=1)
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace('[^\w\s]', '')

    cnt_spam, cnt_ham = Counter(), Counter()

    for index, row in df.iterrows():
        if row['label'] == 1:
            cnt_spam.update(row['text'].split())
        else:
            cnt_ham.update(row['text'].split())

    print('Most common words in spam messages:')
    print(cnt_spam.most_common(50))
    print('Most common words in ham messages:')
    print(cnt_ham.most_common(50))

    sum_spam = sum(cnt_spam.values())
    sum_ham = sum(cnt_ham.values())

    df['spam_cnt'] = df['text'].apply(lambda x: sum([cnt_spam[word] for word in x.split()]) / sum_spam)
    df['ham_cnt'] = df['text'].apply(lambda x: sum([cnt_ham[word] for word in x.split()]) / sum_ham)
    return df

if __name__ == '__main__':
    df = readSpamDataset()
    X = df[['spam_cnt', 'ham_cnt']].values
    y = df['label'].values.astype(np.int8)
    print(X.shape, y.shape)
    ##  Visualization...
    sns.set(font_scale=1.5)
    plt.figure(figsize=(8, 6))
    plt.scatter(X[y==0, 0], X[y==0, 1], cmap='coolwarm', s=20, label='Ham', c='green')
    plt.scatter(X[y==1, 0], X[y==1, 1], cmap='coolwarm', s=20, label='Spam', c='red')
    plt.xlabel('Spam Count')
    plt.ylabel('Ham Count')
    plt.title('Spam vs Ham')
    plt.xlim([0, 0.3])
    plt.ylim([0, 0.4])
    plt.legend()
    plt.tight_layout()
    plt.savefig('spam_visualization.png', layout='tight')
    plt.show()

