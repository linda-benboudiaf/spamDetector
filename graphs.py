import main

from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def wordcloud():

    spam_word_freq = main.spam.copy()
    for column in spam_word_freq.columns[48:]:
        spam_word_freq.drop(column, axis=1, inplace=True)

    spam_avgs = {column.replace("word_freq_", ""): spam_word_freq[column].mean() for column in spam_word_freq}

    non_spam_word_freq = main.non_spam.copy()
    for column in non_spam_word_freq.columns[48:]:
        non_spam_word_freq.drop(column, axis=1, inplace=True)

    non_spam_avgs = {column.replace("word_freq_", ""): non_spam_word_freq[column].mean() for column in non_spam_word_freq}

    spam_cloud = WordCloud(width=480, height=480, margin=0)
    spam_cloud.generate_from_frequencies(spam_avgs)
    plt.subplot(1, 2, 1)
    plt.imshow(spam_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Spam")
    plt.margins(x=0, y=0)

    non_spam_cloud = WordCloud(width=480, height=480, margin=0)
    non_spam_cloud.generate_from_frequencies(non_spam_avgs)
    plt.subplot(1, 2, 2)
    plt.imshow(non_spam_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("Not Spam")
    plt.margins(x=0, y=0)

    plt.show()


def capital_runs():

    df = main.data
    plt.subplot(131)
    sns.boxplot(x=df["is_spam"], y=df["capital_run_length_average"], showfliers=False, linewidth=5)
    plt.subplot(132)
    sns.boxplot(x=df["is_spam"], y=df["capital_run_length_longest"], showfliers=False, linewidth=5)
    plt.subplot(133)
    sns.boxplot(x=df["is_spam"], y=df["capital_run_length_total"], showfliers=False, linewidth=5)
    plt.show()


def matrixes(matrix):
    for key in matrix:
       print_confusion_matrix(key, matrix[key])
    #plt.show()


def print_confusion_matrix(title, confusion_matrix, figsize=(10, 7), fontsize=14):
    df_cm = pd.DataFrame(confusion_matrix)
    fig = plt.figure(figsize=figsize)
    heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False)
    heatmap.yaxis.set_ticklabels(['False', "True"], rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(['False', "True"], rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title)
