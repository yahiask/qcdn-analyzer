import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import chi2_contingency, entropy
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import re

# -------------------------
# Title
# -------------------------
st.title("QCDN Analyzer - Qur'anic Cognitive Network")

# -------------------------
# Lexicon (قابل للتوسيع)
# -------------------------
Z_words = ["إذ","ثم","لما","حين","بعد","قبل","يوم"]
V_words = ["خير","شر","حب","بغض","حزن","فرح","حسد","غضب"]
K_words = ["علم","يعلم","رأى","يرى","رؤيا","تأويل","يعقل"]
P_words = ["قال","أمر","فعل","ذهب","جاء","أرسل","دخل"]
T_words = ["فاز","نجا","خسر","أصبح","عاد","كان"]

lexicon = {
    "Z": Z_words,
    "V": V_words,
    "K": K_words,
    "P": P_words,
    "T": T_words
}

# -------------------------
# Functions
# -------------------------
def score_text(text):
    scores = {}
    for field, words in lexicon.items():
        scores[field] = sum(1 for w in words if w in text)
    return scores

def analyze_text(text):
    verses = [v.strip() for v in text.split("\n") if v.strip()]
    
    data = []
    for i, v in enumerate(verses):
        scores = score_text(v)
        scores["ctu"] = i+1
        data.append(scores)

    df = pd.DataFrame(data)
    return df

def compute_entropy(matrix):
    probs = matrix.values.flatten()
    probs = probs[probs > 0]
    probs = probs / probs.sum()
    return entropy(probs)

def build_transition(df):
    fields = ["Z","V","K","P","T"]
    transitions = []

    for i in range(len(df)-1):
        current = df.iloc[i][fields].idxmax()
        nxt = df.iloc[i+1][fields].idxmax()
        transitions.append((current, nxt))

    counts = Counter(transitions)
    matrix = pd.DataFrame(0, index=fields, columns=fields)

    for (i,j), c in counts.items():
        matrix.loc[i,j] = c

    return matrix

# -------------------------
# Input
# -------------------------
text_input = st.text_area("Paste Qur'anic text here:")

if text_input:

    df = analyze_text(text_input)

    st.subheader("CTU Data")
    st.write(df)

    # Stats
    corr_KP = df["K"].corr(df["P"])
    corr_KT = df["K"].corr(df["T"])

    st.subheader("Statistics")
    st.write("Correlation K-P:", corr_KP)
    st.write("Correlation K-T:", corr_KT)

    # Transition
    matrix = build_transition(df)

    st.subheader("Transition Matrix")
    st.write(matrix)

    # Heatmap
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, ax=ax)
    st.pyplot(fig)

    # Entropy
    H = compute_entropy(matrix)
    st.write("Entropy:", H)

    # Graph
    G = nx.DiGraph()
    for i in matrix.index:
        for j in matrix.columns:
            if matrix.loc[i,j] > 0:
                G.add_edge(i, j, weight=matrix.loc[i,j])

    fig2, ax2 = plt.subplots()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, ax=ax2)
    st.pyplot(fig2)
