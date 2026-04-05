import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import entropy
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import re

# -------------------------
# واجهة
# -------------------------
st.title("QCDN Analyzer - Qur'anic Cognitive Network (Fixed Version)")

# -------------------------
# Lexicon (محسن)
# -------------------------
lexicon = {
    "Z": ["اذ","حين","لما"],
    "V": ["حزن","كيد","حب","ظلم"],
    "K": ["رؤيا","تأويل","يعلم","رأى"],
    "P": ["قال","امر","جاء","ارسل"],
    "T": ["نجا","ملك","سجن"]
}

fields = ["Z","V","K","P","T"]

# -------------------------
# 🔥 1. تنظيف النص
# -------------------------
def normalize(text):
    text = text.lower()
    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)   # إزالة التشكيل
    text = re.sub(r'[^\w\s]', '', text)      # إزالة الرموز
    text = text.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    text = text.replace("ى","ي").replace("ة","ه")
    return text

# -------------------------
# 🔥 2. مطابقة مرنة
# -------------------------
def match_word(text, word):
    return word in text

# -------------------------
# 🔥 3. حساب الدرجات
# -------------------------
def score_text(text):
    text = normalize(text)

    scores = {}
    for field, words in lexicon.items():
        scores[field] = sum(1 for w in words if match_word(text, w))
    return scores

# -------------------------
# تحليل النص
# -------------------------
def analyze_text(text):
    verses = [v.strip() for v in text.split("\n") if v.strip()]

    data = []
    for i, v in enumerate(verses):
        scores = score_text(v)
        scores["ctu"] = i + 1
        data.append(scores)

    df = pd.DataFrame(data)
    return df

# -------------------------
# مصفوفة الانتقال
# -------------------------
def build_transition(df):
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
# Entropy
# -------------------------
def compute_entropy(matrix):
    probs = matrix.values.flatten()
    probs = probs[probs > 0]
    probs = probs / probs.sum()
    return entropy(probs)

# -------------------------
# إدخال النص
# -------------------------
text_input = st.text_area("ضع نص السورة هنا (كل آية في سطر):")

# -------------------------
# التشغيل
# -------------------------
if text_input:

    df = analyze_text(text_input)

    st.subheader("CTU Data")
    st.write(df)

    # إحصائيات
    st.subheader("Statistics")

    if len(df) > 1:
        st.write("Correlation K-P:", df["K"].corr(df["P"]))
        st.write("Correlation K-T:", df["K"].corr(df["T"]))

    # مصفوفة
    matrix = build_transition(df)

    st.subheader("Transition Matrix")
    st.write(matrix)

    # Heatmap
    st.subheader("Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, ax=ax)
    st.pyplot(fig)

    # Entropy
    H = compute_entropy(matrix)
    st.write("Entropy:", H)

    # Graph
    st.subheader("Network Graph")
    G = nx.DiGraph()

    for i in matrix.index:
        for j in matrix.columns:
            if matrix.loc[i,j] > 0:
                G.add_edge(i, j, weight=matrix.loc[i,j])

    fig2, ax2 = plt.subplots()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, ax=ax2)
    st.pyplot(fig2)
