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
st.title("QCDN Analyzer - Cognitive Qur'anic Model (Enhanced)")

# -------------------------
# Lexicon (مُحسَّن علميًا)
# -------------------------
lexicon = {
    "Z": ["اذ","حين","لما"],
    "V": ["حزن","كيد","حب","ظلم"],
    "K": {
    "علم":1, "يعلم":1, "تعلم":1, "تعليم":1,
    "عرف":1, "يعرف":1,
    "رأى":2, "يرى":2, "بصر":2,
    "رؤيا":3, "تأويل":3
}
    "P": ["قال","امر","جاء","ارسل"],
    "T": ["نجا","ملك","سجن"]
}

fields = ["Z","V","K","P","T"]

# -------------------------
# تنظيف النص
# -------------------------
def normalize(text):
    text = text.lower()
    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    text = text.replace("ى","ي").replace("ة","ه")
    return text

# -------------------------
# مطابقة مرنة
# -------------------------
def match_word(text, word):
    return word in text

# -------------------------
# حساب الحقول
# -------------------------
def score_text(text):
    text = normalize(text)
    scores = {}

    for field, words in lexicon.items():
        if isinstance(words, dict):
            scores[field] = sum(weight for w, weight in words.items() if w in text)
        else:
            scores[field] = sum(1 for w in words if w in text)

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
# مصفوفة الانتقال بين الآيات
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
# 🔥 العلاقات داخل الآية (الأهم)
# -------------------------
def co_occurrence(df):
    co_matrix = pd.DataFrame(0, index=fields, columns=fields)

    for _, row in df.iterrows():
        active = [f for f in fields if row[f] > 0]
        for i in active:
            for j in active:
                if i != j:
                    co_matrix.loc[i, j] += 1

    return co_matrix

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
text_input = st.text_area("ضع النص هنا (كل آية أو جملة في سطر):")

# -------------------------
# تشغيل التحليل
# -------------------------
if text_input:

    df = analyze_text(text_input)

    st.subheader("CTU Data")
    st.write(df)

    # -------------------------
    # إحصائيات
    # -------------------------
    st.subheader("Statistics")
    if len(df) > 1:
        st.write("Correlation K-P:", df["K"].corr(df["P"]))
        st.write("Correlation K-T:", df["K"].corr(df["T"]))

    # -------------------------
    # الانتقال بين الآيات
    # -------------------------
    matrix = build_transition(df)

    st.subheader("Transition Matrix (بين الآيات)")
    st.write(matrix)

    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, ax=ax)
    st.pyplot(fig)

    # -------------------------
    # 🔥 العلاقات داخل الآية
    # -------------------------
    co_mat = co_occurrence(df)

    st.subheader("Co-occurrence Matrix (داخل الآية)")
    st.write(co_mat)

    fig_co, ax_co = plt.subplots()
    sns.heatmap(co_mat, annot=True, ax=ax_co)
    st.pyplot(fig_co)

    # -------------------------
    # Entropy
    # -------------------------
    H = compute_entropy(matrix)
    st.write("Entropy:", H)

    # -------------------------
    # الشبكة
    # -------------------------
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
