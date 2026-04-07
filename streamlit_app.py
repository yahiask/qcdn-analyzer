import streamlit as st
import pandas as pd
from collections import Counter
from scipy.stats import entropy
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import re

# -------------------------
# واجهة
# -------------------------
st.title("QCDN Analyzer - Semantic Model (Final Safe Version)")

# -------------------------
# Normalize
# -------------------------
def normalize(text):
    text = text.lower()

    # إزالة التشكيل
    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)

    # توحيد الحروف
    text = text.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    text = text.replace("ى","ي").replace("ة","ه")

    # إزالة الرموز
    text = re.sub(r'[^\w\s]', ' ', text)

    return text

# -------------------------
# Semantic Lexicon
# -------------------------
semantic_lexicon = {

    "K": [
        "علم","يعلم","عرف","ادرك","فهم","يرى","رأى","تبين","درا","شعر"
    ],

    "P": [
        "قال","يقول","امر","دعا","سأل","ذهب","فعل","عمل","ارسل","جاء"
    ],

    "T": [
        "اصبح","صار","نجا","هلك","تغير","حدث","وقع","تحقق","دخل","خرج"
    ],

    "V": [
        "خاف","حزن","فرح","غضب","كره","احب","ضاق"
    ],

    "Z": [
        "اذ","حين","لما","بعد","قبل","ثم","حتى"
    ]
}

fields = ["Z","V","K","P","T"]

# -------------------------
# Semantic Scoring (Safe Matching)
# -------------------------
def semantic_score(text):
    text = normalize(text)
    words_in_text = text.split()
    scores = {}

    for field, lex_words in semantic_lexicon.items():
        score = 0
        for w in lex_words:
            if w in words_in_text:
                score += 1
        scores[field] = score

    return scores

# -------------------------
# Context Analysis
# -------------------------
def analyze_text(text, window=2):
    units = [u.strip() for u in text.split("\n") if u.strip()]
    data = []

    for i in range(len(units)):
        start = max(0, i - window)
        context = " ".join(units[start:i+1])
        scores = semantic_score(context)
        scores["ctu"] = i + 1
        data.append(scores)

    return pd.DataFrame(data)

# -------------------------
# Transition Matrix
# -------------------------
def build_transition(df):
    transitions = []

    for i in range(len(df) - 1):
        current = df.iloc[i][fields].idxmax()
        nxt = df.iloc[i+1][fields].idxmax()
        transitions.append((current, nxt))

    counts = Counter(transitions)
    matrix = pd.DataFrame(0, index=fields, columns=fields)

    for (i, j), c in counts.items():
        matrix.loc[i, j] = c

    return matrix

# -------------------------
# Co-occurrence
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
# Unique Chains
# -------------------------
def chain_K_P_T_unique(df, window=4):
    chains = set()
    n = len(df)

    if n < 3:
        return 0

    for i in range(n):
        if df.iloc[i]["K"] > 0:

            for j in range(i+1, min(i+window, n)):
                if df.iloc[j]["P"] > 0:

                    for k in range(j+1, min(j+window, n)):
                        if df.iloc[k]["T"] > 0:
                            chains.add((i, j, k))
                            break

    return len(chains)

# -------------------------
# Entropy
# -------------------------
def compute_entropy(matrix):
    values = matrix.values.flatten()
    values = values[values > 0]

    if len(values) == 0:
        return 0

    probs = values / values.sum()
    return entropy(probs)

# -------------------------
# Input
# -------------------------
text_input = st.text_area("ضع النص (كل جملة أو آية في سطر):")

# -------------------------
# Run
# -------------------------
if text_input:

    df = analyze_text(text_input)

    st.subheader("CTU Data")
    st.write(df)

    # Chains
    chains = chain_K_P_T_unique(df)

    # CDD
    CDD = chains / len(df) if len(df) > 0 else 0

    st.subheader("Cognitive Dynamic Density (CDD)")
    st.write("Chains:", chains)
    st.write("CDD:", CDD)

    # Transition
    matrix = build_transition(df)

    st.subheader("Transition Matrix")
    st.write(matrix)

    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, ax=ax)
    st.pyplot(fig)

    # Co-occurrence
    co_mat = co_occurrence(df)

    st.subheader("Co-occurrence Matrix")
    st.write(co_mat)

    fig2, ax2 = plt.subplots()
    sns.heatmap(co_mat, annot=True, ax=ax2)
    st.pyplot(fig2)

    # Entropy
    H = compute_entropy(matrix)
    st.write("Entropy:", H)

    # Network
    st.subheader("Network Graph")

    G = nx.DiGraph()

    for i in matrix.index:
        for j in matrix.columns:
            if matrix.loc[i, j] > 0:
                G.add_edge(i, j, weight=matrix.loc[i, j])

    fig3, ax3 = plt.subplots()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, ax=ax3)
    st.pyplot(fig3)
