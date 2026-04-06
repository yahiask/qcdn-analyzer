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
st.title("QCDN Analyzer - Final Stable Model")

# -------------------------
# Lexicon
# -------------------------
lexicon = {
    "Z": {"اذ":1, "حين":1, "لما":1},
    "V": {"حزن":1, "كيد":2, "حب":1, "ظلم":2},
    "K": {
        "علم":1, "يعلم":1, "تعلم":1,
        "عرف":1, "يعرف":1,
        "رأى":2, "يرى":2,
        "رؤيا":3, "تأويل":3
    },
    "P": {"قال":1, "امر":1, "جاء":1, "ارسل":1},
    "T": {
        "نجا":2, "نجي":2,
        "ملك":2, "الملك":2,
        "سجن":2, "السجن":2,
        "خرج":2, "دخل":2,
        "مكن":3, "مكنا":3,
        "رفع":2,
        "بدل":2,
        "جاء":1
    }
}

fields = ["Z","V","K","P","T"]

# -------------------------
# Normalize
# -------------------------
def normalize(text):
    text = text.lower()
    text = re.sub(r'[ًٌٍَُِّْـ]', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.replace("أ","ا").replace("إ","ا").replace("آ","ا")
    text = text.replace("ى","ي").replace("ة","ه")
    return text

# -------------------------
# Score
# -------------------------
def score_text(text):
    text = normalize(text)
    scores = {}

    for field, words in lexicon.items():
        score = 0
        for w, weight in words.items():
            if w in text:
                score += weight
        scores[field] = score

    return scores

# -------------------------
# Context Analysis
# -------------------------
def analyze_text(text, window=2):
    verses = [v.strip() for v in text.split("\n") if v.strip()]
    data = []

    for i in range(len(verses)):
        start = max(0, i - window)
        context = " ".join(verses[start:i+1])
        scores = score_text(context)
        scores["ctu"] = i + 1
        data.append(scores)

    return pd.DataFrame(data)

# -------------------------
# Transition
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
# 🔥 Unique Chains (Fixed)
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
text_input = st.text_area("ضع النص هنا (كل آية في سطر):")

# -------------------------
# Run
# -------------------------
if text_input:

    df = analyze_text(text_input)

    st.subheader("CTU Data")
    st.write(df)

    # -------------------------
    # Chains (Unique)
    # -------------------------
    chains = chain_K_P_T_unique(df)

    # -------------------------
    # CDD (Normalized)
    # -------------------------
    if len(df) > 0:
        CDD = chains / len(df)
    else:
        CDD = 0

    st.subheader("Cognitive Dynamic Density (CDD)")
    st.write("Chains:", chains)
    st.write("CDD:", CDD)

    # -------------------------
    # Statistics
    # -------------------------
    st.subheader("Statistics")

    if len(df) > 1:
        st.write("Correlation K-P:", df["K"].corr(df["P"]))
        st.write("Correlation K-T:", df["K"].corr(df["T"]))

    # -------------------------
    # Transition
    # -------------------------
    matrix = build_transition(df)

    st.subheader("Transition Matrix")
    st.write(matrix)

    fig, ax = plt.subplots()
    sns.heatmap(matrix, annot=True, ax=ax)
    st.pyplot(fig)

    # -------------------------
    # Co-occurrence
    # -------------------------
    co_mat = co_occurrence(df)

    st.subheader("Co-occurrence Matrix")
    st.write(co_mat)

    fig2, ax2 = plt.subplots()
    sns.heatmap(co_mat, annot=True, ax=ax2)
    st.pyplot(fig2)

    # -------------------------
    # Entropy
    # -------------------------
    H = compute_entropy(matrix)
    st.write("Entropy:", H)

    # -------------------------
    # Network
    # -------------------------
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
