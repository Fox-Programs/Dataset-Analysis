import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GPU Market Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        border-left: 4px solid #7c3aed;
        color: white;
    }
    h1 { color: #a78bfa; }
    h2 { color: #c4b5fd; border-bottom: 1px solid #3f3f5a; padding-bottom: 0.3rem; }
    h3 { color: #ddd6fe; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("GPU Market Analysis")
    st.markdown("---")
    st.markdown("**Naviguez par section :**")
    st.markdown("- Données brutes\n- EDA\n- Modélisation")
    st.markdown("---")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("Analyse du Marché des GPU")
st.markdown(
    "> Le marché des GPU a connu de véritables montagnes russes au niveau des prix, "
    "particulièrement avec l'essor du minage de cryptomonnaies et les perturbations "
    "de la chaîne d'approvisionnement mondiale. Cette application explore l'historique "
    "des prix des GPU **NVIDIA** et **AMD**."
)
st.markdown("---")


# ── Load & Process Data ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Chargement simulé (remplace par tes fichiers)
    price_data = pd.read_csv('gpu_price_history.csv')
    metadata = pd.read_csv('gpu_metadata.csv')

    price_data['Date'] = pd.to_datetime(price_data['Date'], errors='coerce')
    price_data.dropna(subset=['Date'], inplace=True)

    merged = pd.merge(price_data, metadata, on='Name', how='inner')

    # 1. Dérivation de nouvelle colonne & assign (0.5 + 0.5)
    # On calcule la marge de prix (différence entre prix retail et occasion)
    merged = merged.assign(Price_Diff=merged['Retail Price'] - merged['Used Price'])

    # 2. apply ou map (0.5)
    # On crée une catégorie de performance basée sur le score 3DMARK
    def performance_category(score):
        if score > 15000: return "Enthusiast"
        if score > 8000:  return "High-End"
        return "Budget"

    merged['Tier'] = merged['3DMARK'].apply(performance_category)

    return price_data, metadata, merged


price_data, metadata, merged_data = load_data()

# ── Section 1 – Raw Data ──────────────────────────────────────────────────────
with st.expander("📂 Données brutes", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Historique des prix")
        st.dataframe(price_data.head(20), use_container_width=True)
    with col2:
        st.subheader("Métadonnées")
        st.dataframe(metadata.head(20), use_container_width=True)

# ── KPI Metrics ───────────────────────────────────────────────────────────────
st.subheader("Indicateurs clés")
k1, k2, k3, k4 = st.columns(4)
k1.metric("GPU référencés",    f"{merged_data['Name'].nunique()}")
k2.metric("Prix retail moyen", f"${merged_data['Retail Price'].mean():,.0f}")
k3.metric("Prix occasion moyen",f"${merged_data['Used Price'].mean():,.0f}")
k4.metric("Décote moyenne",
          f"{((1 - merged_data['Used Price'].mean() / merged_data['Retail Price'].mean()) * 100):.1f}%")

st.markdown("---")

# ── Section 2 – EDA ───────────────────────────────────────────────────────────
st.header("Analyse Exploratoire des Données")

tab1, tab2, tab3 = st.tabs(["Distribution des prix", "Corrélation", "Prix par marque"])

with tab1:
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1a1a2e')
    sns.histplot(merged_data['Retail Price'], kde=True, color='#7c3aed',
                 label='Prix retail', ax=ax, alpha=0.7)
    sns.histplot(merged_data['Used Price'],   kde=True, color='#f97316',
                 label='Prix occasion', ax=ax, alpha=0.7)
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    ax.set_title('Distribution des prix retail vs occasion', color='white', fontsize=14)
    ax.set_xlabel('Prix ($)', color='#aaa')
    ax.set_ylabel('Fréquence', color='#aaa')
    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    numeric_df = merged_data.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1a1a2e')
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax,
                annot_kws={"size": 9}, linewidths=0.5, linecolor='#222')
    ax.set_title('Heatmap de corrélation', color='white', fontsize=14)
    ax.tick_params(colors='#ccc')
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    if 'Brand' in merged_data.columns:
        brand_avg = merged_data.groupby('Brand')[['Retail Price', 'Used Price']].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1a1a2e')
        x = np.arange(len(brand_avg))
        bars1 = ax.bar(x - 0.2, brand_avg['Retail Price'], 0.4, label='Retail', color='#7c3aed')
        bars2 = ax.bar(x + 0.2, brand_avg['Used Price'],   0.4, label='Occasion', color='#f97316')
        ax.set_xticks(x)
        ax.set_xticklabels(brand_avg['Brand'], color='#ccc')
        ax.tick_params(colors='#aaa')
        ax.set_title('Prix moyen par marque', color='white', fontsize=14)
        ax.legend(facecolor='#1a1a2e', labelcolor='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("Colonne 'Brand' non disponible dans les métadonnées.")

st.markdown("---")

# ── Section 3 – Predictive Model ─────────────────────────────────────────────
st.header("Modélisation Prédictive")
st.markdown("Prédiction du **prix d'occasion** à partir du prix retail et du score 3DMark.")

merged_model = merged_data.dropna(subset=['Retail Price', '3DMARK', 'Used Price'])
X = merged_model[['Retail Price', '3DMARK']]
y = merged_model['Used Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)
rmse = np.sqrt(mse)

m1, m2, m3 = st.columns(3)
m1.metric("R² Score",  f"{r2:.4f}",   help="Plus proche de 1 = meilleur modèle")
m2.metric("MSE",       f"{mse:,.2f}", help="Erreur quadratique moyenne")
m3.metric("RMSE",      f"${rmse:,.2f}", help="Racine de l'erreur quadratique moyenne")

# Scatter: actual vs predicted
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('#0e1117')
ax.set_facecolor('#1a1a2e')
ax.scatter(y_test, y_pred, alpha=0.6, color='#7c3aed', edgecolors='#a78bfa', linewidths=0.5)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Idéal')
ax.set_xlabel('Prix réel ($)', color='#aaa')
ax.set_ylabel('Prix prédit ($)', color='#aaa')
ax.set_title('Valeurs réelles vs prédites', color='white', fontsize=14)
ax.tick_params(colors='#aaa')
ax.legend(facecolor='#1a1a2e', labelcolor='white')
for spine in ax.spines.values():
    spine.set_edgecolor('#333')
st.pyplot(fig)
plt.close(fig)

st.markdown("---")

# ── Statistiques Avancées ──
st.header("Statistiques Avancées")
col_stats1, col_stats2 = st.columns(2)

with col_stats1:
    st.subheader("Répartition par Tier (value_counts)")
    # 3. value_counts (0.5)
    tier_counts = merged_data['Tier'].value_counts()
    st.bar_chart(tier_counts)

with col_stats2:
    st.subheader("Analyse de la Volatilité (std & sum)")
    # 4. groupby avec agrégations multiples (sum, std) (1.0)
    # On calcule la somme des prix (volume du marché) et l'écart-type (volatilité)
    stats_df = merged_data.groupby('Brandgit ').agg({
        'Retail Price': ['mean', 'std'],
        'Used Price': 'sum'
    }).reset_index()

    # On renomme pour la clarté
    stats_df.columns = ['Marque', 'Prix Moyen', 'Écart-type (Volatilité)', 'Volume Occasion Total']
    st.dataframe(stats_df, use_container_width=True)

st.markdown("---")




