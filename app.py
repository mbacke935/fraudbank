import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

FICHIER_DONNEES = "scenario1_propre.csv"

# ----------------------------------------------------------------- palette
SURFACE = "#ffffff"
INK = "#161221"
INK2 = "#453f5c"
MUTED = "#8d87a3"
GRID = "#eee9fb"
BASELINE = "#d9d0f0"
ACCENT = "#7c3aed"
ACCENT_CLAIR = "#ede4fd"
ACCENT_FONCE = "#4c1d95"
ROSE = "#ec4899"
STATUT = {"Normal": "#10b981", "Suspect": "#f59e0b", "Fraude": "#f43f5e"}
SEQUENTIEL = ["#f3ecfd", "#dcc4fa", "#c199f5", "#a06aef", "#7c3aed", "#5b21b6", "#3b0764"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "text.color": INK2,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


def fmt(n):
    return f"{n:,.0f}".replace(",", " ")


def _base(figsize):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color(BASELINE)
        ax.spines[cote].set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=9)
    ax.set_axisbelow(True)
    return fig, ax


# ----------------------------------------------------------------- graphiques
def fig_target(df):
    ordre = [c for c in ("Normal", "Suspect", "Fraude") if c in df["Target"].unique()]
    valeurs = df["Target"].value_counts().reindex(ordre)
    fig, ax = _base((6.4, 3.6))
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.spines["left"].set_visible(False)
    ax.bar(range(len(ordre)), valeurs, width=0.5,
           color=[STATUT[c] for c in ordre], zorder=2)
    for i, v in enumerate(valeurs):
        ax.annotate(fmt(v), (i, v), ha="center", va="bottom",
                    fontsize=10, fontweight=500, color=INK2, xytext=(0, 4),
                    textcoords="offset points")
    ax.set_xticks(range(len(ordre)))
    ax.set_xticklabels(ordre, color=INK2, fontsize=10)
    ax.set_ylim(0, valeurs.max() * 1.15)
    fig.tight_layout()
    return fig


def fig_villes(df):
    villes = df["Localisation"].value_counts().head(10).sort_values()
    fig, ax = _base((6.4, 3.9))
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.spines["bottom"].set_visible(False)
    ax.barh(villes.index, villes.values, height=0.55, color=ACCENT, zorder=2)
    for i, v in enumerate(villes.values):
        ax.annotate(fmt(v), (v, i), va="center", fontsize=9,
                    color=INK2, xytext=(5, 0), textcoords="offset points")
    ax.tick_params(axis="y", labelcolor=INK2, labelsize=9.5)
    ax.set_xticks([])
    ax.set_xlim(0, villes.max() * 1.12)
    fig.tight_layout()
    return fig


def fig_boxplot(df):
    types_tx = sorted(df["Type de transaction"].astype(str).unique())
    donnees = [df.loc[(df["Type de transaction"] == t) & (df["Montant"] > 0), "Montant"]
               for t in types_tx]
    fig, ax = _base((9.6, 3.8))
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_yscale("log")
    ax.boxplot(donnees, widths=0.45, patch_artist=True,
               boxprops=dict(facecolor=ACCENT_CLAIR, edgecolor=ACCENT, linewidth=1.1),
               medianprops=dict(color=ACCENT_FONCE, linewidth=1.8),
               whiskerprops=dict(color=BASELINE, linewidth=1),
               capprops=dict(color=BASELINE, linewidth=1),
               flierprops=dict(marker="o", markersize=3,
                               markerfacecolor=MUTED, markeredgecolor="none",
                               alpha=0.45))
    ax.set_xticklabels(types_tx, color=INK2, fontsize=10)
    ax.yaxis.set_major_formatter(lambda x, _: fmt(x))
    ax.yaxis.set_minor_formatter(lambda x, _: "")
    ax.set_ylabel("Montant (FCFA) — échelle log", fontsize=9.5)
    fig.tight_layout()
    return fig


def fig_confusion(cm, classes):
    cmap = LinearSegmentedColormap.from_list("bleu", SEQUENTIEL)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=SURFACE)
    ax.imshow(cm, cmap=cmap)
    seuil = cm.max() * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            couleur = "#ffffff" if cm[i, j] > seuil else INK
            ax.text(j, i, fmt(cm[i, j]), ha="center", va="center",
                    fontsize=11, color=couleur)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, color=INK2, fontsize=9.5)
    ax.set_yticklabels(classes, color=INK2, fontsize=9.5)
    ax.set_xlabel("Prédiction", fontsize=9.5, color=INK2)
    ax.set_ylabel("Vraie valeur", fontsize=9.5, color=INK2)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def fig_importance(modele, features):
    imp = pd.Series(modele.feature_importances_, index=features).sort_values()
    noms = {"Montant": "Montant", "Heure": "Heure", "Jour_semaine": "Jour de la semaine",
            "Mois": "Mois", "Localisation_enc": "Localisation",
            "Type de transaction_enc": "Type de transaction"}
    fig, ax = _base((5.6, 3.6))
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.spines["bottom"].set_visible(False)
    ax.barh([noms.get(f, f) for f in imp.index], imp.values, height=0.5,
            color=ACCENT, zorder=2)
    for i, v in enumerate(imp.values):
        ax.annotate(f"{v:.3f}", (v, i), va="center", fontsize=9,
                    color=INK2, xytext=(5, 0), textcoords="offset points")
    ax.set_xticks([])
    ax.set_xlim(0, imp.max() * 1.18)
    fig.tight_layout()
    return fig


def fig_probas(probas, classes):
    fig, ax = _base((6.0, 2.2))
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.spines["bottom"].set_visible(False)
    couleurs = [STATUT.get(c, ACCENT) for c in classes]
    ax.barh(classes, probas, height=0.5, color=couleurs, zorder=2)
    for i, v in enumerate(probas):
        ax.annotate(f"{v:.0%}", (v, i), va="center", fontsize=10,
                    fontweight=500, color=INK2, xytext=(6, 0),
                    textcoords="offset points")
    ax.tick_params(axis="y", labelcolor=INK2, labelsize=10)
    ax.set_xticks([])
    ax.set_xlim(0, 1.12)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- données et modèle
@st.cache_data
def charger_donnees():
    df = pd.read_csv(FICHIER_DONNEES)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


@st.cache_resource
def entrainer_modele(df):
    df_ml = df.copy()
    df_ml["Heure"] = df_ml["Date"].dt.hour
    df_ml["Jour_semaine"] = df_ml["Date"].dt.dayofweek
    df_ml["Mois"] = df_ml["Date"].dt.month

    encodeurs = {}
    for col in ["Localisation", "Type de transaction"]:
        le = LabelEncoder()
        df_ml[col + "_enc"] = le.fit_transform(df_ml[col].astype(str))
        encodeurs[col] = le

    le_target = LabelEncoder()
    df_ml["Target_enc"] = le_target.fit_transform(df_ml["Target"].astype(str))

    features = ["Montant", "Heure", "Jour_semaine", "Mois",
                "Localisation_enc", "Type de transaction_enc"]
    X = df_ml[features]
    y = df_ml["Target_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    modele = RandomForestClassifier(n_estimators=200, random_state=42,
                                    class_weight="balanced")
    modele.fit(X_train, y_train)
    y_pred = modele.predict(X_test)

    metriques = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "matrice": confusion_matrix(y_test, y_pred),
        "rapport": classification_report(y_test, y_pred,
                                         target_names=le_target.classes_),
        "features": features,
    }
    return modele, encodeurs, le_target, metriques


# ----------------------------------------------------------------- style
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="st-"], .stMarkdown, button, input, textarea {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

.stApp {
    background:
        radial-gradient(1100px circle at 8% -8%, #f1e6fd 0%, transparent 45%),
        radial-gradient(900px circle at 100% 0%, #fde3f0 0%, transparent 42%),
        radial-gradient(800px circle at 50% 100%, #fef6e0 0%, transparent 40%),
        #fbfaff;
}

.block-container { padding-top: 2.2rem; max-width: 1180px; }

.hero { margin-bottom: 0.6rem; }
.hero h1 { font-family: 'Poppins', sans-serif; font-size: 2rem; font-weight: 700;
           letter-spacing: -0.01em; margin: 0;
           background: linear-gradient(90deg, #7c3aed 0%, #ec4899 60%, #f59e0b 100%);
           -webkit-background-clip: text; background-clip: text; color: transparent; }
.hero p { color: #453f5c; margin: 0.35rem 0 0 0; font-size: 1rem; }
.hero .puce { display: inline-block; background: linear-gradient(90deg,#7c3aed,#ec4899);
              color: #ffffff; border-radius: 100px; padding: 3px 14px; font-size: 0.78rem;
              font-weight: 600; margin-left: 10px; vertical-align: 4px;
              box-shadow: 0 3px 10px rgba(124,58,237,0.35); }

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border: 1px solid rgba(124, 58, 237, 0.12);
    border-radius: 20px;
    box-shadow: 0 2px 4px rgba(76, 29, 149, 0.05), 0 10px 26px rgba(124, 58, 237, 0.08);
}
div[data-testid="stVerticalBlockBorderWrapper"] > div > div[data-testid="stVerticalBlock"] {
    padding: 0.4rem 0.5rem;
}
.card-titre { font-family: 'Poppins', sans-serif; font-size: 1rem; font-weight: 600;
              color: #161221; margin: 0 0 0.1rem 0.2rem; }
.card-sous-titre { font-size: 0.82rem; color: #8d87a3; margin: -2px 0 0.4rem 0.2rem; }

[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid rgba(124, 58, 237, 0.10);
    border-top: 4px solid #7c3aed;
    border-radius: 18px;
    padding: 14px 20px 12px 20px;
    box-shadow: 0 2px 4px rgba(76, 29, 149, 0.05), 0 10px 22px rgba(124, 58, 237, 0.08);
}
div[data-testid="column"]:nth-of-type(1) [data-testid="stMetric"] { border-top-color: #7c3aed; }
div[data-testid="column"]:nth-of-type(2) [data-testid="stMetric"] { border-top-color: #ec4899; }
div[data-testid="column"]:nth-of-type(3) [data-testid="stMetric"] { border-top-color: #f59e0b; }
div[data-testid="column"]:nth-of-type(4) [data-testid="stMetric"] { border-top-color: #10b981; }
[data-testid="stMetricLabel"] p { color: #453f5c; font-size: 0.85rem; font-weight: 500; }
[data-testid="stMetricValue"] { color: #161221; font-weight: 700; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; border-bottom: none; background: #f3ecfd;
                                     border-radius: 100px; padding: 6px; display: inline-flex; }
.stTabs [data-baseweb="tab"] { height: 42px; padding: 0 22px; color: #453f5c;
                               font-weight: 600; background: transparent; border-radius: 100px; }
.stTabs [aria-selected="true"] { color: #ffffff !important;
                                  background: linear-gradient(90deg,#7c3aed,#ec4899);
                                  box-shadow: 0 3px 10px rgba(124,58,237,0.35); }
.stTabs [data-baseweb="tab-highlight"] { background-color: transparent; }
.stTabs [data-baseweb="tab-border"] { background: transparent; }

.stFormSubmitButton button, .stButton button {
    background: linear-gradient(90deg, #7c3aed, #ec4899); color: #ffffff; border: none;
    border-radius: 100px; padding: 0.6rem 2rem; font-weight: 600;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stFormSubmitButton button:hover, .stButton button:hover {
    transform: translateY(-1px); box-shadow: 0 6px 20px rgba(124, 58, 237, 0.45); color: #ffffff;
}
[data-testid="stForm"] { border: none; padding: 0; }

.resultat { border-radius: 14px; padding: 16px 20px; font-weight: 600;
            font-size: 1.05rem; margin: 0.4rem 0 0.8rem 0; }
.resultat.normal  { background: linear-gradient(90deg,#d1fae5,#ecfdf5); color: #047857; }
.resultat.suspect { background: linear-gradient(90deg,#fef3c7,#fffbeb); color: #92400e; }
.resultat.fraude  { background: linear-gradient(90deg,#fee2e2,#fff1f2); color: #b91c1c; }
</style>
"""


def carte_titre(titre, sous_titre=None):
    st.markdown(f'<p class="card-titre">{titre}</p>', unsafe_allow_html=True)
    if sous_titre:
        st.markdown(f'<p class="card-sous-titre">{sous_titre}</p>',
                    unsafe_allow_html=True)


# ----------------------------------------------------------------- page
def main():
    st.set_page_config(page_title="Détection de fraude bancaire",
                       page_icon="🛡", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    df = charger_donnees()
    modele, encodeurs, le_target, metriques = entrainer_modele(df)

    st.markdown(
        '<div class="hero"><h1>Détection de fraude bancaire'
        '<span class="puce">Random Forest</span></h1>'
        '<p>Transactions bancaires sénégalaises — Scénario 1</p></div>',
        unsafe_allow_html=True)

    onglet_donnees, onglet_modele, onglet_prediction = st.tabs(
        ["Données", "Modèle", "Prédiction"])

    # ------------------------------------------------------------- Données
    with onglet_donnees:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions", fmt(len(df)))
        c2.metric("Fraudes", fmt((df["Target"] == "Fraude").sum()))
        c3.metric("Villes", df["Localisation"].nunique())
        c4.metric("Montant moyen", fmt(df["Montant"].mean()) + " FCFA")

        st.write("")
        gauche, droite = st.columns(2)
        with gauche, st.container(border=True):
            carte_titre("Répartition des transactions",
                        "Statut attribué à chaque opération")
            st.pyplot(fig_target(df), use_container_width=True)
        with droite, st.container(border=True):
            carte_titre("Top 10 des localisations",
                        "Nombre de transactions par ville")
            st.pyplot(fig_villes(df), use_container_width=True)

        with st.container(border=True):
            carte_titre("Montants par type de transaction",
                        "Distribution en FCFA — les points isolés sont des montants extrêmes")
            st.pyplot(fig_boxplot(df), use_container_width=True)

        with st.container(border=True):
            carte_titre("Aperçu des données", f"{fmt(len(df))} lignes après nettoyage")
            st.dataframe(df.head(100), height=280)

    # ------------------------------------------------------------- Modèle
    with onglet_modele:
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{metriques['accuracy']:.1%}")
        c2.metric("F1-macro", f"{metriques['f1_macro']:.3f}")
        c3.metric("Données de test", fmt(metriques["matrice"].sum()))

        st.write("")
        gauche, droite = st.columns(2)
        with gauche, st.container(border=True):
            carte_titre("Matrice de confusion",
                        "La diagonale contient les prédictions correctes")
            st.pyplot(fig_confusion(metriques["matrice"], le_target.classes_),
                      use_container_width=True)
        with droite, st.container(border=True):
            carte_titre("Importance des variables",
                        "Contribution de chaque variable à la décision")
            st.pyplot(fig_importance(modele, metriques["features"]),
                      use_container_width=True)

        with st.container(border=True):
            carte_titre("Rapport de classification",
                        "Precision : fiabilité des alertes — Recall : taux de détection")
            st.code(metriques["rapport"], language=None)

    # ------------------------------------------------------------- Prédiction
    with onglet_prediction:
        gauche, droite = st.columns([5, 4])

        with gauche, st.container(border=True):
            carte_titre("Simuler une transaction",
                        "Le modèle évalue le risque de la transaction saisie")
            with st.form("formulaire"):
                montant = st.number_input("Montant (FCFA)", min_value=0.0,
                                          value=50000.0, step=1000.0)
                ville = st.selectbox("Localisation",
                                     sorted(df["Localisation"].astype(str).unique()))
                type_tx = st.selectbox("Type de transaction",
                                       sorted(df["Type de transaction"].astype(str).unique()))
                heure = st.slider("Heure", 0, 23, 12)
                jours = ["Lundi", "Mardi", "Mercredi", "Jeudi",
                         "Vendredi", "Samedi", "Dimanche"]
                jour = st.selectbox("Jour de la semaine", jours)
                mois = st.slider("Mois", 1, 12, 6)
                envoyer = st.form_submit_button("Évaluer la transaction")

        with droite:
            if envoyer:
                entree = pd.DataFrame([{
                    "Montant": montant,
                    "Heure": heure,
                    "Jour_semaine": jours.index(jour),
                    "Mois": mois,
                    "Localisation_enc": encodeurs["Localisation"].transform([ville])[0],
                    "Type de transaction_enc":
                        encodeurs["Type de transaction"].transform([type_tx])[0],
                }])
                prediction = le_target.classes_[modele.predict(entree)[0]]
                probas = modele.predict_proba(entree)[0]

                with st.container(border=True):
                    carte_titre("Résultat")
                    classe_css = prediction.lower()
                    st.markdown(
                        f'<div class="resultat {classe_css}">'
                        f'Transaction classée : {prediction}</div>',
                        unsafe_allow_html=True)
                    st.pyplot(fig_probas(probas, list(le_target.classes_)),
                              use_container_width=True)
            else:
                with st.container(border=True):
                    carte_titre("Résultat")
                    st.markdown(
                        '<p class="card-sous-titre">Renseignez la transaction '
                        'puis cliquez sur « Évaluer la transaction ».</p>',
                        unsafe_allow_html=True)


if __name__ == "__main__":
    main()
