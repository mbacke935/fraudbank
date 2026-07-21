# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Fichier de données par défaut
FICHIER_DONNEES = "scenario1_propre.csv"

# ----------------------------------------------------------------- Palette Light Minimal
SURFACE_BG = "#f8fafc"      # Slate 50 (Fond principal)
SURFACE_CARD = "#ffffff"    # Blanc pur (Cartes)
INK = "#0f172a"             # Slate 900 (Texte principal)
INK2 = "#475569"            # Slate 600 (Texte secondaire)
MUTED = "#94a3b8"           # Slate 400 (Éléments discrets)
GRID = "#f1f5f9"            # Slate 100 (Grilles)
BASELINE = "#cbd5e1"        # Slate 300 (Axes)

ACCENT = "#2563eb"          # Bleu Cobalt principal
ACCENT_CLAIR = "#3b82f6"    # Bleu moyen
ACCENT_BG = "#eff6ff"       # Fond bleu très doux

STATUT = {
    "Normal": "#10b981",   # Émeraude
    "Suspect": "#f59e0b",  # Ambre
    "Fraude": "#ef4444"    # Rouge
}

SEQUENTIEL = ["#eff6ff", "#dbeafe", "#bfdbfe", "#93c5fd", "#60a5fa", "#3b82f6", "#1d4ed8"]

# Configuration globale des graphiques Matplotlib
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9.5,
    "text.color": INK2,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
})


def fmt(n):
    """Formate un nombre entier avec des espaces comme séparateurs de milliers."""
    return f"{n:,.0f}".replace(",", " ")


def _base(figsize):
    """Crée une base de graphique Matplotlib harmonisée avec le thème clair."""
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE_CARD)
    ax.set_facecolor(SURFACE_CARD)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color(BASELINE)
        ax.spines[cote].set_linewidth(0.8)
    ax.tick_params(length=0, labelsize=9)
    ax.set_axisbelow(True)
    return fig, ax


# ----------------------------------------------------------------- Graphiques
def fig_target(df):
    """Graphique en barres de la répartition des statuts de transactions."""
    ordre = [c for c in ("Normal", "Suspect", "Fraude") if c in df["Target"].unique()]
    valeurs = df["Target"].value_counts().reindex(ordre)
    fig, ax = _base((6.4, 3.6))
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.spines["left"].set_visible(False)
    ax.bar(range(len(ordre)), valeurs, width=0.45,
           color=[STATUT[c] for c in ordre], zorder=2)
    for i, v in enumerate(valeurs):
        ax.annotate(fmt(v), (i, v), ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=INK, xytext=(0, 5),
                    textcoords="offset points")
    ax.set_xticks(range(len(ordre)))
    ax.set_xticklabels(ordre, color=INK, fontsize=9.5, fontweight="600")
    ax.set_ylim(0, valeurs.max() * 1.15)
    fig.tight_layout()
    return fig


def fig_villes(df):
    """Graphique horizontal du Top 10 des villes."""
    villes = df["Localisation"].value_counts().head(10).sort_values()
    fig, ax = _base((6.4, 3.9))
    ax.grid(axis="x", color=GRID, linewidth=0.8, linestyle="-")
    ax.spines["bottom"].set_visible(False)
    ax.barh(villes.index, villes.values, height=0.5, color=ACCENT_CLAIR, zorder=2)
    for i, v in enumerate(villes.values):
        ax.annotate(fmt(v), (v, i), va="center", fontsize=9,
                    color=INK, xytext=(6, 0), textcoords="offset points")
    ax.tick_params(axis="y", labelcolor=INK, labelsize=9.5)
    ax.set_xticks([])
    ax.set_xlim(0, villes.max() * 1.15)
    fig.tight_layout()
    return fig


def fig_boxplot(df):
    """Boxplot des montants par type de transaction."""
    types_tx = sorted(df["Type de transaction"].astype(str).unique())
    donnees = [df.loc[(df["Type de transaction"] == t) & (df["Montant"] > 0), "Montant"]
               for t in types_tx]
    fig, ax = _base((9.6, 3.8))
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_yscale("log")
    ax.boxplot(donnees, widths=0.4, patch_artist=True,
               boxprops=dict(facecolor=ACCENT_BG, edgecolor=ACCENT_CLAIR, linewidth=1.2),
               medianprops=dict(color=ACCENT, linewidth=2),
               whiskerprops=dict(color=BASELINE, linewidth=1),
               capprops=dict(color=BASELINE, linewidth=1),
               flierprops=dict(marker="o", markersize=3,
                               markerfacecolor=MUTED, markeredgecolor="none",
                               alpha=0.4))
    ax.set_xticklabels(types_tx, color=INK, fontsize=9.5, fontweight="500")
    ax.yaxis.set_major_formatter(lambda x, _: fmt(x))
    ax.yaxis.set_minor_formatter(lambda x, _: "")
    ax.set_ylabel("Montant (FCFA) — échelle log", fontsize=9, color=INK2)
    fig.tight_layout()
    return fig


def fig_confusion(cm, classes):
    """Matrice de confusion sous forme de Heatmap lumineuse."""
    cmap = LinearSegmentedColormap.from_list("blue_light", SEQUENTIEL)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=SURFACE_CARD)
    ax.set_facecolor(SURFACE_CARD)
    ax.imshow(cm, cmap=cmap)
    seuil = cm.max() * 0.55
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            couleur = "#ffffff" if cm[i, j] > seuil else INK
            ax.text(j, i, fmt(cm[i, j]), ha="center", va="center",
                    fontsize=11, fontweight="bold", color=couleur)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, color=INK, fontsize=9.5, fontweight="500")
    ax.set_yticklabels(classes, color=INK, fontsize=9.5, fontweight="500")
    ax.set_xlabel("Prédiction", fontsize=9.5, color=INK2, labelpad=10)
    ax.set_ylabel("Vraie valeur", fontsize=9.5, color=INK2, labelpad=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    return fig


def fig_importance(modele, features):
    """Graphique de l'importance des variables."""
    imp = pd.Series(modele.feature_importances_, index=features).sort_values()
    noms = {"Montant": "Montant", "Heure": "Heure", "Jour_semaine": "Jour de la semaine",
            "Mois": "Mois", "Localisation_enc": "Localisation",
            "Type de transaction_enc": "Type de transaction"}
    fig, ax = _base((5.6, 3.6))
    ax.grid(axis="x", color=GRID, linewidth=0.8, linestyle="-")
    ax.spines["bottom"].set_visible(False)
    ax.barh([noms.get(f, f) for f in imp.index], imp.values, height=0.45,
            color=ACCENT, zorder=2)
    for i, v in enumerate(imp.values):
        ax.annotate(f"{v:.3f}", (v, i), va="center", fontsize=9,
                    color=INK, xytext=(6, 0), textcoords="offset points")
    ax.set_xticks([])
    ax.set_xlim(0, imp.max() * 1.2)
    fig.tight_layout()
    return fig


def fig_probas(probas, classes):
    """Barres horizontales affichant les probabilités prédites."""
    fig, ax = _base((6.0, 2.2))
    ax.grid(axis="x", color=GRID, linewidth=0.8, linestyle="-")
    ax.spines["bottom"].set_visible(False)
    couleurs = [STATUT.get(c, ACCENT) for c in classes]
    ax.barh(classes, probas, height=0.45, color=couleurs, zorder=2)
    for i, v in enumerate(probas):
        ax.annotate(f"{v:.0%}", (v, i), va="center", fontsize=9.5,
                    fontweight="bold", color=INK, xytext=(8, 0),
                    textcoords="offset points")
    ax.tick_params(axis="y", labelcolor=INK, labelsize=9.5)
    ax.set_xticks([])
    ax.set_xlim(0, 1.15)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------- Données et Modèle
@st.cache_data
def charger_donnees():
    """Charge et prépare le jeu de données."""
    df = pd.read_csv(FICHIER_DONNEES)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


@st.cache_resource
def entrainer_modele(df):
    """Entraîne le modèle Random Forest et calcule les métriques de test."""
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


# ----------------------------------------------------------------- CSS Tech Light
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="st-"], .stMarkdown, button, input, textarea {
    font-family: 'Plus Jakarta Sans', system-ui, -apple-system, sans-serif !important;
}

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Background général Light Mode */
.stApp {
    background-color: #f8fafc;
    color: #0f172a;
}

.block-container { padding-top: 2rem; max-width: 1200px; }

/* En-tête / Hero Section */
.hero {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.8rem 2.2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);
}

.hero h1 {
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    color: #0f172a;
}

.hero p {
    color: #64748b;
    margin: 0.4rem 0 0 0;
    font-size: 0.95rem;
}

.badge {
    display: inline-block;
    background: #eff6ff;
    color: #2563eb;
    border: 1px solid #bfdbfe;
    border-radius: 99px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-left: 10px;
    vertical-align: middle;
}

/* Containers de cartes */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04) !important;
}

.card-titre {
    font-size: 1rem;
    font-weight: 600;
    color: #0f172a;
    margin: 0 0 0.1rem 0;
}

.card-sous-titre {
    font-size: 0.82rem;
    color: #64748b;
    margin: 0 0 0.8rem 0;
}

/* Métriques (KPI) */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.03);
}

[data-testid="stMetricLabel"] p {
    color: #64748b;
    font-size: 0.82rem;
    font-weight: 500;
}

[data-testid="stMetricValue"] {
    color: #0f172a;
    font-weight: 700;
    font-size: 1.6rem;
}

/* Onglets Custom */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    border-bottom: 1px solid #e2e8f0;
    background: transparent;
    padding: 0 0 10px 0;
}

.stTabs [data-baseweb="tab"] {
    height: 40px;
    padding: 0 20px;
    color: #64748b;
    font-weight: 500;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
}

.stTabs [aria-selected="true"] {
    color: #2563eb !important;
    background: #eff6ff !important;
    border-color: #bfdbfe !important;
    font-weight: 600 !important;
}

/* Boutons */
.stFormSubmitButton button, .stButton button {
    background: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    width: 100%;
    box-shadow: 0 2px 6px rgba(37, 99, 235, 0.2);
    transition: all 0.2s ease;
}

.stFormSubmitButton button:hover, .stButton button:hover {
    background: #1d4ed8;
    color: #ffffff;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

/* Cartes de résultats */
.resultat {
    border-radius: 10px;
    padding: 16px;
    font-weight: 600;
    font-size: 1.05rem;
    text-align: center;
    margin-bottom: 1rem;
}

.resultat.normal {
    background: #ecfdf5;
    color: #059669;
    border: 1px solid #a7f3d0;
}

.resultat.suspect {
    background: #fffbeb;
    color: #d97706;
    border: 1px solid #fde68a;
}

.resultat.fraude {
    background: #fef2f2;
    color: #dc2626;
    border: 1px solid #fecaca;
}
</style>
"""


def carte_titre(titre, sous_titre=None):
    """Génère un en-tête de carte propre."""
    st.markdown(f'<p class="card-titre">{titre}</p>', unsafe_allow_html=True)
    if sous_titre:
        st.markdown(f'<p class="card-sous-titre">{sous_titre}</p>', unsafe_allow_html=True)


# ----------------------------------------------------------------- Application Principale
def main():
    st.set_page_config(page_title="Détection de fraude bancaire", page_icon="🛡️", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    df = charger_donnees()
    modele, encodeurs, le_target, metriques = entrainer_modele(df)

    # Hero / Banner
    st.markdown(
        '<div class="hero">'
        '<h1>Détection de Fraude Bancaire <span class="badge">Random Forest</span></h1>'
        '<p>Plateforme de surveillance et d\'évaluation du risque des transactions sénégalaises — Scénario 1</p>'
        '</div>',
        unsafe_allow_html=True
    )

    onglet_donnees, onglet_modele, onglet_prediction = st.tabs(
        ["📈 Données & Exploration", "⚙️ Performance Modèle", "⚡ Inférence & Simulation"]
    )

    # ------------------------------------------------------------- 1. Données
    with onglet_donnees:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", fmt(len(df)))
        c2.metric("Cas de Fraude", fmt((df["Target"] == "Fraude").sum()))
        c3.metric("Villes Couvertes", df["Localisation"].nunique())
        c4.metric("Montant Moyen", fmt(df["Montant"].mean()) + " FCFA")

        st.write("")
        gauche, droite = st.columns(2)
        with gauche, st.container(border=True):
            carte_titre("Répartition des Classes", "Distribution du statut des opérations")
            st.pyplot(fig_target(df), use_container_width=True)

        with droite, st.container(border=True):
            carte_titre("Top 10 Villes Actives", "Volume de transactions par localisation")
            st.pyplot(fig_villes(df), use_container_width=True)

        with st.container(border=True):
            carte_titre("Analyse des Montants par Canal", "Distribution financière en FCFA (Échelle Log)")
            st.pyplot(fig_boxplot(df), use_container_width=True)

        with st.container(border=True):
            carte_titre("Aperçu du Dataset Cleaned", f"{fmt(len(df))} opérations enregistrées")
            st.dataframe(df.head(100), use_container_width=True, height=280)

    # ------------------------------------------------------------- 2. Modèle
    with onglet_modele:
        c1, c2, c3 = st.columns(3)
        c1.metric("Exactitude (Accuracy)", f"{metriques['accuracy']:.1%}")
        c2.metric("Score F1-Macro", f"{metriques['f1_macro']:.3f}")
        c3.metric("Échantillon de Test", fmt(metriques["matrice"].sum()))

        st.write("")
        gauche, droite = st.columns(2)
        with gauche, st.container(border=True):
            carte_titre("Matrice de Confusion", "Évaluation des vrais/faux positifs")
            st.pyplot(fig_confusion(metriques["matrice"], le_target.classes_), use_container_width=True)

        with droite, st.container(border=True):
            carte_titre("Importance des Features", "Poids décisionnel de chaque variable dans l'arbre")
            st.pyplot(fig_importance(modele, metriques["features"]), use_container_width=True)

        with st.container(border=True):
            carte_titre("Rapport de Classification Détaillé", "Precision & Recall par classe")
            st.code(metriques["rapport"], language=None)

    # ------------------------------------------------------------- 3. Prédiction
    with onglet_prediction:
        gauche, droite = st.columns([5, 4])

        with gauche, st.container(border=True):
            carte_titre("Simulateur de Transaction", "Renseignez les métadonnées pour évaluer le risque")
            with st.form("formulaire"):
                montant = st.number_input("Montant de la transaction (FCFA)", min_value=0.0, value=50000.0, step=1000.0)
                ville = st.selectbox("Localisation", sorted(df["Localisation"].astype(str).unique()))
                type_tx = st.selectbox("Type de canal", sorted(df["Type de transaction"].astype(str).unique()))
                
                c_h, c_j, c_m = st.columns(3)
                with c_h:
                    heure = st.slider("Heure (0-23h)", 0, 23, 12)
                with c_j:
                    jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
                    jour = st.selectbox("Jour", jours)
                with c_m:
                    mois = st.slider("Mois", 1, 12, 6)
                
                envoyer = st.form_submit_button("Lancer l'évaluation")

        with droite:
            if envoyer:
                entree = pd.DataFrame([{
                    "Montant": montant,
                    "Heure": heure,
                    "Jour_semaine": jours.index(jour),
                    "Mois": mois,
                    "Localisation_enc": encodeurs["Localisation"].transform([ville])[0],
                    "Type de transaction_enc": encodeurs["Type de transaction"].transform([type_tx])[0],
                }])
                prediction = le_target.classes_[modele.predict(entree)[0]]
                probas = modele.predict_proba(entree)[0]

                with st.container(border=True):
                    carte_titre("Diagnostic du Modèle")
                    classe_css = prediction.lower()
                    st.markdown(
                        f'<div class="resultat {classe_css}">'
                        f'Résultat : {prediction.upper()}</div>',
                        unsafe_allow_html=True
                    )
                    st.pyplot(fig_probas(probas, list(le_target.classes_)), use_container_width=True)
            else:
                with st.container(border=True):
                    carte_titre("Diagnostic du Modèle")
                    st.markdown(
                        '<p class="card-sous-titre">En attente de saisie. Remplissez le formulaire et cliquez sur '
                        '« Lancer l\'évaluation ».</p>',
                        unsafe_allow_html=True
                    )


if __name__ == "__main__":
    main()