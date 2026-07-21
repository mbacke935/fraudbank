# Importation de Streamlit pour l'interface web
import streamlit as st
# Importation de Pandas pour la manipulation des données
import pandas as pd
# Importation de Matplotlib pour les graphiques
import matplotlib.pyplot as plt
# Importation de LinearSegmentedColormap pour créer des dégradés
from matplotlib.colors import LinearSegmentedColormap
# Importation de LabelEncoder pour encoder les catégories
from sklearn.preprocessing import LabelEncoder
# Importation de train_test_split pour séparer les données
from sklearn.model_selection import train_test_split
# Importation de RandomForestClassifier pour le modèle d'IA
from sklearn.ensemble import RandomForestClassifier
# Importation des métriques d'évaluation du modèle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Chemin du fichier CSV de données
FICHIER_DONNEES = "scenario1_propre.csv"

# =============================================================================
# IDENTITÉ VISUELLE — Design produit moderne, base bleu foncé (fintech / SaaS)
# =============================================================================

# Fond général (bleu nuit profond)
FOND = "#0A1428"
# Surface des cartes (bleu marine légèrement plus clair)
SURFACE = "#101C3B"
# Surface élevée / survol (encore un ton plus clair)
SURFACE_HAUTE = "#16264C"
# Couleur des séparateurs discrets
BORDURE = "#22315C"
# Texte principal (blanc bleuté doux)
TEXTE = "#EAF0FF"
# Texte secondaire (bleu-gris atténué)
TEXTE_ATTENUE = "#8CA0C7"

# Accent principal (indigo électrique, actions et éléments interactifs)
ACCENT = "#4C6FFF"
# Accent principal, variante claire (dégradés, survols)
ACCENT_CLAIR = "#7C97FF"
# Accent secondaire (sarcelle, complémentaire du bleu)
SARCELLE = "#14D6B8"

# Dictionnaire de couleurs pour les statuts de transactions
STATUT = {
    "Normal": "#14D6B8",   # Sarcelle : validation
    "Suspect": "#F6B93B",  # Ambre : vigilance
    "Fraude": "#FF5E77"    # Corail : alerte
}

# Dégradé séquentiel pour la matrice de confusion (marine -> indigo -> sarcelle)
SEQUENTIEL = ["#101C3B", "#182A55", "#233F7E", "#3557B0", "#4C6FFF", "#7C97FF", "#B7E9E0"]

# Configuration par défaut des textes Matplotlib
plt.rcParams.update({
    "font.family": "sans-serif",   # Police géométrique moderne
    "font.size": 9.5,              # Taille de police standard
    "text.color": TEXTE_ATTENUE,   # Couleur du texte
    "axes.labelcolor": TEXTE_ATTENUE,  # Couleur des étiquettes d'axes
    "xtick.color": TEXTE_ATTENUE,      # Couleur des graduations X
    "ytick.color": TEXTE_ATTENUE,      # Couleur des graduations Y
})


# Fonction de formatage des nombres
def fmt(n):
    # Séparation des milliers par un espace
    return f"{n:,.0f}".replace(",", " ")


# Fonction de base pour créer une figure Matplotlib sans traits ni grilles
def _base(figsize):
    # Création de la figure et de l'axe avec fond sombre
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    # Définition de la couleur de fond de l'axe
    ax.set_facecolor(SURFACE)
    # Suppression de tous les traits d'encadrement (spines)
    for cote in ("top", "right", "left", "bottom"):
        ax.spines[cote].set_visible(False)
    # Suppression de la grille
    ax.grid(False)
    # Suppression des encoches sur les axes
    ax.tick_params(length=0, labelsize=9)
    # Renvoi de la figure et de l'axe
    return fig, ax


# Fonction pour tracer la répartition des statuts
def fig_target(df):
    # Sélection des catégories présentes dans le dataset
    ordre = [c for c in ("Normal", "Suspect", "Fraude") if c in df["Target"].unique()]
    # Comptage des valeurs selon l'ordre choisi
    valeurs = df["Target"].value_counts().reindex(ordre)
    # Création de la figure de base sans traits
    fig, ax = _base((6.4, 3.6))
    # Création du graphique en barres colorées
    ax.bar(range(len(ordre)), valeurs, width=0.45, color=[STATUT[c] for c in ordre], zorder=2)
    # Ajout des valeurs chiffrées au-dessus des barres
    for i, v in enumerate(valeurs):
        ax.annotate(fmt(v), (i, v), ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=TEXTE, xytext=(0, 5), textcoords="offset points")
    # Configuration des positions X
    ax.set_xticks(range(len(ordre)))
    # Configuration des étiquettes X
    ax.set_xticklabels(ordre, color=TEXTE, fontsize=9.5, fontweight="600")
    # Masquage de l'axe Y
    ax.set_yticks([])
    # Ajustement de la limite verticale
    ax.set_ylim(0, valeurs.max() * 1.15)
    # Ajustement automatique des marges
    fig.tight_layout()
    # Renvoi du graphique
    return fig


# Fonction pour tracer le Top 10 des villes
def fig_villes(df):
    # Extraction des 10 villes les plus fréquentes
    villes = df["Localisation"].value_counts().head(10).sort_values()
    # Création de la figure de base sans traits
    fig, ax = _base((6.4, 3.9))
    # Création des barres horizontales
    ax.barh(villes.index, villes.values, height=0.5, color=ACCENT_CLAIR, zorder=2)
    # Affichage du nombre exact à côté de chaque barre
    for i, v in enumerate(villes.values):
        ax.annotate(fmt(v), (v, i), va="center", fontsize=9, color=TEXTE, xytext=(6, 0), textcoords="offset points")
    # Style des étiquettes verticales
    ax.tick_params(axis="y", labelcolor=TEXTE, labelsize=9.5)
    # Masquage des graduations X
    ax.set_xticks([])
    # Définition de la limite horizontale
    ax.set_xlim(0, villes.max() * 1.15)
    # Ajustement des marges
    fig.tight_layout()
    # Renvoi de la figure
    return fig


# Fonction pour tracer les boîtes à moustaches
def fig_boxplot(df):
    # Extraction et tri des types de transactions
    types_tx = sorted(df["Type de transaction"].astype(str).unique())
    # Filtrage des montants par type de transaction
    donnees = [df.loc[(df["Type de transaction"] == t) & (df["Montant"] > 0), "Montant"] for t in types_tx]
    # Création du graphique de base sans traits
    fig, ax = _base((9.6, 3.8))
    # Passage à une échelle logarithmique pour la lisibilité
    ax.set_yscale("log")
    # Dessin des boîtes à moustaches
    ax.boxplot(
        donnees, widths=0.4, patch_artist=True,
        boxprops=dict(facecolor=SURFACE_HAUTE, edgecolor=ACCENT_CLAIR, linewidth=1.2),
        medianprops=dict(color=SARCELLE, linewidth=2),
        whiskerprops=dict(color=TEXTE_ATTENUE, linewidth=1),
        capprops=dict(color=TEXTE_ATTENUE, linewidth=1),
        flierprops=dict(marker="o", markersize=3, markerfacecolor=TEXTE_ATTENUE, markeredgecolor="none", alpha=0.4)
    )
    # Nom des canaux en axe X
    ax.set_xticklabels(types_tx, color=TEXTE, fontsize=9.5, fontweight="500")
    # Formateur personnalisé des montants en axe Y
    ax.yaxis.set_major_formatter(lambda x, _: fmt(x))
    # Suppression des étiquettes secondaires Y
    ax.yaxis.set_minor_formatter(lambda x, _: "")
    # Titre de l'axe Y
    ax.set_ylabel("Montant (FCFA) — échelle log", fontsize=9, color=TEXTE_ATTENUE)
    # Ajustement des marges
    fig.tight_layout()
    # Renvoi du graphique
    return fig


# Fonction pour afficher la matrice de confusion
def fig_confusion(cm, classes):
    # Création d'une palette dégradée bleu -> sarcelle
    cmap = LinearSegmentedColormap.from_list("marine_indigo", SEQUENTIEL)
    # Création de la figure
    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=SURFACE)
    # Couleur de fond
    ax.set_facecolor(SURFACE)
    # Affichage de la matrice sous forme d'image
    ax.imshow(cm, cmap=cmap)
    # Définition du seuil pour la couleur du texte
    seuil = cm.max() * 0.55
    # Boucle sur les lignes et colonnes de la matrice
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Choisir sombre si fond clair du dégradé, sinon texte clair
            couleur = "#0A1428" if cm[i, j] > seuil else TEXTE
            # Inscription de la valeur numérique
            ax.text(j, i, fmt(cm[i, j]), ha="center", va="center", fontsize=11, fontweight="bold", color=couleur)
    # Configuration des graduations X
    ax.set_xticks(range(len(classes)))
    # Configuration des graduations Y
    ax.set_yticks(range(len(classes)))
    # Étiquettes X
    ax.set_xticklabels(classes, color=TEXTE, fontsize=9.5, fontweight="500")
    # Étiquettes Y
    ax.set_yticklabels(classes, color=TEXTE, fontsize=9.5, fontweight="500")
    # Légende de l'axe X
    ax.set_xlabel("Prédiction", fontsize=9.5, color=TEXTE_ATTENUE, labelpad=10)
    # Légende de l'axe Y
    ax.set_ylabel("Vraie valeur", fontsize=9.5, color=TEXTE_ATTENUE, labelpad=10)
    # Suppression des encoches
    ax.tick_params(length=0)
    # Masquage de tous les traits extérieurs
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Ajustement de la disposition
    fig.tight_layout()
    # Renvoi de la matrice visuelle
    return fig


# Fonction pour tracer l'importance des variables
def fig_importance(modele, features):
    # Calcul et tri des importances de variables
    imp = pd.Series(modele.feature_importances_, index=features).sort_values()
    # Dictionnaire de traduction propre des noms de colonnes
    noms = {"Montant": "Montant", "Heure": "Heure", "Jour_semaine": "Jour de la semaine", "Mois": "Mois", "Localisation_enc": "Localisation", "Type de transaction_enc": "Type de transaction"}
    # Création du graphique de base sans traits
    fig, ax = _base((5.6, 3.6))
    # Création des barres d'importance
    ax.barh([noms.get(f, f) for f in imp.index], imp.values, height=0.45, color=ACCENT, zorder=2)
    # Affichage du score exact à côté de la barre
    for i, v in enumerate(imp.values):
        ax.annotate(f"{v:.3f}", (v, i), va="center", fontsize=9, color=TEXTE, xytext=(6, 0), textcoords="offset points")
    # Masquer les graduations X
    ax.set_xticks([])
    # Ajuster la largeur max
    ax.set_xlim(0, imp.max() * 1.2)
    # Ajuster le layout
    fig.tight_layout()
    # Renvoi de la figure
    return fig


# Fonction pour tracer les probabilités estimées
def fig_probas(probas, classes):
    # Création du graphique de base sans traits
    fig, ax = _base((6.0, 2.2))
    # Associer la couleur correspondant au statut
    couleurs = [STATUT.get(c, ACCENT) for c in classes]
    # Barres de probabilités
    ax.barh(classes, probas, height=0.45, color=couleurs, zorder=2)
    # Pourcentage affiché en texte
    for i, v in enumerate(probas):
        ax.annotate(f"{v:.0%}", (v, i), va="center", fontsize=9.5, fontweight="bold", color=TEXTE, xytext=(8, 0), textcoords="offset points")
    # Couleur du texte des catégories
    ax.tick_params(axis="y", labelcolor=TEXTE, labelsize=9.5)
    # Masquer les étiquettes X
    ax.set_xticks([])
    # Limite max de l'axe X
    ax.set_xlim(0, 1.15)
    # Ajustement final
    fig.tight_layout()
    # Renvoi du graphique
    return fig


# Fonction mise en cache pour charger les données
@st.cache_data
def charger_donnees():
    # Lecture du fichier CSV
    df = pd.read_csv(FICHIER_DONNEES)
    # Conversion de la colonne Date en datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    # Renvoi du dataframe
    return df


# Fonction mise en cache pour entraîner le modèle
@st.cache_resource
def entrainer_modele(df):
    # Copie de sécurité des données
    df_ml = df.copy()
    # Extraction de l'heure
    df_ml["Heure"] = df_ml["Date"].dt.hour
    # Extraction du jour de la semaine
    df_ml["Jour_semaine"] = df_ml["Date"].dt.dayofweek
    # Extraction du mois
    df_ml["Mois"] = df_ml["Date"].dt.month

    # Dictionnaire de stockage des encodeurs
    encodeurs = {}
    # Encodage des variables texte en valeurs numériques
    for col in ["Localisation", "Type de transaction"]:
        le = LabelEncoder()
        df_ml[col + "_enc"] = le.fit_transform(df_ml[col].astype(str))
        encodeurs[col] = le

    # Encodage de la variable cible
    le_target = LabelEncoder()
    df_ml["Target_enc"] = le_target.fit_transform(df_ml["Target"].astype(str))

    # Définition des variables explicatives
    features = ["Montant", "Heure", "Jour_semaine", "Mois", "Localisation_enc", "Type de transaction_enc"]
    # Matrice de caractéristiques
    X = df_ml[features]
    # Vecteur cible
    y = df_ml["Target_enc"]

    # Séparation entraînement (80%) et test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Initialisation de la forêt aléatoire
    modele = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    # Entraînement de l'algorithme
    modele.fit(X_train, y_train)
    # Prédictions sur l'échantillon de test
    y_pred = modele.predict(X_test)

    # Calcul des indicateurs de performance
    metriques = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "matrice": confusion_matrix(y_test, y_pred),
        "rapport": classification_report(y_test, y_pred, target_names=le_target.classes_),
        "features": features,
    }
    # Renvoi des éléments entraînés
    return modele, encodeurs, le_target, metriques


# =============================================================================
# CSS — design produit moderne, base bleu foncé, cartes en verre dépoli léger,
# dégradés indigo/sarcelle, typographie géométrique (Space Grotesk + Inter).
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

html, body, [class*="st-"], .stMarkdown, button, input, textarea, select {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Fond général bleu nuit */
.stApp {
    background: radial-gradient(1200px 600px at 15% -10%, #16264C 0%, #0A1428 55%), #0A1428;
    color: #EAF0FF;
}

.block-container { padding-top: 2rem; max-width: 1200px; }

/* Bandeau d'en-tête : dégradé indigo profond, halo lumineux discret */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, #16264C 0%, #101C3B 60%, #0E1A38 100%);
    border: 1px solid #22315C;
    border-radius: 20px;
    padding: 2rem 2.4rem;
    margin-bottom: 1.6rem;
}

.hero::after {
    content: "";
    position: absolute;
    top: -60px;
    right: -60px;
    width: 260px;
    height: 260px;
    background: radial-gradient(circle, rgba(76,111,255,0.35) 0%, rgba(76,111,255,0) 70%);
    pointer-events: none;
}

.hero .pill {
    display: inline-block;
    background: rgba(76,111,255,0.15);
    border: 1px solid rgba(76,111,255,0.4);
    color: #7C97FF;
    border-radius: 99px;
    padding: 5px 14px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    margin: 0;
    color: #FFFFFF;
    letter-spacing: -0.01em;
    position: relative;
    z-index: 1;
}

.hero p {
    color: #8CA0C7;
    margin: 0.5rem 0 0 0;
    font-size: 0.98rem;
    position: relative;
    z-index: 1;
}

/* Cartes / conteneurs Streamlit */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #101C3B;
    border: 1px solid #22315C !important;
    border-radius: 16px !important;
}

.card-titre {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #EAF0FF;
    margin: 0 0 0.15rem 0;
}

.card-sous-titre {
    font-size: 0.82rem;
    color: #8CA0C7;
    margin: 0 0 0.9rem 0;
}

/* Métriques d'en-tête (KPIs) avec accent latéral coloré */
[data-testid="stMetric"] {
    background: #101C3B;
    border: 1px solid #22315C;
    border-left: 3px solid #4C6FFF;
    border-radius: 12px;
    padding: 16px 20px;
}

[data-testid="stMetricLabel"] p {
    color: #8CA0C7;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.01em;
}

[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk', sans-serif;
    color: #FFFFFF;
    font-weight: 700;
    font-size: 1.65rem;
}

/* Onglets en pilule (segmented control) */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #101C3B;
    padding: 6px;
    border: 1px solid #22315C;
    border-radius: 12px;
    border-bottom: none;
    width: fit-content;
}

.stTabs [data-baseweb="tab"] {
    height: 38px;
    padding: 0 20px;
    color: #8CA0C7;
    font-weight: 500;
    background: transparent;
    border: none;
    border-radius: 8px;
}

.stTabs [aria-selected="true"] {
    color: #FFFFFF !important;
    background: linear-gradient(135deg, #4C6FFF 0%, #3557B0 100%) !important;
    font-weight: 600 !important;
}

/* Boutons : dégradé indigo avec légère élévation au survol */
.stFormSubmitButton button, .stButton button {
    background: linear-gradient(135deg, #4C6FFF 0%, #3557B0 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2rem;
    font-weight: 600;
    width: 100%;
    box-shadow: 0 4px 14px rgba(76,111,255,0.25);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}

.stFormSubmitButton button:hover, .stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(76,111,255,0.4);
    color: #FFFFFF;
}

/* Bannière de diagnostic : barre latérale colorée + halo, sans pictogramme */
.diagnostic {
    border-radius: 14px;
    padding: 20px 22px;
    margin-bottom: 1rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
}

.diagnostic .label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    opacity: 0.85;
    margin-bottom: 4px;
    display: block;
}

.diagnostic .valeur {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
}

.diagnostic.normal {
    background: rgba(20,214,184,0.08);
    border-color: rgba(20,214,184,0.4);
    border-left: 4px solid #14D6B8;
    color: #14D6B8;
}

.diagnostic.suspect {
    background: rgba(246,185,59,0.08);
    border-color: rgba(246,185,59,0.4);
    border-left: 4px solid #F6B93B;
    color: #F6B93B;
}

.diagnostic.fraude {
    background: rgba(255,94,119,0.08);
    border-color: rgba(255,94,119,0.4);
    border-left: 4px solid #FF5E77;
    color: #FF5E77;
}

/* Style du tableau Streamlit */
div[data-testid="stDataFrame"] {
    background-color: #101C3B;
    border-radius: 12px;
    border: 1px solid #22315C;
    overflow: hidden;
}

div[data-testid="stDataFrame"] iframe {
    border-radius: 12px;
}

code, .stCodeBlock, pre {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Champs de saisie : bordure discrète, focus indigo */
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {
    background: #0E1A38 !important;
    border-color: #22315C !important;
    border-radius: 10px !important;
}
</style>
"""


# Fonction pour afficher le titre d'une section
def carte_titre(titre, sous_titre=None):
    # Injection du titre en HTML
    st.markdown(f'<p class="card-titre">{titre}</p>', unsafe_allow_html=True)
    # Injection du sous-titre si présent
    if sous_titre:
        st.markdown(f'<p class="card-sous-titre">{sous_titre}</p>', unsafe_allow_html=True)


# Fonction principale exécutant l'application
def main():
    # Configuration initiale de la page web
    st.set_page_config(page_title="Détection de fraude bancaire", layout="wide")
    # Application des styles CSS
    st.markdown(CSS, unsafe_allow_html=True)

    # Chargement du jeu de données
    df = charger_donnees()
    # Entraînement du modèle et récupération des métriques
    modele, encodeurs, le_target, metriques = entrainer_modele(df)

    # Création du bloc d'en-tête
    st.markdown(
        '<div class="hero">'
        '<span class="pill">Random Forest · Scénario 1</span>'
        '<h1>Détection de fraude bancaire</h1>'
        '<p>Plateforme de surveillance et d\'évaluation du risque des transactions sénégalaises</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # Création des onglets
    onglet_donnees, onglet_modele, onglet_prediction = st.tabs(
        ["Données & Exploration", "Performance Modèle", "Inférence & Simulation"]
    )

    # --- Onglet 1 : Exploration des données ---
    with onglet_donnees:
        # Création de 4 colonnes de statistiques
        c1, c2, c3, c4 = st.columns(4)
        # Affichage du nombre total de transactions
        c1.metric("Total Transactions", fmt(len(df)))
        # Affichage du total de fraudes détectées
        c2.metric("Cas de Fraude", fmt((df["Target"] == "Fraude").sum()))
        # Affichage du nombre de villes couvertes
        c3.metric("Villes Couvertes", df["Localisation"].nunique())
        # Affichage du montant moyen par transaction
        c4.metric("Montant Moyen", fmt(df["Montant"].mean()) + " FCFA")

        # Espace vide de séparation
        st.write("")
        # Disposition en 2 colonnes pour les graphiques
        gauche, droite = st.columns(2)
        # Bloc de gauche pour le statut des opérations
        with gauche, st.container(border=True):
            carte_titre("Répartition des Classes", "Distribution du statut des opérations")
            st.pyplot(fig_target(df), use_container_width=True)

        # Bloc de droite pour les villes
        with droite, st.container(border=True):
            carte_titre("Top 10 Villes Actives", "Volume de transactions par localisation")
            st.pyplot(fig_villes(df), use_container_width=True)

        # Conteneur pour le boxplot
        with st.container(border=True):
            carte_titre("Analyse des Montants par Canal", "Distribution financière en FCFA (Échelle Log)")
            st.pyplot(fig_boxplot(df), use_container_width=True)

        # Conteneur pour le tableau de données
        with st.container(border=True):
            carte_titre("Aperçu du Dataset Cleaned", f"{fmt(len(df))} opérations enregistrées au total")

            # Barre d'outils de filtrage rapide au-dessus du tableau
            f_col1, f_col2 = st.columns([1, 2])
            with f_col1:
                # Filtre par statut
                filtre_statut = st.multiselect(
                    "Filtrer par statut",
                    options=list(df["Target"].unique()),
                    default=list(df["Target"].unique()),
                    placeholder="Choisir un statut"
                )
            with f_col2:
                # Recherche textuelle sur la ville
                recherche_ville = st.text_input("Rechercher une ville", placeholder="Ex: Dakar, Thiès...")

            # Application des filtres
            df_filtre = df[df["Target"].isin(filtre_statut)]
            if recherche_ville:
                df_filtre = df_filtre[df_filtre["Localisation"].str.contains(recherche_ville, case=False, na=False)]

            # Affichage du tableau interactif stylisé
            st.dataframe(
                df_filtre,
                use_container_width=True,
                height=320,
                hide_index=True,
                column_config={
                    "Date": st.column_config.DatetimeColumn("Date & Heure", format="DD/MM/YYYY HH:mm"),
                    "Montant": st.column_config.NumberColumn("Montant (FCFA)", format="%d FCFA"),
                    "Target": st.column_config.TextColumn("Statut"),
                    "Localisation": st.column_config.TextColumn("Ville"),
                    "Type de transaction": st.column_config.TextColumn("Canal")
                }
            )

    # --- Onglet 2 : Performance du modèle ---
    with onglet_modele:
        # Colonnes de métriques globales
        c1, c2, c3 = st.columns(3)
        # Taux d'exactitude (Accuracy)
        c1.metric("Exactitude (Accuracy)", f"{metriques['accuracy']:.1%}")
        # Score F1 global
        c2.metric("Score F1-Macro", f"{metriques['f1_macro']:.3f}")
        # Nombre de cas de test
        c3.metric("Échantillon de Test", fmt(metriques["matrice"].sum()))

        # Espace vide
        st.write("")
        # Colonnes pour les graphiques
        gauche, droite = st.columns(2)
        # Matrice de confusion
        with gauche, st.container(border=True):
            carte_titre("Matrice de Confusion", "Évaluation des vrais/faux positifs")
            st.pyplot(fig_confusion(metriques["matrice"], le_target.classes_), use_container_width=True)

        # Importance des variables
        with droite, st.container(border=True):
            carte_titre("Importance des Features", "Poids décisionnel de chaque variable dans l'arbre")
            st.pyplot(fig_importance(modele, metriques["features"]), use_container_width=True)

        # Rapport de classification texte
        with st.container(border=True):
            carte_titre("Rapport de Classification Détaillé", "Precision & Recall par classe")
            st.code(metriques["rapport"], language=None)

    # --- Onglet 3 : Simulation et prédiction ---
    with onglet_prediction:
        # Division en deux colonnes (formulaire à gauche, résultat à droite)
        gauche, droite = st.columns([5, 4])

        # Formulaire de saisie utilisateur
        with gauche, st.container(border=True):
            carte_titre("Simulateur de Transaction", "Renseignez les métadonnées pour évaluer le risque")
            # Début du formulaire Streamlit
            with st.form("formulaire"):
                # Saisie du montant
                montant = st.number_input("Montant de la transaction (FCFA)", min_value=0.0, value=50000.0, step=1000.0)
                # Sélection de la ville
                ville = st.selectbox("Localisation", sorted(df["Localisation"].astype(str).unique()))
                # Sélection du canal de paiement
                type_tx = st.selectbox("Type de canal", sorted(df["Type de transaction"].astype(str).unique()))

                # Sous-colonnes pour la date et heure
                c_h, c_j, c_m = st.columns(3)
                with c_h:
                    # Sélecteur d'heure
                    heure = st.slider("Heure (0-23h)", 0, 23, 12)
                with c_j:
                    # Liste des jours de la semaine
                    jours = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
                    # Sélecteur de jour
                    jour = st.selectbox("Jour", jours)
                with c_m:
                    # Sélecteur de mois
                    mois = st.slider("Mois", 1, 12, 6)

                # Bouton de soumission
                envoyer = st.form_submit_button("Lancer l'évaluation")

        # Traitement et affichage de la prédiction
        with droite:
            if envoyer:
                # Préparation du tableau d'entrée pour le modèle
                entree = pd.DataFrame([{
                    "Montant": montant,
                    "Heure": heure,
                    "Jour_semaine": jours.index(jour),
                    "Mois": mois,
                    "Localisation_enc": encodeurs["Localisation"].transform([ville])[0],
                    "Type de transaction_enc": encodeurs["Type de transaction"].transform([type_tx])[0],
                }])
                # Prédiction de la classe
                prediction = le_target.classes_[modele.predict(entree)[0]]
                # Calcul des probabilités pour chaque classe
                probas = modele.predict_proba(entree)[0]

                # Affichage de la carte de résultat
                with st.container(border=True):
                    carte_titre("Diagnostic du Modèle")
                    # Classe CSS dynamique
                    classe_css = prediction.lower()
                    # Bannière de diagnostic avec barre latérale colorée
                    st.markdown(
                        f'<div class="diagnostic {classe_css}">'
                        f'<span class="label">Résultat de l\'évaluation</span>'
                        f'<span class="valeur">{prediction.upper()}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    # Graphique des probabilités
                    st.pyplot(fig_probas(probas, list(le_target.classes_)), use_container_width=True)
            else:
                # Message en attente de soumission
                with st.container(border=True):
                    carte_titre("Diagnostic du Modèle")
                    st.markdown(
                        '<p class="card-sous-titre">En attente de saisie. Remplissez le formulaire et cliquez sur '
                        '« Lancer l\'évaluation ».</p>',
                        unsafe_allow_html=True
                    )


# Point d'entrée du script Python
if __name__ == "__main__":
    # Exécution de la fonction principale
    main()