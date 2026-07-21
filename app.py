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
# IDENTITÉ VISUELLE — « Registre des opérations »
# Inspiration : le grand livre comptable / registre papier des agences
# bancaires ouest-africaines. Papier ivoire, encre indigo (teinture bogolan),
# rehauts ocre (or du Sahel) et tampon terracotta pour les alertes.
# =============================================================================

# Couleur de fond général (papier ivoire)
PAPIER = "#F4EEDF"
# Couleur des cartes / blocs (crème plus clair, façon page)
PAPIER_CARTE = "#FFFCF4"
# Couleur des lignes de séparation (ligne de cahier)
LIGNE = "#D8CCA8"
# Couleur d'encre principale (texte, quasi noir-indigo)
ENCRE = "#221B12"
# Couleur d'encre atténuée (texte secondaire)
ENCRE_ATTENUEE = "#6B6250"

# Couleur d'accent principal (indigo bogolan)
INDIGO = "#2C3E6B"
# Couleur d'accent indigo clair
INDIGO_CLAIR = "#4A5D95"
# Couleur d'accent secondaire (ocre / or du Sahel)
OCRE = "#B9862E"
# Couleur d'alerte (tampon terracotta)
TERRACOTTA = "#AE4430"

# Dictionnaire de couleurs pour les statuts de transactions
STATUT = {
    "Normal": "#3F7A54",    # Vert sourdine, encre de validation
    "Suspect": "#B9862E",   # Ocre, encre d'attention
    "Fraude": "#AE4430"     # Terracotta, encre de tampon d'alerte
}

# Dégradé séquentiel papier -> indigo pour la matrice de confusion
SEQUENTIEL = ["#F4EEDF", "#DDD2AC", "#B7A672", "#8C7A54", "#5C6B98", "#2C3E6B", "#1B2440"]

# Configuration par défaut des textes Matplotlib (police serif, esprit registre)
plt.rcParams.update({
    "font.family": "serif",       # Police avec empattement, esprit livre relié
    "font.size": 9.5,             # Taille de police standard
    "text.color": ENCRE,          # Couleur du texte
    "axes.labelcolor": ENCRE_ATTENUEE,  # Couleur des étiquettes d'axes
    "xtick.color": ENCRE_ATTENUEE,      # Couleur des graduations X
    "ytick.color": ENCRE_ATTENUEE,      # Couleur des graduations Y
})


# Fonction de formatage des nombres
def fmt(n):
    # Séparation des milliers par un espace
    return f"{n:,.0f}".replace(",", " ")


# Fonction de base pour créer une figure Matplotlib sobre, esprit page de registre
def _base(figsize):
    # Création de la figure et de l'axe avec fond papier
    fig, ax = plt.subplots(figsize=figsize, facecolor=PAPIER_CARTE)
    # Définition de la couleur de fond de l'axe
    ax.set_facecolor(PAPIER_CARTE)
    # Suppression de la plupart des traits d'encadrement (spines)
    for cote in ("top", "right", "left"):
        ax.spines[cote].set_visible(False)
    # Conservation d'une ligne de base fine, façon ligne de cahier
    ax.spines["bottom"].set_color(LIGNE)
    ax.spines["bottom"].set_linewidth(1)
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
    # Création de la figure de base
    fig, ax = _base((6.4, 3.6))
    # Création du graphique en barres colorées, largeur fine façon écriture manuscrite
    ax.bar(range(len(ordre)), valeurs, width=0.38, color=[STATUT[c] for c in ordre], zorder=2)
    # Ajout des valeurs chiffrées au-dessus des barres
    for i, v in enumerate(valeurs):
        ax.annotate(fmt(v), (i, v), ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=ENCRE, xytext=(0, 5), textcoords="offset points")
    # Configuration des positions X
    ax.set_xticks(range(len(ordre)))
    # Configuration des étiquettes X
    ax.set_xticklabels(ordre, color=ENCRE, fontsize=9.5, fontweight="600")
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
    # Création de la figure de base
    fig, ax = _base((6.4, 3.9))
    # Création des barres horizontales, couleur indigo clair
    ax.barh(villes.index, villes.values, height=0.45, color=INDIGO_CLAIR, zorder=2)
    # Affichage du nombre exact à côté de chaque barre
    for i, v in enumerate(villes.values):
        ax.annotate(fmt(v), (v, i), va="center", fontsize=9, color=ENCRE, xytext=(6, 0), textcoords="offset points")
    # Style des étiquettes verticales
    ax.tick_params(axis="y", labelcolor=ENCRE, labelsize=9.5)
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
    # Création du graphique de base
    fig, ax = _base((9.6, 3.8))
    # Passage à une échelle logarithmique pour la lisibilité
    ax.set_yscale("log")
    # Dessin des boîtes à moustaches, encre indigo sur fond papier
    ax.boxplot(
        donnees, widths=0.4, patch_artist=True,
        boxprops=dict(facecolor="#EDE4CC", edgecolor=INDIGO, linewidth=1.2),
        medianprops=dict(color=TERRACOTTA, linewidth=2),
        whiskerprops=dict(color=ENCRE_ATTENUEE, linewidth=1),
        capprops=dict(color=ENCRE_ATTENUEE, linewidth=1),
        flierprops=dict(marker="o", markersize=3, markerfacecolor=ENCRE_ATTENUEE, markeredgecolor="none", alpha=0.35)
    )
    # Nom des canaux en axe X
    ax.set_xticklabels(types_tx, color=ENCRE, fontsize=9.5, fontweight="500")
    # Formateur personnalisé des montants en axe Y
    ax.yaxis.set_major_formatter(lambda x, _: fmt(x))
    # Suppression des étiquettes secondaires Y
    ax.yaxis.set_minor_formatter(lambda x, _: "")
    # Titre de l'axe Y
    ax.set_ylabel("Montant (FCFA) — échelle log", fontsize=9, color=ENCRE_ATTENUEE)
    # Ajustement des marges
    fig.tight_layout()
    # Renvoi du graphique
    return fig


# Fonction pour afficher la matrice de confusion
def fig_confusion(cm, classes):
    # Création d'une palette dégradée papier -> indigo
    cmap = LinearSegmentedColormap.from_list("registre_indigo", SEQUENTIEL)
    # Création de la figure
    fig, ax = plt.subplots(figsize=(5.2, 4.4), facecolor=PAPIER_CARTE)
    # Couleur de fond
    ax.set_facecolor(PAPIER_CARTE)
    # Affichage de la matrice sous forme d'image
    ax.imshow(cm, cmap=cmap)
    # Définition du seuil pour la couleur du texte
    seuil = cm.max() * 0.55
    # Boucle sur les lignes et colonnes de la matrice
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Choisir crème si case foncée, sinon encre
            couleur = "#F4EEDF" if cm[i, j] > seuil else ENCRE
            # Inscription de la valeur numérique
            ax.text(j, i, fmt(cm[i, j]), ha="center", va="center", fontsize=11, fontweight="bold", color=couleur)
    # Configuration des graduations X
    ax.set_xticks(range(len(classes)))
    # Configuration des graduations Y
    ax.set_yticks(range(len(classes)))
    # Étiquettes X
    ax.set_xticklabels(classes, color=ENCRE, fontsize=9.5, fontweight="500")
    # Étiquettes Y
    ax.set_yticklabels(classes, color=ENCRE, fontsize=9.5, fontweight="500")
    # Légende de l'axe X
    ax.set_xlabel("Prédiction", fontsize=9.5, color=ENCRE_ATTENUEE, labelpad=10)
    # Légende de l'axe Y
    ax.set_ylabel("Vraie valeur", fontsize=9.5, color=ENCRE_ATTENUEE, labelpad=10)
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
    # Création du graphique de base
    fig, ax = _base((5.6, 3.6))
    # Création des barres d'importance, couleur indigo
    ax.barh([noms.get(f, f) for f in imp.index], imp.values, height=0.42, color=INDIGO, zorder=2)
    # Affichage du score exact à côté de la barre
    for i, v in enumerate(imp.values):
        ax.annotate(f"{v:.3f}", (v, i), va="center", fontsize=9, color=ENCRE, xytext=(6, 0), textcoords="offset points")
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
    # Création du graphique de base
    fig, ax = _base((6.0, 2.2))
    # Associer la couleur correspondant au statut
    couleurs = [STATUT.get(c, INDIGO) for c in classes]
    # Barres de probabilités
    ax.barh(classes, probas, height=0.42, color=couleurs, zorder=2)
    # Pourcentage affiché en texte
    for i, v in enumerate(probas):
        ax.annotate(f"{v:.0%}", (v, i), va="center", fontsize=9.5, fontweight="bold", color=ENCRE, xytext=(8, 0), textcoords="offset points")
    # Couleur du texte des catégories
    ax.tick_params(axis="y", labelcolor=ENCRE, labelsize=9.5)
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
# CSS — esprit « registre papier », sans icône ni tuile IA générique.
# Empattement Fraunces pour les titres, Plex Sans pour le texte, Plex Mono
# pour les chiffres et références, comme sur un relevé bancaire imprimé.
# =============================================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="st-"], .stMarkdown, button, input, textarea, select {
    font-family: 'IBM Plex Sans', system-ui, -apple-system, sans-serif !important;
}

#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Fond général papier ivoire */
.stApp {
    background-color: #F4EEDF;
    color: #221B12;
}

.block-container { padding-top: 2.2rem; max-width: 1180px; }

/* En-tête façon page de garde d'un registre : filet double, pas de carte flottante */
.hero {
    border-top: 3px solid #221B12;
    border-bottom: 1px solid #221B12;
    padding: 0.9rem 0 1.3rem 0;
    margin-bottom: 1.8rem;
}

.hero .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #B9862E;
    font-weight: 600;
}

.hero h1 {
    font-family: 'Fraunces', serif;
    font-size: 2.35rem;
    font-weight: 600;
    margin: 0.25rem 0 0.3rem 0;
    color: #221B12;
    letter-spacing: -0.01em;
}

.hero p {
    color: #6B6250;
    margin: 0;
    font-size: 0.95rem;
    font-style: italic;
}

.hero .ref {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #6B6250;
    float: right;
    margin-top: 0.4rem;
}

/* Cartes / conteneurs Streamlit : filet fin, pas d'arrondi lourd, pas d'ombre */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFCF4;
    border: 1px solid #D8CCA8 !important;
    border-radius: 3px !important;
}

.card-titre {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #221B12;
    margin: 0 0 0.15rem 0;
}

.card-sous-titre {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.03em;
    color: #6B6250;
    margin: 0 0 0.9rem 0;
    text-transform: uppercase;
}

/* Métriques : entrées de registre, chiffres en mono, séparées par un filet */
[data-testid="stMetric"] {
    background: #FFFCF4;
    border: none;
    border-bottom: 2px solid #221B12;
    border-radius: 0;
    padding: 10px 4px 12px 4px;
}

[data-testid="stMetricLabel"] p {
    font-family: 'IBM Plex Mono', monospace;
    color: #6B6250;
    font-size: 0.72rem;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    font-weight: 500;
}

[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif;
    color: #2C3E6B;
    font-weight: 700;
    font-size: 1.7rem;
}

/* Onglets : soulignement façon signets de registre, pas de pilules pleines */
.stTabs [data-baseweb="tab-list"] {
    gap: 26px;
    background: transparent;
    padding: 0 0 0 0;
    border-bottom: 1px solid #D8CCA8;
}

.stTabs [data-baseweb="tab"] {
    height: 38px;
    padding: 0 2px;
    color: #6B6250;
    font-weight: 500;
    background: transparent;
    border: none;
    border-radius: 0;
}

.stTabs [aria-selected="true"] {
    color: #221B12 !important;
    background: transparent !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #B9862E !important;
}

/* Boutons : rectangulaires, encre indigo, petites capitales espacées */
.stFormSubmitButton button, .stButton button {
    background: #2C3E6B;
    color: #F4EEDF;
    border: none;
    border-radius: 2px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-size: 0.85rem;
    width: 100%;
    transition: background 0.15s ease;
}

.stFormSubmitButton button:hover, .stButton button:hover {
    background: #1B2440;
    color: #F4EEDF;
}

/* Tampon de diagnostic : remplace la pastille pleine par un cachet circulaire */
.tampon-cadre {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 1.2rem 0 0.6rem 0;
}

.tampon {
    width: 148px;
    height: 148px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    text-align: center;
    transform: rotate(-7deg);
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.95rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    line-height: 1.3;
}

.tampon-interieur {
    width: 122px;
    height: 122px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0.4rem;
}

.tampon.normal { border: 2px solid #3F7A54; }
.tampon.normal .tampon-interieur { border: 1px dashed #3F7A54; color: #3F7A54; }

.tampon.suspect { border: 2px solid #B9862E; }
.tampon.suspect .tampon-interieur { border: 1px dashed #B9862E; color: #B9862E; }

.tampon.fraude { border: 2px solid #AE4430; }
.tampon.fraude .tampon-interieur { border: 1px dashed #AE4430; color: #AE4430; }

/* Style du tableau Streamlit, esprit page de grand livre */
div[data-testid="stDataFrame"] {
    background-color: #FFFCF4;
    border-radius: 2px;
    border: 1px solid #D8CCA8;
    overflow: hidden;
}

div[data-testid="stDataFrame"] iframe {
    border-radius: 2px;
}

code, .stCodeBlock, pre {
    font-family: 'IBM Plex Mono', monospace !important;
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
    st.set_page_config(page_title="Registre des transactions — Détection de fraude", layout="wide")
    # Application des styles CSS
    st.markdown(CSS, unsafe_allow_html=True)

    # Chargement du jeu de données
    df = charger_donnees()
    # Entraînement du modèle et récupération des métriques
    modele, encodeurs, le_target, metriques = entrainer_modele(df)

    # Création du bloc d'en-tête façon page de garde, sans icône ni pastille
    st.markdown(
        '<div class="hero">'
        '<span class="ref">RÉF. SCÉNARIO 1 · MODÈLE FORÊT ALÉATOIRE</span>'
        '<div class="eyebrow">Registre des opérations</div>'
        '<h1>Surveillance de la fraude bancaire</h1>'
        '<p>Relevé d\'analyse et d\'évaluation du risque des transactions sénégalaises</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # Création des onglets sans icônes
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

                # Affichage de la carte de résultat sous forme de cachet
                with st.container(border=True):
                    carte_titre("Diagnostic du Modèle")
                    # Classe CSS dynamique
                    classe_css = prediction.lower()
                    # Inscription du résultat sous forme de tampon circulaire
                    st.markdown(
                        f'<div class="tampon-cadre">'
                        f'<div class="tampon {classe_css}">'
                        f'<div class="tampon-interieur">{prediction}</div>'
                        f'</div></div>',
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