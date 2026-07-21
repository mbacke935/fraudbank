# Importation de Streamlit pour l'interface web
import streamlit as st
# Importation de Pandas pour la manipulation des données
import pandas as pd
# Importation de Matplotlib pour les graphiques
import matplotlib.pyplot as plt
# Importation de LinearSegmentedColormap pour créer des dégradés
from matplotlib.colors import LinearSegmentedColormap
# Importation de Patch pour construire des légendes personnalisées (catégories)
from matplotlib.patches import Patch
# Importation de Line2D pour construire des légendes personnalisées (tailles de bulles)
from matplotlib.lines import Line2D
# Importation de FuncFormatter pour formater les axes (pourcentages)
from matplotlib.ticker import FuncFormatter
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
# Accent secondaire froid (cyan, reste dans la famille du bleu — info/validation)
CYAN = "#22D3EE"
# Accent secondaire froid (violet, toujours dans un registre bleu-froid — vigilance)
VIOLET = "#A78BFA"
# Seul accent chaud de toute l'interface, réservé exclusivement à l'alerte la plus grave
ROSE = "#F43F5E"

# Dictionnaire de couleurs pour les statuts de transactions
# Un seul repère chaud (Fraude) ressort volontairement sur fond de palette froide
STATUT = {
    "Normal": CYAN,    # Cyan : validation, reste dans la famille du bleu
    "Suspect": VIOLET, # Violet : vigilance, toujours froid
    "Fraude": ROSE      # Rose/rouge : seule alerte chaude de l'app
}

# Dictionnaire de couleurs pour les deux grandes familles de variables du modèle
CATEGORIE_FEATURE = {
    "Montant": "Transactionnelle",
    "Localisation_enc": "Transactionnelle",
    "Type de transaction_enc": "Transactionnelle",
    "Heure": "Temporelle",
    "Jour_semaine": "Temporelle",
    "Mois": "Temporelle",
}
# Couleur associée à chaque famille de variables (cohérente avec le reste de la palette)
COULEUR_CATEGORIE = {"Transactionnelle": ACCENT, "Temporelle": VIOLET}

# Dégradé séquentiel pour la matrice de confusion (marine -> indigo -> cyan)
SEQUENTIEL = ["#101C3B", "#182A55", "#233F7E", "#3557B0", "#4C6FFF", "#7C97FF", "#A5F3FC"]

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


# Fonction pour tracer la répartition des statuts sous forme de donut avec légende
def fig_target(df):
    # Sélection des catégories présentes dans le dataset
    ordre = [c for c in ("Normal", "Suspect", "Fraude") if c in df["Target"].unique()]
    # Comptage des valeurs selon l'ordre choisi
    valeurs = df["Target"].value_counts().reindex(ordre)
    # Total des opérations, affiché au centre du donut
    total = valeurs.sum()
    # Création de la figure avec fond sombre
    fig, ax = plt.subplots(figsize=(6.6, 3.9), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    # Couleurs associées à chaque statut, dans l'ordre fixe
    couleurs = [STATUT[c] for c in ordre]
    # Tracé du donut : anneau fin, séparé par de légers liserés couleur du fond
    parts, _ = ax.pie(
        valeurs, colors=couleurs, startangle=90, counterclock=False,
        wedgeprops=dict(width=0.42, edgecolor=SURFACE, linewidth=3), radius=1.0
    )
    # Nombre total inscrit au centre de l'anneau
    ax.text(0, 0.07, fmt(total), ha="center", va="center", fontsize=19, fontweight="bold", color=TEXTE)
    # Légende "transactions" sous le total
    ax.text(0, -0.16, "transactions", ha="center", va="center", fontsize=8.5, color=TEXTE_ATTENUE)
    # Légende externe avec la part exacte de chaque statut
    etiquettes = [f"{c}   ·   {v/total:.1%}" for c, v in zip(ordre, valeurs)]
    ax.legend(
        parts, etiquettes, loc="center left", bbox_to_anchor=(1.05, 0.5),
        frameon=False, fontsize=9.5, labelcolor=TEXTE, handlelength=1.1, handleheight=1.4,
        borderaxespad=0
    )
    # Anneau parfaitement circulaire
    ax.set_aspect("equal")
    # Ajustement automatique des marges
    fig.tight_layout()
    # Renvoi du graphique
    return fig


# Fonction pour tracer le Top 10 des villes sous forme de graphique en sucettes (lollipop)
def fig_villes(df):
    # Extraction des 10 villes les plus fréquentes
    villes = df["Localisation"].value_counts().head(10).sort_values()
    # Création de la figure de base sans traits
    fig, ax = _base((6.4, 4.1))
    # Tige fine reliant l'axe à chaque valeur
    ax.hlines(y=range(len(villes)), xmin=0, xmax=villes.values, color=BORDURE, linewidth=1.6, zorder=1)
    # Point plein au bout de chaque tige
    ax.scatter(villes.values, range(len(villes)), s=115, color=CYAN, zorder=3, edgecolor=SURFACE, linewidth=1.4)
    # Affichage du nombre exact à côté de chaque point
    for i, v in enumerate(villes.values):
        ax.annotate(fmt(v), (v, i), va="center", fontsize=9, fontweight="600", color=TEXTE, xytext=(10, 0), textcoords="offset points")
    # Étiquettes des villes sur l'axe Y
    ax.set_yticks(range(len(villes)))
    ax.set_yticklabels(villes.index, color=TEXTE, fontsize=9.5)
    # Masquage des graduations X
    ax.set_xticks([])
    # Définition de la limite horizontale
    ax.set_xlim(0, villes.max() * 1.2)
    # Ajustement des marges
    fig.tight_layout()
    # Renvoi de la figure
    return fig


# Fonction pour tracer, sous forme de bulles, le risque et le volume de chaque canal de transaction
def fig_canaux(df):
    # Agrégation par canal : montant moyen, volume et taux de fraude observé
    stats = (
        df.groupby("Type de transaction")
        .agg(montant_moyen=("Montant", "mean"), volume=("Montant", "size"), taux_fraude=("Target", lambda s: (s == "Fraude").mean()))
        .reset_index()
    )
    # Dégradé du risque : cyan (faible) -> violet -> rose (élevé), cohérent avec les statuts
    cmap_risque = LinearSegmentedColormap.from_list("risque_canal", [CYAN, VIOLET, ROSE])
    # Normalisation du taux de fraude pour le mappage de couleur
    ecart = stats["taux_fraude"].max() - stats["taux_fraude"].min()
    normalise = (stats["taux_fraude"] - stats["taux_fraude"].min()) / (ecart if ecart > 0 else 1)
    # Taille des bulles proportionnelle au volume de transactions du canal
    tailles = 320 + (stats["volume"] / stats["volume"].max()) * 2400

    # Création du graphique de base sans traits
    fig, ax = _base((9.6, 4.4))
    # Nuage de points : position = montant moyen / taux de fraude, taille = volume, couleur = niveau de risque
    dispersion = ax.scatter(
        stats["montant_moyen"], stats["taux_fraude"], s=tailles,
        c=normalise, cmap=cmap_risque, alpha=0.88, edgecolor=SURFACE, linewidth=1.6, zorder=3
    )
    # Nom du canal inscrit au centre de chaque bulle
    for _, ligne in stats.iterrows():
        ax.annotate(
            ligne["Type de transaction"], (ligne["montant_moyen"], ligne["taux_fraude"]),
            ha="center", va="center", fontsize=8.5, fontweight="700", color=TEXTE, zorder=4
        )
    # Étiquette de l'axe X
    ax.set_xlabel("Montant moyen (FCFA)", fontsize=9.5, color=TEXTE_ATTENUE, labelpad=10)
    # Étiquette de l'axe Y
    ax.set_ylabel("Taux de fraude", fontsize=9.5, color=TEXTE_ATTENUE, labelpad=10)
    # Formatage de l'axe Y en pourcentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.0%}"))
    # Réactivation discrète des graduations Y pour lire l'échelle du risque
    ax.tick_params(axis="y", length=0, labelsize=8.5)
    ax.tick_params(axis="x", length=0, labelsize=8.5)
    # Marges autour du nuage de points
    ax.set_xlim(stats["montant_moyen"].min() * 0.7, stats["montant_moyen"].max() * 1.3)
    marge_y = max(stats["taux_fraude"].max() * 0.35, 0.02)
    ax.set_ylim(max(0, stats["taux_fraude"].min() - marge_y), stats["taux_fraude"].max() + marge_y)
    # Légende de taille : le volume de transactions représenté par deux bulles de référence
    reperes_taille = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TEXTE_ATTENUE, alpha=0.45, markersize=7, label="Faible volume"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=TEXTE_ATTENUE, alpha=0.45, markersize=15, label="Fort volume"),
    ]
    ax.legend(handles=reperes_taille, loc="upper left", frameon=False, fontsize=8.5, labelcolor=TEXTE, handletextpad=1.1, borderaxespad=0)
    # Barre de couleur : légende du niveau de risque (taux de fraude)
    barre_couleur = fig.colorbar(dispersion, ax=ax, fraction=0.045, pad=0.02)
    barre_couleur.ax.tick_params(colors=TEXTE_ATTENUE, length=0, labelsize=0)
    barre_couleur.outline.set_visible(False)
    barre_couleur.set_label("Niveau de risque (faible → élevé)", fontsize=8.5, color=TEXTE_ATTENUE)
    # Ajustement des marges
    fig.tight_layout()
    # Renvoi du graphique
    return fig


# Fonction pour tracer l'évolution mensuelle des transactions, par statut
def fig_evolution(df):
    # Copie de sécurité des données
    tmp = df.dropna(subset=["Date"]).copy()
    # Regroupement par mois calendaire
    tmp["Periode"] = tmp["Date"].dt.to_period("M").astype(str)
    # Comptage croisé mois x statut
    pivot = tmp.groupby(["Periode", "Target"]).size().unstack(fill_value=0)
    # Ordre fixe des statuts pour un code couleur stable
    ordre = [c for c in ("Normal", "Suspect", "Fraude") if c in pivot.columns]
    pivot = pivot[ordre].sort_index()

    # Création du graphique de base sans traits
    fig, ax = _base((9.6, 3.9))
    # Une courbe par statut, remplissage léger sous la courbe pour la lisibilité
    for statut in ordre:
        ax.plot(pivot.index, pivot[statut], marker="o", markersize=4.5, linewidth=2.2, color=STATUT[statut], label=statut, zorder=3)
        ax.fill_between(pivot.index, pivot[statut], color=STATUT[statut], alpha=0.07, zorder=1)
    # Étiquettes du temps en abscisse, légèrement inclinées
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.index, color=TEXTE_ATTENUE, fontsize=8.5, rotation=35, ha="right")
    # Masquage de l'axe Y (les valeurs exactes comptent moins que la tendance)
    ax.set_yticks([])
    # Légende horizontale au-dessus du graphique, une entrée par statut
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, labelcolor=TEXTE, ncol=len(ordre), bbox_to_anchor=(0, 1.12))
    # Ajustement des marges
    fig.tight_layout()
    # Renvoi du graphique
    return fig


# Fonction pour afficher la matrice de confusion
def fig_confusion(cm, classes):
    # Création d'une palette dégradée bleu -> sarcelle
    cmap = LinearSegmentedColormap.from_list("marine_indigo", SEQUENTIEL)
    # Création de la figure
    fig, ax = plt.subplots(figsize=(5.4, 4.4), facecolor=SURFACE)
    # Couleur de fond
    ax.set_facecolor(SURFACE)
    # Affichage de la matrice sous forme d'image
    image = ax.imshow(cm, cmap=cmap)
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
    # Barre de couleur : légende du nombre de cas représenté par chaque teinte
    barre_couleur = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    barre_couleur.ax.tick_params(colors=TEXTE_ATTENUE, length=0, labelsize=8)
    barre_couleur.outline.set_visible(False)
    barre_couleur.set_label("Nombre de cas", fontsize=8.5, color=TEXTE_ATTENUE)
    # Ajustement de la disposition
    fig.tight_layout()
    # Renvoi de la matrice visuelle
    return fig


# Fonction pour tracer l'importance des variables, regroupées par famille (temporelle / transactionnelle)
def fig_importance(modele, features):
    # Calcul et tri des importances de variables
    imp = pd.Series(modele.feature_importances_, index=features).sort_values()
    # Dictionnaire de traduction propre des noms de colonnes
    noms = {"Montant": "Montant", "Heure": "Heure", "Jour_semaine": "Jour de la semaine", "Mois": "Mois", "Localisation_enc": "Localisation", "Type de transaction_enc": "Type de transaction"}
    # Couleur de chaque barre selon la famille de la variable
    couleurs = [COULEUR_CATEGORIE[CATEGORIE_FEATURE.get(f, "Transactionnelle")] for f in imp.index]
    # Création du graphique de base sans traits
    fig, ax = _base((5.6, 3.8))
    # Création des barres d'importance
    ax.barh([noms.get(f, f) for f in imp.index], imp.values, height=0.45, color=couleurs, zorder=2)
    # Affichage du score exact à côté de la barre
    for i, v in enumerate(imp.values):
        ax.annotate(f"{v:.3f}", (v, i), va="center", fontsize=9, color=TEXTE, xytext=(6, 0), textcoords="offset points")
    # Masquer les graduations X
    ax.set_xticks([])
    # Ajuster la largeur max, avec marge pour la légende
    ax.set_xlim(0, imp.max() * 1.25)
    # Légende des deux familles de variables
    reperes_categories = [Patch(color=couleur, label=cat) for cat, couleur in COULEUR_CATEGORIE.items()]
    ax.legend(handles=reperes_categories, loc="lower right", frameon=False, fontsize=8.5, labelcolor=TEXTE, borderaxespad=0)
    # Ajuster le layout
    fig.tight_layout()
    # Renvoi de la figure
    return fig


# Fonction pour tracer les probabilités estimées par le modèle pour la transaction simulée
def fig_probas(probas, classes):
    # Création du graphique de base sans traits
    fig, ax = _base((6.0, 2.4))
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
    ax.set_xlim(0, 1.18)
    # Ligne verticale de repère au seuil de décision (50%)
    ax.axvline(0.5, color=TEXTE_ATTENUE, linestyle="--", linewidth=1, alpha=0.55, zorder=1)
    ax.annotate("seuil 50%", (0.5, len(classes) - 0.55), fontsize=7.5, color=TEXTE_ATTENUE, ha="center", va="top")
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
    background: rgba(34,211,238,0.08);
    border-color: rgba(34,211,238,0.4);
    border-left: 4px solid #22D3EE;
    color: #22D3EE;
}

.diagnostic.suspect {
    background: rgba(167,139,250,0.08);
    border-color: rgba(167,139,250,0.4);
    border-left: 4px solid #A78BFA;
    color: #A78BFA;
}

.diagnostic.fraude {
    background: rgba(244,63,94,0.1);
    border-color: rgba(244,63,94,0.45);
    border-left: 4px solid #F43F5E;
    color: #F43F5E;
}

/* Sous-panneau imbriqué (barre de filtres) : fond distinct, plus clair que la carte parente */
div[data-testid="stVerticalBlockBorderWrapper"] div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #16264C !important;
    border-color: #2B3B6B !important;
}

/* Étiquette compacte au-dessus d'un filtre */
.mini-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #8CA0C7;
}

/* Rangée de mini-statistiques reflétant la sélection filtrée du tableau */
.stats-filtre {
    display: flex;
    gap: 12px;
    margin: 1rem 0 1.1rem 0;
    flex-wrap: wrap;
}

.mini-stat {
    flex: 1;
    min-width: 170px;
    background: #16264C;
    border: 1px solid #2B3B6B;
    border-left: 3px solid var(--accent-mini, #4C6FFF);
    border-radius: 12px;
    padding: 12px 16px;
}

.mini-stat .mini-stat-label {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #8CA0C7;
    display: block;
    margin-bottom: 4px;
}

.mini-stat .mini-stat-valeur {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #EAF0FF;
}

.mini-stat .mini-stat-valeur .mini-stat-total {
    color: #8CA0C7;
    font-weight: 500;
    font-size: 0.82rem;
    font-family: 'Inter', sans-serif;
}

/* Multiselect du filtre par statut : puces colorées et cohérentes avec le code couleur */
div[data-baseweb="tag"] {
    border-radius: 8px !important;
    font-weight: 600 !important;
    border: none !important;
}

div[data-baseweb="tag"]:nth-of-type(3n+1) { background: rgba(34,211,238,0.18) !important; color: #22D3EE !important; }
div[data-baseweb="tag"]:nth-of-type(3n+2) { background: rgba(167,139,250,0.18) !important; color: #A78BFA !important; }
div[data-baseweb="tag"]:nth-of-type(3n+3) { background: rgba(244,63,94,0.2) !important; color: #F43F5E !important; }

div[data-baseweb="tag"] svg { fill: currentColor !important; }

/* Menu déroulant du multiselect, rendu en portail : on l'aligne sur le thème sombre */
ul[data-baseweb="menu"] {
    background: #101C3B !important;
    border: 1px solid #22315C !important;
}

ul[data-baseweb="menu"] li {
    color: #EAF0FF !important;
}

ul[data-baseweb="menu"] li:hover {
    background: rgba(76,111,255,0.15) !important;
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
        ["Exploration des Données", "Performance du Modèle", "Simuler une Transaction"]
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
        # Bloc de gauche : donut de répartition des statuts
        with gauche, st.container(border=True):
            carte_titre("Répartition des Classes", "Part de chaque statut, en anneau · légende avec pourcentages")
            st.pyplot(fig_target(df), use_container_width=True)

        # Bloc de droite : sucettes des villes les plus actives
        with droite, st.container(border=True):
            carte_titre("Top 10 Villes Actives", "Volume de transactions par localisation")
            st.pyplot(fig_villes(df), use_container_width=True)

        # Conteneur pour le risque et le volume par canal, sous forme de bulles
        with st.container(border=True):
            carte_titre(
                "Risque et Volume par Canal de Paiement",
                "Position = montant moyen & taux de fraude · Taille = volume de transactions · Couleur = niveau de risque"
            )
            st.pyplot(fig_canaux(df), use_container_width=True)

        # Conteneur pour l'évolution mensuelle des transactions
        with st.container(border=True):
            carte_titre("Évolution Mensuelle des Transactions", "Nombre d'opérations par mois, par statut")
            st.pyplot(fig_evolution(df), use_container_width=True)

        # Conteneur pour le tableau de données
        with st.container(border=True):
            carte_titre("Aperçu du Dataset Cleaned", "Explorez, filtrez et inspectez chaque transaction")

            # Ordre fixe des statuts, garantit un code couleur stable pour les puces
            ordre_statuts = [c for c in ("Normal", "Suspect", "Fraude") if c in df["Target"].unique()]

            # Sous-panneau de filtres, fond distinct de la carte et du tableau
            with st.container(border=True):
                f_col1, f_col2 = st.columns([1.3, 1])
                with f_col1:
                    st.markdown('<span class="mini-label">Filtrer par statut</span>', unsafe_allow_html=True)
                    # Filtre par statut : sélection multiple, ordre fixe pour les couleurs
                    filtre_statut = st.multiselect(
                        "Filtrer par statut",
                        options=ordre_statuts,
                        default=ordre_statuts,
                        placeholder="Choisir un statut",
                        label_visibility="collapsed"
                    )
                with f_col2:
                    st.markdown('<span class="mini-label">Rechercher une ville</span>', unsafe_allow_html=True)
                    # Recherche textuelle sur la ville
                    recherche_ville = st.text_input(
                        "Rechercher une ville",
                        placeholder="Ex: Dakar, Thiès...",
                        label_visibility="collapsed"
                    )

            # Application des filtres
            df_filtre = df[df["Target"].isin(filtre_statut)]
            if recherche_ville:
                df_filtre = df_filtre[df_filtre["Localisation"].str.contains(recherche_ville, case=False, na=False)]

            # Calcul des indicateurs sur la sélection filtrée actuelle
            taux_fraude_filtre = (df_filtre["Target"] == "Fraude").mean() if len(df_filtre) else 0
            montant_moyen_filtre = df_filtre["Montant"].mean() if len(df_filtre) else 0

            # Rangée de mini-cartes chiffrées qui réagissent au filtre, en remplacement de la simple phrase de comptage
            st.markdown(
                '<div class="stats-filtre">'
                f'<div class="mini-stat" style="--accent-mini:{ACCENT}">'
                '<span class="mini-stat-label">Opérations affichées</span>'
                f'<span class="mini-stat-valeur">{fmt(len(df_filtre))} '
                f'<span class="mini-stat-total">/ {fmt(len(df))} au total</span></span>'
                '</div>'
                f'<div class="mini-stat" style="--accent-mini:{CYAN}">'
                '<span class="mini-stat-label">Montant moyen (sélection)</span>'
                f'<span class="mini-stat-valeur">{fmt(montant_moyen_filtre)} FCFA</span>'
                '</div>'
                f'<div class="mini-stat" style="--accent-mini:{ROSE}">'
                '<span class="mini-stat-label">Taux de fraude (sélection)</span>'
                f'<span class="mini-stat-valeur">{taux_fraude_filtre:.1%}</span>'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            # Tri des transactions les plus récentes en premier, et ajout d'une pastille de couleur devant le statut
            pastille = {"Normal": "🔵", "Suspect": "🟣", "Fraude": "🔴"}
            df_affiche = df_filtre.sort_values("Date", ascending=False).copy()
            df_affiche["Target"] = df_affiche["Target"].map(lambda s: f"{pastille.get(s, '')} {s}")

            # Affichage du tableau interactif stylisé
            st.dataframe(
                df_affiche,
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
        # Matrice de confusion, avec légende de couleur (nombre de cas)
        with gauche, st.container(border=True):
            carte_titre("Matrice de Confusion", "Vrais/faux positifs · couleur = nombre de cas")
            st.pyplot(fig_confusion(metriques["matrice"], le_target.classes_), use_container_width=True)

        # Importance des variables, regroupées par famille avec légende
        with droite, st.container(border=True):
            carte_titre("Importance des Features", "Poids décisionnel par variable, groupé par famille")
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
                    # Graphique des probabilités, avec seuil de décision repère
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