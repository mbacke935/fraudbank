# Détection de fraude bancaire — Sénégal (Scénario 1)

Application Streamlit : exploration des données, performance du modèle Random Forest et prédiction de fraude sur une transaction saisie manuellement.

## Fichiers

- `app.py` — l'application
- `scenario1_propre.csv` — données nettoyées (exportées depuis le notebook Colab)
- `requirements.txt` — dépendances

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501

## Lancer avec Docker

```bash
docker build -t fraude-app .
docker run -d --name fraude-app -p 8501:8501 fraude-app
```

Puis ouvrir http://localhost:8501

- Arrêter : `docker stop fraude-app`
- Relancer : `docker start fraude-app`
- Reconstruire après une modification : `docker build -t fraude-app . && docker rm -f fraude-app && docker run -d --name fraude-app -p 8501:8501 fraude-app`

## Déployer sur Streamlit Community Cloud (gratuit)

1. Créer un dépôt GitHub et y pousser les 4 fichiers (`app.py`, `scenario1_propre.csv`, `requirements.txt`, `README.md`).
2. Aller sur https://share.streamlit.io et se connecter avec GitHub.
3. Cliquer sur **Create app** → choisir le dépôt, la branche `main` et le fichier `app.py`.
4. Cliquer sur **Deploy**. L'application obtient une URL publique du type `https://<nom>.streamlit.app`.

Chaque `git push` sur le dépôt redéploie automatiquement l'application.
