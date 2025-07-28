# Utilisation d'une image Python officielle légère
FROM python:3.11.7-slim-bookworm

# Définir le répertoire de travail dans le conteneur
WORKDIR /health

# Définir les variables d'environnement
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_NO_CACHE_DIR=off

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier le fichier requirements en premier pour exploiter le cache de Docker
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste des fichiers du projet
COPY . .

EXPOSE 4000

# Commande par défaut au démarrage du conteneur
# Remplacez 'app:app' par le point d'entrée de votre application si nécessaire
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "4000"]
