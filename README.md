# Health Tech Assistant Backend

## Description
Backend FastAPI pour l'assistant de santé utilisant l'IA pour fournir des descriptions médicales et répondre aux questions des patients.

## Fonctionnalités
- **Description patient** : Génère une description médicale empathique basée sur les données du patient
- **Chat médical** : Répond aux questions des patients avec mémoire de conversation
- **Synthèse vocale** : Convertit le texte en audio pour une meilleure accessibilité
- **Accès par summary_id** : Authentification simple via ID de résumé

## Installation

1. Installer les dépendances :
```bash
pip install -r requirements.txt
```

2. Placer votre fichier `clinical_summaries.csv` dans le dossier backend

3. Lancer le serveur :
```bash
python main.py
```

Le serveur sera accessible sur `http://localhost:4000`

## Endpoints

### GET /
- **Description** : Message de bienvenue
- **Réponse** : `{"message": "Welcome to the Health Tech Assistant API"}`

### POST /patient-description
- **Description** : Obtient une description médicale du patient
- **Body** :
```json
{
  "summary_id": "PAT001",
  "language": "french"
}
```

### POST /chat-response
- **Description** : Répond aux questions du patient
- **Body** :
```json
{
  "summary_id": "PAT001",
  "question": "Que dois-je faire pour ma tension ?"
}
```

### POST /text-to-speech
- **Description** : Convertit le texte en audio
- **Body** :
```json
{
  "text": "Votre tension est normale",
  "language": "french"
}
```

### GET /health
- **Description** : Vérification de l'état du serveur
- **Réponse** : `{"status": "healthy", "data_loaded": true}`

## Structure des données CSV

Le fichier `clinical_summaries.csv` doit contenir les colonnes suivantes :
- `summary_id` : Identifiant unique du patient
- `patient_name` : Nom du patient
- `age` : Âge
- `gender` : Sexe (M/F)
- `temperature` : Température corporelle
- `blood_pressure` : Tension artérielle
- `heart_rate` : Rythme cardiaque
- `diagnosis` : Diagnostic
- `symptoms` : Symptômes
- `treatment` : Traitement
- `notes` : Notes additionnelles

## Configuration

- **Port** : 4000
- **CORS** : Configuré pour localhost:3000 et localhost:8080
- **Modèle IA** : Llama3-Med42-8B via HuggingFace
- **Base de données** : SQLite pour la mémoire des conversations

## Sécurité

⚠️ **Important** : Remplacez la clé API HuggingFace dans le code par votre propre clé avant la production.
