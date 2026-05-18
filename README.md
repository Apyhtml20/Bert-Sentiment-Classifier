# twitter-sentiment-bert

Analyse de sentiment en NLP utilisant **BERT** pour classifier les émotions dans des tweets et des critiques de films. Le projet explore deux approches : un modèle BERT custom avec PyTorch/Transformers et un modèle pré-entraîné via TensorFlow Hub.

---

## Structure du projet

```
twitter-sentiment-bert/
├── sentiment_analysis.ipynb      # Classification binaire (Positif/Négatif) sur IMDB avec BERT
├── Twitter_analysis.ipynb        # Classification 3 classes sur tweets réels
├── twitter_training.csv          # Dataset Twitter (id, entity, sentiment, text)
├── bert/                         # Modèle BERT SavedModel TensorFlow (local)
│   ├── assets/vocab.txt          # Vocabulaire BERT (30 522 tokens)
│   ├── saved_model.pb            # Graphe TensorFlow sérialisé
│   └── variables/                # Poids entraînés du modèle
├── main.py                       # Point d'entrée du package
└── pyproject.toml                # Dépendances (uv)
```

---

## Modèles utilisés

### 1. BERT via TensorFlow Hub

```python
encoder_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4"
preprocess_url = "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3"
```

Modèle `bert_en_uncased_L-12_H-768_A-12` :
- 12 couches Transformer
- 768 dimensions cachées
- 12 têtes d'attention multi-head

### 2. BERT via HuggingFace + PyTorch

```python
from transformers import BertTokenizer, BertModel
import torch.nn as nn

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased")

class SentimentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(768, 2)  # 768 → 2 classes

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # token [CLS]
        x = self.dropout(pooled)
        return self.fc(x)
```

### 3. DistilBERT pré-fine-tuné (Twitter)

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)
```

---

## Pipeline

```
Texte brut
  → Tokenization BERT (padding + truncation 512 tokens)
  → Encodage BERT (vecteur [CLS] de 768 dims)
  → Dropout (0.2) + Couche Dense (768 → N classes)
  → Softmax → Probabilités par classe
  → argmax → Prédiction finale
```

---

## Datasets

| Dataset | Source | Classes | Taille |
|---|---|---|---|
| IMDB Reviews | HuggingFace `datasets` | Positif / Négatif | 50 000 avis |
| Twitter Sentiment | `twitter_training.csv` | Positif / Neutre / Négatif | Variable |

---

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/Apyhtml20/twitter-sentiment-bert.git
cd twitter-sentiment-bert

# Créer l'environnement virtuel et installer les dépendances
uv sync
```

### Dépendances principales

```toml
tensorflow==2.18.0
tensorflow-hub>=0.16.1
torch>=2.11.0
transformers>=5.6.2
datasets>=4.8.4
```

---

## Utilisation

Ouvrir les notebooks Jupyter dans l'ordre :

1. **`sentiment_analysis.ipynb`** — Entraînement du modèle BERT sur IMDB (classification binaire)
2. **`Twitter_analysis.ipynb`** — Application sur des tweets réels (3 classes)

---

## Résultats

Le modèle prédit une des classes suivantes selon le dataset :

- **IMDB** : `Positif` / `Négatif`
- **Twitter** : `Positif` / `Neutre` / `Négatif`

---

## Technologies

![Python](https://img.shields.io/badge/Python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-FF6F00)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![BERT](https://img.shields.io/badge/BERT-base--uncased-green)
