# 💳 Détection de Fraude Bancaire — ML sur données déséquilibrées

> Projet de classification binaire — Comparaison de 3 modèles de Machine Learning pour détecter automatiquement les transactions frauduleuses sur un dataset réel de cartes bancaires.

---

## 🎯 Problématique

Une fraude non détectée coûte de l'argent, mais une fausse alarme peut bloquer une transaction légitime d'un client innocent. Le dataset est **extrêmement déséquilibré** : les fraudes représentent seulement **0,17%** des transactions. L'accuracy classique est donc inutilisable — un modèle qui prédirait toujours "Normal" aurait 99,83% d'accuracy sans rien avoir appris. Les métriques clés ici sont le **ROC-AUC** et la **Balanced Accuracy**.

---

## 🗂️ Structure du projet

```
📁 fraude_detection/
│
├── 📓 Fraude_Detection_Sarra.ipynb     # Notebook principal — Pipeline complet
├── 🐍 fraude_detection.py              # Script Python
│
├── 📊 creditcard.csv                   # Dataset principal
│
└── 📁 images/
    └── comparaison_modeles.png         # Comparaison ROC-AUC & Balanced Accuracy
```

📥 **Dataset :** [`creditcard.csv`](https://github.com/sarra725/fraud-detection-ml/blob/main/01_data/data_Fraude.csv) — Source : [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud?utm_source=chatgpt.com)

- **284 807 transactions** bancaires anonymisées
- Colonnes `V1`–`V28` : résultat d'une PCA pour anonymisation
- `Amount` : montant de la transaction | `Time` : secondes depuis la 1ère transaction
- `Class` : 0 = Normale, 1 = Fraude

---

## ⚙️ Pipeline ML

### 1️⃣ Analyse exploratoire (EDA)
- Vérification des valeurs manquantes et doublons
- Suppression des colonnes avec >40% de valeurs manquantes
- Visualisation de la distribution très déséquilibrée de la cible

### 2️⃣ Préparation des données
- Standardisation avec `StandardScaler`
- Split **80/20** avec `stratify=y` pour préserver le ratio fraude/normal
- Gestion du déséquilibre via `class_weight='balanced'` (SMOTE écarté pour éviter le biais d'évaluation sur données synthétiques)

### 3️⃣ Optimisation du seuil de décision
- Seuil par défaut de 0.5 remplacé par le seuil **optimal** maximisant le F1-score sur la classe fraude

---

## 🤖 Modèles comparés

| Modèle | Paramètres clés |
|---|---|
| **Logistic Regression** | `class_weight='balanced'`, `max_iter=1000` |
| **Random Forest** | `n_estimators=100`, `max_depth=10`, `class_weight='balanced'` |
| **XGBoost** | `scale_pos_weight` = ratio déséquilibre, `n_estimators=100` |

📄 [`fraude_detection.py`](https://github.com/sarra725/fraud-detection-ml/blob/main/02_scripts/Fraude_Detection.ipynb)

---

## 📊 Résultats & Comparaison

| Modèle | ROC-AUC | Balanced Accuracy |
|:---|:---:|:---:|
| Logistic Regression | 0.9657 | **0.8989** |
| Random Forest | 0.9646 | 0.8789 |
| 🥇 **XGBoost** | **0.9790** | 0.8842 |

> 💡 **Conclusion :** **XGBoost** obtient le meilleur ROC-AUC (0.979), ce qui en fait le modèle le plus performant pour discriminer fraudes et transactions normales. La **Logistic Regression** surprend avec la meilleure Balanced Accuracy (0.899), montrant qu'un modèle simple peut être très compétitif sur ce type de problème.

---

## 📈 Comparaison des Modèles

![Comparaison ROC-AUC et Balanced Accuracy des modèles](https://github.com/sarra725/fraud-detection-ml/blob/main/03_images/comparaison_des_mod%C3%A9les.png)

> *Le graphique de gauche (ROC-AUC) montre la supériorité de XGBoost. Le graphique de droite (Balanced Accuracy) révèle que la Logistic Regression gère mieux l'équilibre entre détection des fraudes et minimisation des fausses alarmes.*

---

## 🔍 Test de Prédiction

Le modèle XGBoost expose une fonction de prédiction sur de nouvelles transactions :

```python
sample = X_test[0:1]
prediction = models["XGBoost"].predict(sample)
proba = models["XGBoost"].predict_proba(sample)[0][1]

# Exemple de sortie :
# Probabilité de fraude : 0.00%
# Prédiction            : Normale
# Vraie valeur          : Normale
# Résultat              : ✓ Correct !
```

---

## 🚀 Installation & Utilisation

### Prérequis

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn tqdm
```

### Lancer le script

```bash
python fraude_detection.py
```

### Lancer le notebook

```bash
jupyter notebook Fraude_Detection_Sarra.ipynb
```

> ⚠️ **Important :** Placer `creditcard.csv` à la racine du projet ou adapter le chemin :
> ```python
> df = pd.read_csv("creditcard.csv")
> ```

---

## 📦 Dépendances

| Bibliothèque | Usage |
|---|---|
| `pandas` / `numpy` | Manipulation des données |
| `scikit-learn` | Preprocessing, modèles ML, métriques |
| `xgboost` | Algorithme XGBoost |
| `imbalanced-learn` | SMOTE (exploré, non retenu) |
| `matplotlib` / `seaborn` | Visualisation des résultats |
| `tqdm` | Barre de progression |

---

## 👩‍💻 Auteure

**Sarra** — Étudiante en Machine Learning  
Projet réalisé dans le cadre d'un cours de classification sur données déséquilibrées.
