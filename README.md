[README.md](https://github.com/user-attachments/files/22627972/README.md)
# Recomendador de Match — Decision

Aplicação (Streamlit) que **ranqueia candidatos por probabilidade de match** para uma vaga específica.  
O app lê um _dataset mínimo_ de inferência (compactado) ou, em fallback, faz o **merge dos 3 CSVs**: `vaga_vf.csv`, `applicants_vf.csv`, `prospects_vf.csv`.  
O usuário escolhe a vaga e recebe um ranking de candidatos com **probabilidade de match** e um **threshold** ajustável.

## ✨ Entregáveis

- **Repositório (GitHub):** <https://github.com/diegophatanaka80/datathon_fase5>  
- **Aplicação (Streamlit Cloud):** <https://datathonfase5-match-vagas.streamlit.app/>

---

## 🧠 Descrição do projeto

- **Objetivo:** priorizar candidatos com maior probabilidade de “match” para cada vaga, auxiliando o time de negócio na triagem.  
- **Abordagem:** pipeline de pré-processamento + engenharia de features + classificação supervisionada.  
  - O modelo é treinado offline; o app **carrega o modelo treinado** e aplica em tempo de execução.
- **Entrada de dados:**
  - **Preferencial:** `data/processed/app_inference.min.csv.gz` (conjunto mínimo para inferência, já com colunas necessárias).
  - **Fallback:** merge automático de `vaga_vf.csv`, `applicants_vf.csv`, `prospects_vf.csv` (pasta `data/processed`, `data/raw` ou raiz do repo).
  - O **nome da vaga** vem da coluna `título_vaga` (após prefixo no app: `job__título_vaga`).
- **Saída:** ranking com probabilidade de match e botão de **download (CSV)**.

---

## 🧰 Stack utilizada

- **Linguagem:** Python 3.11+
- **App:** [Streamlit](https://streamlit.io/)
- **ML:** scikit-learn (classificadores e pipeline), joblib (serialização)
- **Dados:** pandas, numpy, pyarrow (leitura/escrita)
- **(Opcional)** XGBoost – se a técnica escolhida usar esse estimador

---

## 📁 Estrutura (resumo)

```
.
├─ app/
│  └─ app.py                     # aplicação Streamlit
├─ data/
│  └─ processed/
│     └─ app_inference.min.csv.gz  # dataset mínimo de inferência (preferido pelo app)
├─ models/
│  ├─ recommender_small.joblib   # modelo (compactado) carregado no app
│  ├─ recommender.pkl            # modelo completo (fallback)
│  ├─ recommender_meta.json      # threshold ótimo, lista de features, etc.
│  ├─ utils.py                   # utils usados no treino (expostos p/ joblib.load)
│  └─ tfidf_vectorizer.joblib    # (se aplicável ao seu pipeline)
├─ notebooks/
│  ├─ Projeto_Datathon_Fase5_Módulos.ipynb
│  ├─ Projeto_Datathon_Fase5_Normalização.ipynb
│  └─ Projeto_Datathon_Fase5_Treinamento*.ipynb
└─ src/
   ├─ data_io.py
   ├─ preprocessing.py
   ├─ feature_engineering.py
   ├─ labeling.py
   └─ model_utils.py
```

---

## 🚀 Como rodar o app localmente

1. **Clone o repo**
   ```bash
   git clone https://github.com/diegophatanaka80/datathon_fase5.git
   cd datathon_fase5
   ```

2. **Crie um ambiente virtual e instale as dependências**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   # source .venv/bin/activate   # macOS / Linux

   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Garanta os artefatos**
   - `models/recommender_small.joblib` **ou** `models/recommender.pkl`
   - `models/recommender_meta.json`
   - `data/processed/app_inference.min.csv.gz` (recomendado)
     - Se não existir, o app tenta ler e juntar automaticamente os CSVs `vaga_vf.csv`, `applicants_vf.csv`, `prospects_vf.csv`.

4. **Execute o Streamlit**
   ```bash
   streamlit run app/app.py
   ```

---

## 🧩 Instruções de instalação (versões de bibliotecas)

As versões do ambiente devem **combinar com as do modelo** (para evitar erros de desserialização).  
Utilize este `requirements.txt` (alinhado ao treino final: `sklearn==1.5.0`, `joblib==1.4.2`):

```txt
streamlit==1.38.0
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
joblib==1.4.2
scipy==1.11.4
pyarrow==16.1.0
xgboost==2.1.1
```

> Se re-treinar com outras versões, **alinhe o `requirements.txt` do deploy** às versões usadas no treino.

---

## 🔄 Como treinar o modelo novamente

Você tem **duas opções**:

### Opção A) Via notebooks (recomendado para exploração)

1. **Normalização/Feature engineering**  
   Abra `notebooks/Projeto_Datathon_Fase5_Normalização.ipynb` e gere:
   - o dataset consolidado para treino;  
   - o **dataset mínimo de inferência** `data/processed/app_inference.min.csv.gz` (ex.: `to_csv(..., compression="gzip")`).

2. **Treinamento**  
   Abra `notebooks/Projeto_Datathon_Fase5_Treinamento*.ipynb` e rode:
   - Treino comparando modelos (quatro técnicas) e seleção da melhor.
   - Cálculo do **threshold ótimo** (F1 ou métrica-alvo).
   - Salvamento dos artefatos em `models/`:
     - `recommender_small.joblib` (ou `recommender.pkl`)
     - `recommender_meta.json` (contendo, no mínimo: `best_threshold`, `num_cols`, `cat_cols`)

### Opção B) Via módulo Python (roteiro mínimo)

> Exemplo de script ad-hoc para (re)treinar direto no Python — adapte ao seu pipeline atual:

```python
# Exemplo: treinar e salvar
from src.preprocessing import basic_preprocessing
from src.feature_engineering import make_features
from src.labeling import label_match
from pathlib import Path
import pandas as pd
import numpy as np
import json, joblib

# 1) Carregue sua base de treino (ex.: parquet/CSV consolidado)
df = pd.read_parquet("data/decision_consolidated.parquet")  # ou o seu consolidado

# 2) Pré e features
df = basic_preprocessing(df)
df = make_features(df)

# 3) Targets
df["target_match"] = label_match(df)

# 4) Separe X/y com as colunas usadas no seu treino
num_cols = ["prospect_comment_len","feat_skill_overlap","feat_senioridade","feat_senioridade_gap","feat_ingles_match"]
cat_cols = ["job__nivel_profissional","app__area"]
X = df[num_cols + cat_cols].copy()
y = df["target_match"].astype(int)

# 5) Monte o pipeline (o mesmo usado no notebook)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_cols),
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat_cols)
])

pipe = Pipeline([
    ("pre", pre),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
])

pipe.fit(X, y)

# 6) Salve modelo + meta
Path("models").mkdir(parents=True, exist_ok=True)
joblib.dump(pipe, "models/recommender_small.joblib", compress=3)
with open("models/recommender_meta.json","w",encoding="utf-8") as f:
    json.dump({
        "best_threshold": 0.5,         # ajuste com base na curva PR/F1
        "num_cols": num_cols,
        "cat_cols": cat_cols
    }, f, indent=2, ensure_ascii=False)
```

### (Opcional) Gerar o dataset mínimo do app

Após consolidar e criar features, gere `data/processed/app_inference.min.csv.gz` apenas com as colunas necessárias (`pair_id`, `job_id`, `job__título_vaga`, dados do candidato + features usadas pelo modelo):

```python
from pathlib import Path
import pandas as pd

df = ...  # seu dataframe consolidado pós features
keep = ["pair_id","job_id","job__título_vaga","app__nome","app__email",
        "prospect_comment_len","feat_skill_overlap","feat_senioridade",
        "feat_senioridade_gap","feat_ingles_match","job__nivel_profissional","app__area"]
Path("data/processed").mkdir(parents=True, exist_ok=True)
df[keep].to_csv("data/processed/app_inference.min.csv.gz", index=False, compression="gzip")
```

---

## 🛠️ Troubleshooting (comum em deploy)

- **“Can’t get attribute … on sklearn/… ou utils …”**  
  ➜ Versões diferentes de scikit-learn/joblib entre treino e deploy **ou** módulo auxiliar ausente.  
  - Alinhe as versões (veja `requirements.txt` acima).  
  - Garanta que `models/utils.py` exista (o app já inclui `models/` no `PYTHONPATH`).

- **Modelo treinado com XGBoost** e não carrega no Cloud  
  ➜ Inclua `xgboost==2.1.1` no `requirements.txt`.

---

## 📦 Serialização do modelo

O repositório inclui:
- `models/recommender_small.joblib` (modelo compactado)
- `models/recommender.pkl` (fallback)
- `models/recommender_meta.json` (threshold & colunas)

> Se reentreinar: **substitua** esses três arquivos e mantenha as **mesmas versões** no `requirements.txt`.

---

## 📜 Licença

Projeto acadêmico / demonstrativo. Ajuste conforme a necessidade.
