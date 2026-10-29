# Bilverdi – prisestimat for brukt bil med maskinlæring

Nettside som estimerer salgsprisen på en brukt bil ut fra merke, modell, årsmodell, kjørelengde, motor m.m.
Prosjektinnlevering i DAT158. Rapport: `reports/rapport.docx`.

## Struktur
- `data/used_cars.csv` – datasett (4 009 annonser, USA)
- `src/features.py` – parsing og feature engineering (delt av trening og app)
- `src/train.py` – rensing, modellsammenligning, tuning, evaluering, figurer, lagrer `models/model.joblib`
- `src/app.py` + `templates/index.html` – Flask-nettside og `/predict`-API
- `reports/` – rapport, figurer, resultater (`results.json`, `model_comparison.csv`)

## Kjør lokalt
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/train.py        # reproduserer resultatene (tar ca. 1 min)
python src/app.py          # åpne http://localhost:5000
```

## Deploy
- Docker: `docker build -t bilverdi . && docker run -p 5000:5000 bilverdi`
- Render/Railway/Fly.io: koble GitHub-repoet, start-kommando `gunicorn --chdir src app:app`.

## Kilder
Datasett: Kaggle «Used Car Price Prediction Dataset». Verktøy: scikit-learn, Flask, pandas.
Koden er utviklet med hjelp av Claude (Anthropic) som AI-kodeassistent.
