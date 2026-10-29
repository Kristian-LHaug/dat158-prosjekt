"""Trener og evaluerer modeller for prisestimering av brukt bil."""
import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from features import build_features, CAT, NUM

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

raw = pd.read_csv(ROOT / "data/used_cars.csv")
raw["price"] = raw["price"].str.replace(r"[^\d]", "", regex=True).astype(float)
raw["milage"] = raw["milage"].str.replace(r"[^\d]", "", regex=True).astype(float)

# Rens: fjern urealistiske priser og de dyreste 0,5 % (ekstreme outliere)
n0 = len(raw)
raw = raw[(raw.price >= 2000) & (raw.price <= raw.price.quantile(0.995))].reset_index(drop=True)
print(f"Fjernet {n0-len(raw)} rader som outliere; {len(raw)} igjen")

X, y = build_features(raw), raw["price"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

lin_pre = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5), CAT),
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM)])
tree_pre = ColumnTransformer([
    ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, min_frequency=5,
                           encoded_missing_value=-1), CAT),
    ("num", "passthrough", NUM)])
cat_mask = [True] * len(CAT) + [False] * len(NUM)

def logged(est):  # trener på log(pris) fordi prisfordelingen er høyreskjev
    return TransformedTargetRegressor(est, func=np.log1p, inverse_func=np.expm1)

models = {
    "Baseline (median)": Pipeline([("p", lin_pre), ("m", DummyRegressor(strategy="median"))]),
    "Ridge": Pipeline([("p", lin_pre), ("m", logged(RidgeCV(alphas=np.logspace(-2, 3, 20))))]),
    "Random Forest": Pipeline([("p", tree_pre), ("m", logged(RandomForestRegressor(
        n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED)))]),
    "Gradient Boosting": Pipeline([("p", tree_pre), ("m", logged(HistGradientBoostingRegressor(
        categorical_features=cat_mask, random_state=SEED)))]),
}

cv = KFold(5, shuffle=True, random_state=SEED)
results = {}
for name, m in models.items():
    cvs = -cross_val_score(m, Xtr, ytr, cv=cv, scoring="neg_mean_absolute_error")
    results[name] = {"cv_mae": float(cvs.mean())}
    print(name, "CV MAE", round(cvs.mean()))

search = RandomizedSearchCV(
    models["Gradient Boosting"],
    {"m__regressor__learning_rate": [0.03, 0.05, 0.08, 0.1],
     "m__regressor__max_iter": [200, 400, 600],
     "m__regressor__max_leaf_nodes": [15, 31, 63],
     "m__regressor__min_samples_leaf": [10, 20, 40],
     "m__regressor__l2_regularization": [0, 0.1, 1.0]},
    n_iter=20, cv=cv, scoring="neg_mean_absolute_error", random_state=SEED, n_jobs=-1)
search.fit(Xtr, ytr)
print("Beste parametre:", search.best_params_)
models["Gradient Boosting (tunet)"] = search.best_estimator_
results["Gradient Boosting (tunet)"] = {"cv_mae": float(-search.best_score_)}

preds = {}
for name, m in models.items():
    if name != "Gradient Boosting (tunet)":
        m.fit(Xtr, ytr)
    p = m.predict(Xte); preds[name] = p
    results[name].update(
        mae=float(mean_absolute_error(yte, p)), medae=float(median_absolute_error(yte, p)),
        rmse=float(np.sqrt(mean_squared_error(yte, p))), r2=float(r2_score(yte, p)),
        mape=float(np.mean(np.abs(yte - p) / yte) * 100),
        within_20pct=float(np.mean(np.abs(yte - p) / yte <= 0.2) * 100))
res = pd.DataFrame(results).T.round(2); print(res.to_string())

best_name = "Gradient Boosting (tunet)"
best, pbest = models[best_name], preds[best_name]

fig_dir = ROOT / "reports/figures"
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(raw.price, bins=50, color="#4c72b0"); ax[0].set_title("Prisfordeling (USD)")
ax[1].hist(np.log10(raw.price), bins=50, color="#55a868"); ax[1].set_title("log10(pris)")
plt.tight_layout(); plt.savefig(fig_dir / "pris_fordeling.png", dpi=130); plt.close()

plt.figure(figsize=(5, 5)); plt.scatter(yte, pbest, s=8, alpha=.5)
lim = [0, max(yte.max(), pbest.max())]; plt.plot(lim, lim, "r--")
plt.xlabel("Faktisk pris (USD)"); plt.ylabel("Predikert pris (USD)"); plt.title("Predikert vs. faktisk (test)")
plt.tight_layout(); plt.savefig(fig_dir / "pred_vs_faktisk.png", dpi=130); plt.close()

pi = permutation_importance(best, Xte, yte, n_repeats=8, random_state=SEED,
                            scoring="neg_mean_absolute_error", n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
imp.plot.barh(figsize=(6, 4), color="#c44e52"); plt.title("Permutation importance (økning i MAE, USD)")
plt.tight_layout(); plt.savefig(fig_dir / "feature_importance.png", dpi=130); plt.close()

err = pd.DataFrame({"brand": Xte.brand, "age": Xte.age, "price": yte, "pred": pbest})
err["ape"] = (err.price - err.pred).abs() / err.price * 100
by_brand = err.groupby("brand").agg(n=("ape", "size"), mape=("ape", "mean")).query("n>=20").sort_values("mape")
err["age_bin"] = pd.cut(err.age, [-1, 3, 7, 12, 20, 60], labels=["0-3", "4-7", "8-12", "13-20", "21+"])
by_age = err.groupby("age_bin", observed=True).agg(n=("ape", "size"), mape=("ape", "mean"))
err["pb"] = pd.qcut(err.price, 4, labels=["Q1 (billigst)", "Q2", "Q3", "Q4 (dyrest)"])
by_price = err.groupby("pb", observed=True).agg(n=("ape", "size"), mape=("ape", "mean"))
print(by_age.round(1), by_price.round(1), by_brand.round(1), sep="\n")
fig, ax = plt.subplots(figsize=(6, 3.5)); by_age.mape.plot.bar(ax=ax, color="#8172b2")
ax.set_ylabel("MAPE (%)"); ax.set_xlabel("Alder (år)"); ax.set_title("Relativ feil per aldersgruppe")
plt.tight_layout(); plt.savefig(fig_dir / "feil_per_alder.png", dpi=130); plt.close()

final = search.best_estimator_.fit(X, y)   # sluttmodell trenes på alle data
lr = np.log(yte.values) - np.log(pbest)
meta = {"model": best_name, "q10": float(np.quantile(lr, .1)), "q90": float(np.quantile(lr, .9)),
        "brands": sorted(raw.brand.unique()),
        "models_by_brand": {b: sorted(g.model.str.split().str[0].str.upper().unique())
                            for b, g in raw.groupby("brand")},
        "fuel_types": sorted(X.fuel_type.unique()), "trans_types": sorted(X.trans_type.unique()),
        "colors": sorted(X.ext_col.value_counts().head(15).index),
        "int_colors": sorted(X.int_col.value_counts().head(10).index),
        "defaults": {"hp": float(X.hp.median()), "liters": float(X.liters.median()),
                     "cylinders": float(X.cylinders.median())}}
joblib.dump(final, ROOT / "models/model.joblib")
(ROOT / "models/meta.json").write_text(json.dumps(meta, indent=1))
(ROOT / "reports/results.json").write_text(json.dumps(
    {"results": results, "n_rows": len(raw), "n_removed": n0 - len(raw), "best_params": search.best_params_,
     "mape_by_age": by_age.mape.round(1).to_dict(), "mape_by_price": by_price.mape.round(1).to_dict(),
     "mape_by_brand_best": by_brand.head(3).mape.round(1).to_dict(),
     "mape_by_brand_worst": by_brand.tail(3).mape.round(1).to_dict(),
     "importance": imp.round(0).to_dict(), "missing": X.isna().mean().round(3).to_dict()}, indent=1, default=str))
res.to_csv(ROOT / "reports/model_comparison.csv")
