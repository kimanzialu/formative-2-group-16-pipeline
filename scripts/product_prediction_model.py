"""
Predict the product category a customer will purchase, using merged
transaction history + social media profile data.

Design decisions
 - Key harmonisation: transactions.customer_id_legacy (int) -> 'A'+id == social.customer_id_new
 - Social file is many rows/customer -> aggregated to ONE row per customer BEFORE joining
   (avoids duplicating transactions / inflating the sample).
 - Target = product_category of the transaction.
 - Leakage control: purchase_amount & customer_rating of the *current* transaction are
   outcomes of the purchase, not known when a customer visits a social page, so they are NOT
   used directly. Instead we use time-aware purchase HISTORY (only earlier transactions).
 - Evaluation is time-ordered (expanding-window), matching real deployment.
"""
import pandas as pd, numpy as np, joblib, warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA, MODELS = ROOT / "data", ROOT / "models"

t = pd.read_csv(DATA / "customer_transactions.csv", parse_dates=["purchase_date"])
s = pd.read_csv(DATA / "customer_social_profiles.csv")

# ---------- 1. harmonise key + aggregate social profile to customer level ----------
t["customer_id"] = "A" + t["customer_id_legacy"].astype(str)
s = s.rename(columns={"customer_id_new": "customer_id"})
sent = pd.crosstab(s.customer_id, s.review_sentiment, normalize="index").add_prefix("sent_share_")
plat = pd.crosstab(s.customer_id, s.social_media_platform, normalize="index").add_prefix("plat_share_")
agg = s.groupby("customer_id").agg(
    social_n_profiles=("social_media_platform", "size"),
    social_n_platforms=("social_media_platform", "nunique"),
    engagement_mean=("engagement_score", "mean"),
    engagement_max=("engagement_score", "max"),
    interest_mean=("purchase_interest_score", "mean"),
    interest_max=("purchase_interest_score", "max"),
)
soc = agg.join(sent).join(plat)
soc["sentiment_score"] = soc["sent_share_Positive"] - soc["sent_share_Negative"]
soc["top_platform"] = plat.idxmax(axis=1).str.replace("plat_share_", "")
soc = soc.reset_index()

# ---------- 2. merge (left join keeps every transaction) ----------
df = t.merge(soc, on="customer_id", how="left").sort_values(["purchase_date", "transaction_id"]).reset_index(drop=True)
df["has_social_profile"] = df["social_n_profiles"].notna().astype(int)
df["top_platform"] = df["top_platform"].fillna("Unknown")
assert len(df) == len(t), "merge duplicated rows"

# ---------- 3. time-aware purchase-history features (strictly prior transactions) ----------
cats = sorted(df.product_category.unique())
g = df.groupby("customer_id")
df["prior_txn_count"] = g.cumcount()
df["prior_avg_amount"] = g["purchase_amount"].transform(lambda x: x.shift().expanding().mean())
df["prior_avg_rating"] = g["customer_rating"].transform(lambda x: x.shift().expanding().mean())
df["days_since_last"] = g["purchase_date"].diff().dt.days
df["last_category"] = g["product_category"].shift().fillna("None")
for c in cats:
    ind = (df.product_category == c).astype(int)
    prior = ind.groupby(df.customer_id).transform(lambda x: x.shift().cumsum())
    df[f"prior_share_{c}"] = (prior / df["prior_txn_count"].replace(0, np.nan))
df["dow"] = df.purchase_date.dt.dayofweek
df["month"] = df.purchase_date.dt.month

df.to_csv(DATA / "merged_dataset.csv", index=False)

# ---------- 4. feature sets ----------
social_num = ["social_n_profiles","social_n_platforms","engagement_mean","engagement_max","interest_mean",
              "interest_max","sentiment_score"] + list(sent.columns) + list(plat.columns) + ["has_social_profile"]
social_cat = ["top_platform"]
hist_num = ["prior_txn_count","prior_avg_amount","prior_avg_rating","days_since_last","dow","month"] + [f"prior_share_{c}" for c in cats]
hist_cat = ["last_category"]
sets = {"Social only": (social_num, social_cat),
        "History only": (hist_num, hist_cat),
        "Merged (Social + History)": (social_num + hist_num, social_cat + hist_cat)}

def pre(num, cat):
    return ColumnTransformer([
        ("n", Pipeline([("i", SimpleImputer(strategy="median", keep_empty_features=True)), ("s", StandardScaler())]), num),
        ("c", OneHotEncoder(handle_unknown="ignore"), cat)])

models = {
    "Majority baseline": lambda: DummyClassifier(strategy="prior"),
    "Logistic Regression": lambda: LogisticRegression(C=0.3, max_iter=2000),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=400, min_samples_leaf=3, random_state=42),
    "Gradient Boosting": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=42),
}
y = df.product_category
tscv = TimeSeriesSplit(n_splits=5)
test_idx = np.concatenate([te for _, te in tscv.split(df)])   # rows that get a time-ordered out-of-fold prediction
rows, preds = [], {}
for sname, (num, cat) in sets.items():
    X = df[num + cat]
    for mname, mk in models.items():
        if mname == "Majority baseline" and sname != "Merged (Social + History)": continue
        pipe = Pipeline([("p", pre(num, cat)), ("m", mk())])
        p = np.empty(len(df), dtype=object)
        for tr, te in tscv.split(df):
            p[te] = pipe.fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
        preds[(sname, mname)] = p
        rows.append(dict(feature_set=sname, model=mname,
                         accuracy=accuracy_score(y.iloc[test_idx], p[test_idx]),
                         macro_f1=f1_score(y.iloc[test_idx], p[test_idx], average="macro")))
res = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
res.to_csv(DATA / "model_comparison.csv", index=False)
print("Evaluated on", len(test_idx), "time-ordered out-of-fold rows; chance ~ 0.20, majority ~", round((y.iloc[test_idx]==y.iloc[test_idx].mode()[0]).mean(),3))
print(res.round(3).to_string(index=False))

# ---------- 5. final model = best merged-feature model, fit on all data ----------
best = res[res.feature_set == "Merged (Social + History)"].query("model != 'Majority baseline'").iloc[0]["model"]
num, cat = sets["Merged (Social + History)"]
final = Pipeline([("p", pre(num, cat)), ("m", models[best]())]).fit(df[num + cat], y)
joblib.dump({"pipeline": final, "numeric": num, "categorical": cat, "classes": list(final.classes_)}, MODELS / "product_model.joblib")
print("\nFinal model:", best)
bp = preds[("Merged (Social + History)", best)]
print(classification_report(y.iloc[test_idx], bp[test_idx], zero_division=0))
print(pd.DataFrame(confusion_matrix(y.iloc[test_idx], bp[test_idx], labels=cats), index=cats, columns=cats))

m = final.named_steps["m"]
if hasattr(m, "feature_importances_"):
    names = final.named_steps["p"].get_feature_names_out()
    imp = pd.Series(m.feature_importances_, index=names).sort_values(ascending=False).head(10)
    print("\nTop features:\n", imp.round(3).to_string())
print("\nMerge coverage: %d/%d transactions (%.0f%%) matched to a social profile" % (df.has_social_profile.sum(), len(df), 100*df.has_social_profile.mean()))
