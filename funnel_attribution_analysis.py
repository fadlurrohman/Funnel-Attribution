import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

FUNNEL_STEPS = ["view", "search", "detail", "checkout", "purchase"]

def compute_funnel(df: pd.DataFrame) -> pd.DataFrame:
    funnel = []
    for step in FUNNEL_STEPS:
        users_at_step = df[df["step"] == step]["user_id"].nunique()
        funnel.append({"step": step, "unique_users": users_at_step})
    funnel_df = pd.DataFrame(funnel)
    funnel_df["conv_rate_vs_prev"] = funnel_df["unique_users"].pct_change().fillna(1.0) + 1e-9
    return funnel_df

def funnel_by_channel(df: pd.DataFrame) -> pd.DataFrame:
    result = []
    for ch, grp in df.groupby("channel"):
        row = {"channel": ch}
        for step in FUNNEL_STEPS:
            row[f"{step}_users"] = grp[grp["step"] == step]["user_id"].nunique()
        result.append(row)
    return pd.DataFrame(result)

def last_touch_attribution(df: pd.DataFrame) -> pd.DataFrame:
    purchases = df[df["step"] == "purchase"].copy()
    if purchases.empty:
        return pd.DataFrame()
    attribution = purchases.groupby("channel")["revenue_usd"].sum().reset_index()
    attribution = attribution.rename(columns={"revenue_usd": "attributed_revenue"})
    attribution["share"] = attribution["attributed_revenue"] / attribution["attributed_revenue"].sum()
    return attribution.sort_values("attributed_revenue", ascending=False)

def build_conversion_model(df: pd.DataFrame):
    users = df.groupby("user_id").agg(
        channel=("channel", "first"),
        device=("device", "first"),
        purchased=("step", lambda s: int("purchase" in set(s)))
    ).reset_index()

    X = pd.get_dummies(users[["channel", "device"]], drop_first=True)
    y = users["purchased"]

    if y.nunique() < 2:
        print("Not enough positive/negative examples to fit model.")
        return None, users

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    users["predicted_prob_purchase"] = model.predict_proba(X)[:, 1]
    return model, users

if __name__ == "__main__":
    df = pd.read_csv("funnel_events.csv", parse_dates=["timestamp"])
    print("===== Overall funnel =====")
    funnel = compute_funnel(df)
    print(funnel)

    print("\n===== Funnel by channel =====")
    ch_funnel = funnel_by_channel(df)
    print(ch_funnel)

    print("\n===== Last-touch revenue attribution =====")
    attr = last_touch_attribution(df)
    print(attr)

    print("\n===== Simple conversion model (logistic regression) =====")
    model, users = build_conversion_model(df)
    if model is not None:
        print("Model coefficients (first 10 features):")
        print(dict(list(zip(users.columns[1:-2], model.coef_[0]))[:10]))
        print("\nAverage predicted probability of purchase by channel:")
        print(users.groupby("channel")["predicted_prob_purchase"].mean().sort_values(ascending=False))
