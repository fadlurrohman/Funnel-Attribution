import numpy as np
import pandas as pd

FUNNEL_STEPS = ["view", "search", "detail", "checkout", "purchase"]

def generate_funnel_data(n_users: int = 3000, random_state: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    channels = np.array(["Google", "Facebook", "Baidu", "Naver", "Direct", "Email"])
    devices = np.array(["mobile", "desktop"])
    countries = np.array(["TH", "TW", "KR", "JP", "SG"])

    rows = []
    base_dates = pd.date_range("2025-01-01", "2025-03-31", freq="D")

    for user_id in range(1, n_users + 1):
        uid = f"U{user_id:05d}"
        first_date = rng.choice(base_dates)
        channel = rng.choice(channels)
        device = rng.choice(devices)
        country = rng.choice(countries)

        # Base conversion probabilities by channel
        base_prob = {
            "Google": 0.25,
            "Facebook": 0.18,
            "Baidu": 0.22,
            "Naver": 0.20,
            "Direct": 0.30,
            "Email": 0.28,
        }[channel]

        reached_step = "view"
        for step_idx, step in enumerate(FUNNEL_STEPS):
            if step_idx == 0:
                prob = 1.0
            else:
                prob = base_prob * (0.85 ** step_idx)
            if rng.random() <= prob:
                reached_step = step
                timestamp = first_date + pd.to_timedelta(step_idx, "h")
                rows.append({
                    "user_id": uid,
                    "timestamp": timestamp,
                    "channel": channel,
                    "device": device,
                    "country": country,
                    "step": step,
                })
            else:
                break

        if reached_step == "purchase":
            revenue = rng.uniform(40, 400)
            rows[-1]["revenue_usd"] = float(round(revenue, 2))

    df = pd.DataFrame(rows)
    if "revenue_usd" not in df.columns:
        df["revenue_usd"] = 0.0
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.to_csv("funnel_events.csv", index=False)
    print(f"Generated funnel_events.csv with {len(df)} rows for {n_users} users")
    return df

if __name__ == "__main__":
    generate_funnel_data()
