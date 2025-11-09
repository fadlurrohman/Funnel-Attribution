# Conversion Funnel & Channel Attribution

This project simulates a typical Agoda booking funnel:

1. Users arrive from different marketing channels.
2. They go through funnel steps: view listing → search → hotel detail → checkout → purchase.
3. We compute funnel conversion rates by channel and perform simple **last-touch attribution** for revenue.
4. Optionally, a logistic regression predicts the probability of purchase based on channel and device.

## Files

- `generate_funnel_data.py` – create synthetic event-level funnel data.
- `funnel_attribution_analysis.py` – compute funnel metrics and channel revenue attribution.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python generate_funnel_data.py
python funnel_attribution_analysis.py
```
