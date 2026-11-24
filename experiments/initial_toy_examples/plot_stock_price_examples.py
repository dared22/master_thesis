"""Plot indexed adjusted share prices for four Oslo Stock Exchange stocks."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

repo_root = Path(__file__).resolve().parents[2]
data_path = repo_root / "Data/eligible/price_histories.csv"
output = repo_root / "thesis/figures/theory/stock-price-examples.pdf"
preview = Path("/tmp/stock-price-examples-preview.png")

STOCKS = [
    ("NO0010161896", "DNB Bank", "#0072B2"),
    ("NO0010096985", "Equinor", "#D55E00"),
    ("NO0005052605", "Norsk Hydro", "#009E73"),
    ("NO0010063308", "Telenor", "#CC79A7"),
]

df = pd.read_csv(data_path, usecols=["stock_id", "Date", "AdjustedPrice"], parse_dates=["Date"])

fig, ax = plt.subplots(figsize=(6.2, 3.0), layout="constrained")
ax.axhline(100, color="0.6", linewidth=0.9, linestyle=":")

for stock_id, label, color in STOCKS:
    sub = df[df["stock_id"] == stock_id].sort_values("Date").copy()
    assert not sub.empty, f"Missing data for {stock_id}"
    assert sub["AdjustedPrice"].notna().all(), f"Missing prices for {stock_id}"
    indexed = 100.0 * sub["AdjustedPrice"] / sub["AdjustedPrice"].iloc[0]
    ax.plot(sub["Date"], indexed, color=color, linewidth=1.1, label=label)

ax.set(
    xlabel="Date",
    ylabel="Scaled price (Sep 2016 = 100)",
    xlim=(pd.Timestamp("2016-09-08"), pd.Timestamp("2026-09-08")),
)
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.2)
ax.legend(loc="upper left", frameon=False, fontsize=9)

fig.savefig(output)
fig.savefig(preview, dpi=180)
plt.close(fig)
print(f"Saved {output}")
