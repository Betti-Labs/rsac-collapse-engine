#!/usr/bin/env python3
"""
Quick analysis of RSAC benchmark results.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Read the results
df = pd.read_csv("data/sat_series_checks.csv")

# Calculate speedups
df["speedup"] = df["bf_checks"] / df["rsac_checks"]

# Group by n and calculate statistics
stats = (
    df.groupby("n")
    .agg(
        {
            "bf_checks": ["mean", "std"],
            "rsac_checks": ["mean", "std"],
            "speedup": ["mean", "std", "max"],
        }
    )
    .round(2)
)

print("RSAC Performance Analysis")
print("=" * 50)
print(stats)

# Show individual cases with significant speedup
print("\nCases with speedup > 2x:")
significant = df[df["speedup"] > 2.0]
for _, row in significant.iterrows():
    print(
        f"n={row['n']}: {row['bf_checks']} → {row['rsac_checks']} ({row['speedup']:.1f}x speedup)"
    )

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Average checks comparison
grouped = df.groupby("n").mean()
ax1.plot(grouped.index, grouped["bf_checks"], "bo-", label="Brute Force")
ax1.plot(grouped.index, grouped["rsac_checks"], "ro-", label="RSAC")
ax1.set_xlabel("Number of Variables (n)")
ax1.set_ylabel("Average Checks")
ax1.set_title("Average Checks: BF vs RSAC")
ax1.legend()
ax1.grid(True)

# Speedup plot
speedup_avg = df.groupby("n")["speedup"].mean()
ax2.plot(speedup_avg.index, speedup_avg.values, "go-")
ax2.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="No improvement")
ax2.set_xlabel("Number of Variables (n)")
ax2.set_ylabel("Average Speedup (x)")
ax2.set_title("RSAC Speedup vs Problem Size")
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig("rsac_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nAnalysis plot saved as 'rsac_analysis.png'")
