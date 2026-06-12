# =============================================================================
# ALFIDO TECH — TASK 1: CUSTOMER BEHAVIOR ANALYSIS
# Dataset: https://www.kaggle.com/datasets/bhanupratapbiswas/customer-behavior-analysis
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Plot style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.family'] = 'DejaVu Sans'

# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================
print("=" * 60)
print("SECTION 1: LOADING DATASET")
print("=" * 60)

# ── Change this path to your downloaded CSV file ──────────────────────────────
df = pd.read_csv("ecommerce_customer_data.csv")
# ─────────────────────────────────────────────────────────────────────────────

print(f"Shape          : {df.shape}")
print(f"Columns        : {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)

# =============================================================================
# SECTION 2: DATA CLEANING & FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 2: DATA CLEANING & FEATURE ENGINEERING")
print("=" * 60)

# --- 2.1 Missing values -------------------------------------------------------
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Numerical: fill with median
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Categorical: fill with mode
cat_cols = df.select_dtypes(include=['object']).columns
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# --- 2.2 Duplicates -----------------------------------------------------------
dupes = df.duplicated().sum()
print(f"\nDuplicate rows removed : {dupes}")
df.drop_duplicates(inplace=True)

# --- 2.3 Parse dates ----------------------------------------------------------
date_col = [c for c in df.columns if 'date' in c.lower() or 'Date' in c][0]
df[date_col] = pd.to_datetime(df[date_col])
df['Year']       = df[date_col].dt.year
df['Month']      = df[date_col].dt.month
df['DayOfWeek']  = df[date_col].dt.day_name()
df['Hour']       = df[date_col].dt.hour
print(f"\nDate column '{date_col}' parsed. Range: {df[date_col].min()} → {df[date_col].max()}")

# --- 2.4 Outlier removal (purchase amount) ------------------------------------
amt_col = [c for c in df.columns if 'amount' in c.lower() or 'Amount' in c][0]
q1, q3 = df[amt_col].quantile([0.01, 0.99])
before = len(df)
df = df[(df[amt_col] >= q1) & (df[amt_col] <= q3)]
print(f"Outlier rows removed   : {before - len(df)}")
print(f"Clean dataset shape    : {df.shape}")

# --- 2.5 Identify key column names -------------------------------------------
cust_col = [c for c in df.columns if 'customer' in c.lower() and 'id' in c.lower()][0]
cat_col  = [c for c in df.columns if 'category' in c.lower() or 'Category' in c][0]
qty_col  = [c for c in df.columns if 'quantity' in c.lower() or 'Quantity' in c][0]

print(f"\nKey columns identified:")
print(f"  Customer ID : {cust_col}")
print(f"  Date        : {date_col}")
print(f"  Amount      : {amt_col}")
print(f"  Category    : {cat_col}")
print(f"  Quantity    : {qty_col}")

# =============================================================================
# SECTION 3: RFM FEATURE ENGINEERING
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 3: RFM FEATURE ENGINEERING")
print("=" * 60)

REFERENCE_DATE = df[date_col].max() + pd.Timedelta(days=1)

rfm = df.groupby(cust_col).agg(
    Recency   = (date_col,  lambda x: (REFERENCE_DATE - x.max()).days),
    Frequency = (date_col,  'count'),
    Monetary  = (amt_col,   'sum')
).reset_index()

rfm['AvgOrderValue']    = rfm['Monetary'] / rfm['Frequency']
rfm['PurchaseInterval'] = rfm['Recency']  / rfm['Frequency']

print(f"RFM table shape : {rfm.shape}")
print("\nRFM summary statistics:")
print(rfm[['Recency','Frequency','Monetary','AvgOrderValue']].describe().round(2))

# --- RFM Scoring (quartile-based 1–4) ----------------------------------------
rfm['R_Score'] = pd.qcut(rfm['Recency'],   4, labels=[4, 3, 2, 1]).astype(int)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1, 2, 3, 4]).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'],  4, labels=[1, 2, 3, 4]).astype(int)
rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

# --- Segment mapping ---------------------------------------------------------
def segment(score):
    if score >= 10:  return 'Champions'
    elif score >= 7: return 'High Value'
    elif score >= 5: return 'Mid Value'
    else:            return 'Low Value'

rfm['Segment'] = rfm['RFM_Score'].apply(segment)
print("\nSegment distribution:")
print(rfm['Segment'].value_counts())

# =============================================================================
# SECTION 4: VISUALIZATIONS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 4: GENERATING VISUALIZATIONS")
print("=" * 60)

COLORS = {
    'Champions' : '#2E7D32',
    'High Value': '#1565C0',
    'Mid Value' : '#F57C00',
    'Low Value' : '#C62828',
}
SEG_ORDER = ['Champions', 'High Value', 'Mid Value', 'Low Value']

# ── Plot 1: Segment Distribution (Pie + Bar) ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Customer Segmentation — RFM Analysis', fontsize=16, fontweight='bold')

seg_counts = rfm['Segment'].value_counts().reindex(SEG_ORDER)
seg_rev    = rfm.groupby('Segment')['Monetary'].sum().reindex(SEG_ORDER)

axes[0].pie(seg_counts, labels=SEG_ORDER, autopct='%1.1f%%',
            colors=[COLORS[s] for s in SEG_ORDER], startangle=140,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0].set_title('Customer Count by Segment')

bars = axes[1].bar(SEG_ORDER, seg_rev / seg_rev.sum() * 100,
                   color=[COLORS[s] for s in SEG_ORDER], edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, seg_rev / seg_rev.sum() * 100):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{val:.1f}%', ha='center', fontweight='bold')
axes[1].set_title('Revenue Contribution by Segment (%)')
axes[1].set_ylabel('Revenue Share (%)')
axes[1].set_ylim(0, 55)

plt.tight_layout()
plt.savefig('plot1_segmentation.png', bbox_inches='tight')
plt.show()
print("  Saved: plot1_segmentation.png")

# ── Plot 2: RFM Distributions ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('RFM Metric Distributions', fontsize=16, fontweight='bold')

metrics = [('Recency', '#1565C0', 'Days since last purchase'),
           ('Frequency', '#2E7D32', 'Number of orders'),
           ('Monetary', '#F57C00', 'Total spend (₹)')]

for ax, (col, color, xlabel) in zip(axes, metrics):
    ax.hist(rfm[col], bins=40, color=color, edgecolor='white', alpha=0.85)
    ax.axvline(rfm[col].mean(), color='red', linestyle='--', linewidth=1.5,
               label=f'Mean: {rfm[col].mean():.0f}')
    ax.set_title(col)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Customer Count')
    ax.legend()

plt.tight_layout()
plt.savefig('plot2_rfm_distributions.png', bbox_inches='tight')
plt.show()
print("  Saved: plot2_rfm_distributions.png")

# ── Plot 3: Purchase Patterns by Category ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Purchase Patterns by Product Category', fontsize=16, fontweight='bold')

cat_txn = df[cat_col].value_counts()
cat_rev = df.groupby(cat_col)[amt_col].sum().sort_values(ascending=False)

axes[0].barh(cat_txn.index, cat_txn.values, color=sns.color_palette("Blues_r", len(cat_txn)))
axes[0].set_title('Transaction Count by Category')
axes[0].set_xlabel('Number of Transactions')

axes[1].barh(cat_rev.index, cat_rev.values / 1e6, color=sns.color_palette("Greens_r", len(cat_rev)))
axes[1].set_title('Revenue by Category (₹ Millions)')
axes[1].set_xlabel('Revenue (₹ Millions)')

plt.tight_layout()
plt.savefig('plot3_category_patterns.png', bbox_inches='tight')
plt.show()
print("  Saved: plot3_category_patterns.png")

# ── Plot 4: Monthly Purchase Trend ───────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('Temporal Purchase Trends', fontsize=16, fontweight='bold')

monthly = df.groupby([df[date_col].dt.to_period('M')])[amt_col].agg(['sum','count'])
monthly.index = monthly.index.astype(str)

axes[0].plot(monthly.index, monthly['sum'] / 1e6, color='#1565C0', linewidth=2, marker='o', markersize=3)
axes[0].fill_between(monthly.index, monthly['sum'] / 1e6, alpha=0.2, color='#1565C0')
axes[0].set_title('Monthly Revenue (₹ Millions)')
axes[0].set_ylabel('Revenue (₹ M)')
axes[0].tick_params(axis='x', rotation=45)

axes[1].plot(monthly.index, monthly['count'], color='#2E7D32', linewidth=2, marker='o', markersize=3)
axes[1].fill_between(monthly.index, monthly['count'], alpha=0.2, color='#2E7D32')
axes[1].set_title('Monthly Transaction Count')
axes[1].set_ylabel('Transactions')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('plot4_monthly_trends.png', bbox_inches='tight')
plt.show()
print("  Saved: plot4_monthly_trends.png")

# ── Plot 5: Churn Risk Distribution ──────────────────────────────────────────
def churn_risk(recency):
    if recency <= 30:   return 'Active (Safe)'
    elif recency <= 90: return 'Moderate Risk'
    elif recency <= 180:return 'High Risk'
    else:               return 'Churned'

rfm['ChurnRisk'] = rfm['Recency'].apply(churn_risk)
risk_order = ['Active (Safe)', 'Moderate Risk', 'High Risk', 'Churned']
risk_colors = ['#2E7D32', '#F9A825', '#E65100', '#C62828']
risk_counts = rfm['ChurnRisk'].value_counts().reindex(risk_order)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Churn Risk Assessment', fontsize=16, fontweight='bold')

axes[0].pie(risk_counts, labels=risk_order, autopct='%1.1f%%',
            colors=risk_colors, startangle=140,
            wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[0].set_title('Churn Risk Distribution')

axes[1].bar(risk_order, risk_counts.values, color=risk_colors, edgecolor='white', linewidth=1.5)
for i, (bar, val) in enumerate(zip(axes[1].patches, risk_counts.values)):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{val:,}', ha='center', fontweight='bold')
axes[1].set_title('Churn Risk Customer Count')
axes[1].set_ylabel('Number of Customers')
axes[1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.savefig('plot5_churn_risk.png', bbox_inches='tight')
plt.show()
print("  Saved: plot5_churn_risk.png")

# ── Plot 6: RFM Heatmap (avg per segment) ────────────────────────────────────
rfm_heatmap = rfm.groupby('Segment')[['Recency','Frequency','Monetary','AvgOrderValue']].mean().reindex(SEG_ORDER)
rfm_norm = (rfm_heatmap - rfm_heatmap.min()) / (rfm_heatmap.max() - rfm_heatmap.min())

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(rfm_norm, annot=rfm_heatmap.round(1), fmt='g', cmap='Blues',
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Normalized Score'})
ax.set_title('RFM Metric Heatmap by Segment (raw values annotated)', fontsize=14, fontweight='bold')
ax.set_ylabel('')
plt.tight_layout()
plt.savefig('plot6_rfm_heatmap.png', bbox_inches='tight')
plt.show()
print("  Saved: plot6_rfm_heatmap.png")

# ── Plot 7: Day-of-Week & Hour Patterns ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Purchase Timing Patterns', fontsize=16, fontweight='bold')

dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
dow_counts = df['DayOfWeek'].value_counts().reindex(dow_order)
axes[0].bar(dow_order, dow_counts.values, color=sns.color_palette("Blues_r", 7), edgecolor='white')
axes[0].set_title('Transactions by Day of Week')
axes[0].set_ylabel('Transaction Count')
axes[0].tick_params(axis='x', rotation=30)

hour_counts = df['Hour'].value_counts().sort_index()
axes[1].plot(hour_counts.index, hour_counts.values, color='#1565C0', linewidth=2, marker='o', markersize=4)
axes[1].fill_between(hour_counts.index, hour_counts.values, alpha=0.2, color='#1565C0')
axes[1].set_title('Transactions by Hour of Day')
axes[1].set_xlabel('Hour (0–23)')
axes[1].set_ylabel('Transaction Count')
axes[1].set_xticks(range(0, 24))

plt.tight_layout()
plt.savefig('plot7_timing_patterns.png', bbox_inches='tight')
plt.show()
print("  Saved: plot7_timing_patterns.png")

# =============================================================================
# SECTION 5: KEY FINDINGS SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 5: KEY FINDINGS SUMMARY")
print("=" * 60)

total_customers = len(rfm)
total_revenue   = rfm['Monetary'].sum()

for seg in SEG_ORDER:
    seg_df = rfm[rfm['Segment'] == seg]
    rev_pct = seg_df['Monetary'].sum() / total_revenue * 100
    print(f"\n  [{seg}]")
    print(f"    Customers   : {len(seg_df):,} ({len(seg_df)/total_customers*100:.1f}%)")
    print(f"    Revenue     : ₹{seg_df['Monetary'].sum():,.0f} ({rev_pct:.1f}%)")
    print(f"    Avg Recency : {seg_df['Recency'].mean():.0f} days")
    print(f"    Avg Frequency: {seg_df['Frequency'].mean():.1f} orders")
    print(f"    Avg Monetary : ₹{seg_df['Monetary'].mean():,.0f}")

print(f"\n  Churn Risk Breakdown:")
for risk in risk_order:
    cnt = (rfm['ChurnRisk'] == risk).sum()
    print(f"    {risk:20s}: {cnt:,} ({cnt/total_customers*100:.1f}%)")

print(f"\n  Total Customers Analyzed : {total_customers:,}")
print(f"  Total Revenue (all time) : ₹{total_revenue:,.0f}")
print(f"  Avg Order Value          : ₹{rfm['AvgOrderValue'].mean():,.2f}")

# =============================================================================
# SECTION 6: SAVE RFM TABLE
# =============================================================================
rfm.to_csv("rfm_segments_output.csv", index=False)
print("\n  RFM table saved to: rfm_segments_output.csv")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE — All plots & CSV saved successfully!")
print("=" * 60)
