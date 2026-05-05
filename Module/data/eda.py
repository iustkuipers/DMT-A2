"""
EDA module voor Expedia hotel ranking (Assignment 2)
Modulaire EDA gefocust op LambdaMART feature motivatie
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11

PLOT_DIR = Path("plots/eda")


def _save(name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_DIR / name, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {name}")


# ─── PHASE 1: DATA QUALITY & LOGICAL STRUCTURE ───────────────────────────────

def basic_counts(train, test=None):
    """Basisstatistieken: unieke IDs, hotels per zoekopdracht, train/test overlap."""
    print("\n" + "="*60)
    print("PHASE 1: DATA QUALITY & LOGICAL STRUCTURE")
    print("="*60)

    n_srch = train['srch_id'].nunique()
    n_prop = train['prop_id'].nunique()
    hotels_per_srch = train.groupby('srch_id').size()
    print(f"\nBASIC COUNTS:")
    print(f"  Trainset rijen:            {len(train):,}")
    print(f"  Unieke srch_ids:           {n_srch:,}")
    print(f"  Unieke prop_ids:           {n_prop:,}")
    print(f"  Hotels per zoekopdracht:   mean={hotels_per_srch.mean():.1f}, "
          f"median={hotels_per_srch.median():.0f}, max={hotels_per_srch.max()}")

    if test is not None:
        print(f"\n  Testset rijen:             {len(test):,}")
        print(f"  Testset unieke srch_ids:   {test['srch_id'].nunique():,}")
        overlap = len(set(train['srch_id']) & set(test['srch_id']))
        print(f"  srch_id overlap train/test: {overlap} "
              f"({100*overlap/test['srch_id'].nunique():.1f}% van test)")


def missingness_overview(train):
    """Barplot van % missende waarden per kolom; prints top-15."""
    print("\nMISSINGNESS OVERVIEW:")
    miss = (100 * train.isna().sum() / len(train)).sort_values(ascending=False)
    miss = miss[miss > 0]
    print(f"  Kolommen met missing data: {len(miss)}")
    print(f"  Top 10:")
    for col, pct in miss.head(10).items():
        print(f"    {col}: {pct:.1f}%")

    fig, ax = plt.subplots(figsize=(12, 6))
    miss.head(20).sort_values().plot(kind='barh', ax=ax, color='salmon', edgecolor='black')
    ax.set_xlabel('% Missing')
    ax.set_title('Missende waarden per kolom (top 20)')
    plt.tight_layout()
    _save("01_missingness.png")


def competitor_missingness_heatmap(train):
    """
    Heatmap van de 8×3 competitor-kolom matrix (rate / inv / percent_diff).
    Toont % missing per competitor-kolom — motiveert geaggregeerde comp-feature.
    """
    print("\nCOMPETITOR MISSINGNESS HEATMAP:")
    comp_rate   = [f'comp{i}_rate'              for i in range(1, 9)]
    comp_inv    = [f'comp{i}_inv'               for i in range(1, 9)]
    comp_pct    = [f'comp{i}_rate_percent_diff' for i in range(1, 9)]

    # Bouw matrix: rijen = comp1..8, kolommen = rate/inv/pct_diff
    matrix = pd.DataFrame({
        'rate':         [100 * train[c].isna().mean() if c in train.columns else np.nan for c in comp_rate],
        'inv':          [100 * train[c].isna().mean() if c in train.columns else np.nan for c in comp_inv],
        'rate_pct_diff':[100 * train[c].isna().mean() if c in train.columns else np.nan for c in comp_pct],
    }, index=[f'comp{i}' for i in range(1, 9)])

    print(f"  Gemiddeld % missing over alle comp-kolommen: {matrix.values.mean():.1f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(matrix, annot=True, fmt='.0f', cmap='Reds', ax=ax,
                linewidths=0.5, cbar_kws={'label': '% Missing'})
    ax.set_title('Competitor velden: % missing per kolom')
    ax.set_xlabel('Veldtype')
    ax.set_ylabel('Competitor')
    plt.tight_layout()
    _save("02_competitor_missingness_heatmap.png")


def range_and_type_checks(train):
    """Hard-bounds checks op bekende domeinwaarden."""
    print("\nRANGE & TYPE CHECKS:")

    checks = {
        'prop_starrating == 0 (onbekend)': (train['prop_starrating'] == 0).sum() if 'prop_starrating' in train.columns else None,
        'prop_review_score == 0 (geen reviews)': (train['prop_review_score'] == 0).sum() if 'prop_review_score' in train.columns else None,
        'prop_review_score null': train['prop_review_score'].isna().sum() if 'prop_review_score' in train.columns else None,
        'price_usd < 0': (train['price_usd'] < 0).sum() if 'price_usd' in train.columns else None,
        'price_usd > 10000': (train['price_usd'] > 10000).sum() if 'price_usd' in train.columns else None,
        'prop_log_historical_price == 0 (niet verkocht)': (train['prop_log_historical_price'] == 0).sum() if 'prop_log_historical_price' in train.columns else None,
    }
    for label, val in checks.items():
        if val is not None:
            print(f"  {label}: {val:,}")

    if 'price_usd' in train.columns:
        p = train['price_usd']
        print(f"  price_usd bereik: ${p.min():.2f} – ${p.max():.2f} "
              f"(mean=${p.mean():.2f}, median=${p.median():.2f})")


# ─── PHASE 2: DATA EXPLORATION ───────────────────────────────────────────────

def target_distribution(train):
    """Click- en bookingrate, extreme class-imbalance zichtbaar maken."""
    print("\n" + "="*60)
    print("PHASE 2: DATA EXPLORATION")
    print("="*60)
    print("\nTARGET VARIABELEN:")

    click_rate   = train['click_bool'].mean()   if 'click_bool'   in train.columns else None
    booking_rate = train['booking_bool'].mean() if 'booking_bool' in train.columns else None

    if click_rate is not None:
        print(f"  Click rate:   {100*click_rate:.2f}%")
    if booking_rate is not None:
        print(f"  Booking rate: {100*booking_rate:.2f}% (extreme imbalance)")
        bps = train.groupby('srch_id')['booking_bool'].sum()
        print(f"  Boekingen per zoekopdracht: mean={bps.mean():.3f}, max={bps.max()}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (col, label) in zip(axes, [('click_bool', 'Click'), ('booking_bool', 'Boeking')]):
        if col not in train.columns:
            continue
        counts = train[col].value_counts().sort_index()
        bars = ax.bar(['Nee', 'Ja'], counts.values, color=['#aec6cf', '#2a9d8f'], edgecolor='black')
        ax.set_title(f'{label}s distributie')
        ax.set_ylabel('Aantal')
        rates = 100 * train[col].value_counts(normalize=True).sort_index()
        for bar, pct in zip(bars, rates.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    _save("03_target_distribution.png")


def position_bias(train):
    """
    Click/boekingrate per positie, gesplitst op random_bool.
    Kernbevinding: positiebias is intrinsiek gebruikersgedrag → motiveert NDCG + LambdaMART.
    """
    print("\nPOSITIEBIAS ANALYSE:")
    if not all(c in train.columns for c in ['position', 'random_bool', 'click_bool', 'booking_bool']):
        print("  Overgeslagen: benodigde kolommen ontbreken")
        return

    rand = train[train['random_bool'] == 1]
    nonrand = train[train['random_bool'] == 0]
    print(f"  Random sort: {len(rand):,} rijen ({100*len(rand)/len(train):.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (data, title) in zip(axes, [
        (nonrand, 'Expedia-algoritme (random_bool=0)'),
        (rand,    'Willekeurige volgorde (random_bool=1)')
    ]):
        stats = data[data['position'] <= 20].groupby('position')[['click_bool', 'booking_bool']].mean() * 100
        ax.plot(stats.index, stats['click_bool'],   marker='o', label='Click rate', linewidth=2)
        ax.plot(stats.index, stats['booking_bool'], marker='s', label='Booking rate', linewidth=2)
        ax.set_xlabel('Positie')
        ax.set_ylabel('Rate (%)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Kwantificeer decay in random conditie
    if len(rand) > 0:
        rs = rand[rand['position'].between(1, 10)].groupby('position')['click_bool'].mean()
        if 1 in rs.index and 10 in rs.index:
            print(f"  Random conditie — click rate positie 1: {rs[1]*100:.1f}%, "
                  f"positie 10: {rs[10]*100:.1f}% "
                  f"(decay {(rs[1]-rs[10])*100:.1f}pp — intrinsiek gedrag)")

    plt.tight_layout()
    _save("04_position_bias.png")


def temporal_analysis(train):
    """
    Boekings- en clickvolume per maand.
    Motiveert temporele train/test split strategie.
    """
    print("\nTEMPORELE ANALYSE:")
    dt_col = 'date_time' if 'date_time' in train.columns else ('datetime' if 'datetime' in train.columns else None)
    if dt_col is None:
        print("  Overgeslagen: geen datetime-kolom gevonden")
        return

    df = train.copy()
    df['month'] = pd.to_datetime(df[dt_col]).dt.to_period('M')
    monthly = df.groupby('month').agg(
        n_searches=('srch_id', 'nunique'),
        booking_rate=('booking_bool', 'mean') if 'booking_bool' in df.columns else ('srch_id', lambda x: np.nan)
    ).reset_index()
    monthly['month_str'] = monthly['month'].astype(str)

    print(f"  Datumrange: {monthly['month_str'].min()} – {monthly['month_str'].max()}")
    print(f"  Unieke maanden: {len(monthly)}")

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()
    ax1.bar(range(len(monthly)), monthly['n_searches'], color='steelblue', alpha=0.6, label='Zoekopdrachten')
    if 'booking_bool' in df.columns:
        ax2.plot(range(len(monthly)), monthly['booking_rate']*100, color='red',
                 marker='o', linewidth=2, label='Booking rate (%)')
        ax2.set_ylabel('Booking rate (%)', color='red')

    ax1.set_xticks(range(len(monthly)))
    ax1.set_xticklabels(monthly['month_str'], rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Aantal unieke zoekopdrachten')
    ax1.set_title('Zoekvolume en booking rate per maand')
    ax1.legend(loc='upper left')
    if 'booking_bool' in df.columns:
        ax2.legend(loc='upper right')
    plt.tight_layout()
    _save("05_temporal.png")


def price_distribution(train):
    """Prijsdistributie normaal + log-schaal; identificeert outliers."""
    print("\nPRIJSANALYSE:")
    if 'price_usd' not in train.columns:
        print("  Overgeslagen")
        return

    p = train['price_usd']
    print(f"  Mean: ${p.mean():.2f}, Median: ${p.median():.2f}, "
          f"Std: ${p.std():.2f}, Max: ${p.max():.2f}")
    print(f"  Outliers >$10k: {(p > 10000).sum():,}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(p.clip(upper=p.quantile(0.99)), bins=60, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Prijs (USD)')
    axes[0].set_ylabel('Frequentie')
    axes[0].set_title('Prijsdistributie (geknipte staart, 99e percentiel)')

    axes[1].hist(np.log1p(p[p > 0]), bins=60, color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('log(prijs + 1)')
    axes[1].set_ylabel('Frequentie')
    axes[1].set_title('Prijsdistributie (log-schaal)')
    plt.tight_layout()
    _save("06_price_distribution.png")


def destination_analysis(train):
    """
    Top destinations op boekingsvolume en booking rate.
    Motiveert destination-level aggregate features naast prop_id.
    """
    print("\nDESTINATIE-ANALYSE:")
    if 'srch_destination_id' not in train.columns or 'booking_bool' not in train.columns:
        print("  Overgeslagen")
        return

    dest = train.groupby('srch_destination_id').agg(
        n_searches=('srch_id', 'nunique'),
        booking_rate=('booking_bool', 'mean')
    ).reset_index()

    print(f"  Unieke destinations: {len(dest):,}")
    print(f"  Booking rate variatie: min={dest['booking_rate'].min():.4f}, "
          f"max={dest['booking_rate'].max():.4f}, std={dest['booking_rate'].std():.4f}")
    print(f"  → Hoge variatie motiveert destination-level booking rate als feature")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Top 20 destinations op volume
    top20 = dest.nlargest(20, 'n_searches')
    axes[0].barh(range(20), top20['n_searches'].values, color='steelblue', edgecolor='black')
    axes[0].set_yticks(range(20))
    axes[0].set_yticklabels(top20['srch_destination_id'].astype(str), fontsize=8)
    axes[0].set_xlabel('Aantal unieke zoekopdrachten')
    axes[0].set_title('Top 20 destinations op zoekvolume')

    # Booking rate distributie over alle destinations
    axes[1].hist(dest['booking_rate'], bins=50, color='coral', edgecolor='black')
    axes[1].set_xlabel('Booking rate per destination')
    axes[1].set_ylabel('Aantal destinations')
    axes[1].set_title('Verdeling booking rate per destination')
    plt.tight_layout()
    _save("07_destination_analysis.png")


def site_id_analysis(train):
    """Booking rate per site_id (Expedia.com / .co.uk / .co.jp etc.)."""
    print("\nSITE_ID ANALYSE:")
    if 'site_id' not in train.columns or 'booking_bool' not in train.columns:
        print("  Overgeslagen")
        return

    site = train.groupby('site_id').agg(
        n_rows=('srch_id', 'count'),
        booking_rate=('booking_bool', 'mean')
    ).reset_index().sort_values('n_rows', ascending=False)

    print(f"  Unieke site_ids: {len(site)}")
    print(f"  Booking rate variatie tussen sites: "
          f"min={site['booking_rate'].min():.4f}, max={site['booking_rate'].max():.4f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(site['site_id'].astype(str), site['booking_rate']*100,
                  color='mediumpurple', edgecolor='black')
    ax.set_xlabel('Site ID')
    ax.set_ylabel('Booking rate (%)')
    ax.set_title('Booking rate per Expedia-site (site_id)')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    _save("08_site_id_booking_rate.png")


# ─── PHASE 3: IMPORTANCE & DEPENDENCIES ─────────────────────────────────────

def correlation_with_target(train):
    """
    Absolute Pearson-correlatie van alle numerieke features met booking_bool.
    Geeft feature-importantie indicatie voorafgaand aan modelleren.
    """
    print("\n" + "="*60)
    print("PHASE 3: IMPORTANCE & DEPENDENCIES")
    print("="*60)
    print("\nCORRELATIE MET BOOKING_BOOL:")
    if 'booking_bool' not in train.columns:
        print("  Overgeslagen")
        return

    # Sluit ID-kolommen en targets zelf uit
    exclude = {'srch_id', 'prop_id', 'position', 'click_bool', 'booking_bool',
               'gross_booking_usd', 'random_bool'}
    numeric = train.select_dtypes(include=[np.number]).columns
    features = [c for c in numeric if c not in exclude]

    corr = train[features].corrwith(train['booking_bool']).abs().dropna().sort_values(ascending=False)

    print("  Top 15:")
    for col, val in corr.head(15).items():
        print(f"    {col}: {val:.4f}")

    fig, ax = plt.subplots(figsize=(10, 7))
    corr.head(20).sort_values().plot(kind='barh', ax=ax, color='teal', edgecolor='black')
    ax.set_xlabel('|Pearson correlatie|')
    ax.set_title('Feature correlatie met booking_bool (top 20)')
    plt.tight_layout()
    _save("09_correlation_booking.png")


def propid_aggregate_analysis(train):
    """
    Booking rate distributie over prop_ids + appearance counts.
    Kernmotivatie voor prop_id mean/std/median features + Bayesian shrinkage.
    """
    print("\nPROP_ID AGGREGATES:")
    if 'booking_bool' not in train.columns:
        print("  Overgeslagen")
        return

    prop = train.groupby('prop_id').agg(
        n=('booking_bool', 'count'),
        booking_rate=('booking_bool', 'mean'),
        click_rate=('click_bool', 'mean') if 'click_bool' in train.columns else ('booking_bool', lambda x: np.nan)
    ).reset_index()

    print(f"  Unieke properties: {len(prop):,}")
    print(f"  Booking rate range: {prop['booking_rate'].min():.4f} – {prop['booking_rate'].max():.4f}")
    print(f"  Properties met <5 verschijningen: {(prop['n'] < 5).sum():,} "
          f"({100*(prop['n'] < 5).sum()/len(prop):.1f}%) → hoge variantie, shrinkage nodig")
    print(f"  Properties met ≥50 verschijningen: {(prop['n'] >= 50).sum():,}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(prop['booking_rate'], bins=60, color='coral', edgecolor='black')
    axes[0].set_xlabel('Booking rate per property')
    axes[0].set_ylabel('Aantal properties')
    axes[0].set_title('Verdeling booking rate per prop_id\n(kernfeature voor LambdaMART)')

    axes[1].hist(prop['n'].clip(upper=100), bins=50, color='steelblue', edgecolor='black')
    axes[1].set_xlabel('Aantal verschijningen (geknipte staart bij 100)')
    axes[1].set_ylabel('Aantal properties')
    axes[1].set_title('Hoe vaak verschijnt elke property?\n(motiveert Bayesian shrinkage)')
    plt.tight_layout()
    _save("10_propid_aggregates.png")


def within_query_ranks(train):
    """
    Booking rate als functie van prijsrank en sterrenrang binnen de zoekopdracht.
    Bewijst dat relative ranking features cruciaal zijn voor LambdaMART.
    """
    print("\nWITHIN-QUERY RANK FEATURES:")
    if not all(c in train.columns for c in ['price_usd', 'prop_starrating', 'booking_bool', 'srch_id']):
        print("  Overgeslagen")
        return

    df = train.copy()
    df['price_rank']      = df.groupby('srch_id')['price_usd'].rank(method='first')
    df['starrating_rank'] = df.groupby('srch_id')['prop_starrating'].rank(ascending=False, method='first')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prijsrank vs booking rate
    pr_stats = df[df['price_rank'] <= 20].groupby('price_rank')['booking_bool'].mean() * 100
    axes[0].plot(pr_stats.index, pr_stats.values, marker='o', linewidth=2, color='steelblue')
    axes[0].fill_between(pr_stats.index, pr_stats.values, alpha=0.2, color='steelblue')
    axes[0].set_xlabel('Prijsrank binnen zoekopdracht (1 = goedkoopst)')
    axes[0].set_ylabel('Booking rate (%)')
    axes[0].set_title('Prijsrank vs booking rate\n(within-query relatief)')
    axes[0].grid(True, alpha=0.3)

    decay = pr_stats.iloc[0] - pr_stats.iloc[-1]
    print(f"  Prijsrank 1 booking rate: {pr_stats.iloc[0]:.2f}%, "
          f"rank {int(pr_stats.index[-1])}: {pr_stats.iloc[-1]:.2f}% "
          f"(decay {decay:.2f}pp)")

    # Sterrenrang vs booking rate
    sr_stats = df[df['starrating_rank'] <= 10].groupby('starrating_rank')['booking_bool'].mean() * 100
    bars = axes[1].bar(sr_stats.index, sr_stats.values, color='lightcoral', edgecolor='black', alpha=0.8)
    axes[1].set_xlabel('Sterrenrang binnen zoekopdracht (1 = hoogste sterren)')
    axes[1].set_ylabel('Booking rate (%)')
    axes[1].set_title('Sterrenrang vs booking rate\n(within-query relatief)')
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save("11_within_query_ranks.png")


def visitor_history_segment(train):
    """
    Terugkerende klanten (met history) vs nieuwe klanten: verschil in booking rate.
    Motiveert het behoud van visitor_hist kolommen ondanks hoge missingness.
    """
    print("\nVISITOR HISTORY SEGMENTATIE:")
    if not all(c in train.columns for c in ['visitor_hist_starrating', 'booking_bool']):
        print("  Overgeslagen")
        return

    has_hist = train['visitor_hist_starrating'].notna()
    pct_repeat = 100 * has_hist.mean()
    rate_repeat = train[has_hist]['booking_bool'].mean()
    rate_new    = train[~has_hist]['booking_bool'].mean()
    uplift = 100 * (rate_repeat - rate_new) / rate_new if rate_new > 0 else np.nan

    print(f"  Terugkerende klanten: {pct_repeat:.1f}%")
    print(f"  Booking rate terugkerend: {100*rate_repeat:.2f}%, nieuw: {100*rate_new:.2f}%")
    print(f"  Uplift: {uplift:.1f}% → {'Behoud visitor_hist features' if abs(uplift) > 5 else 'Marginaal signaal'}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].pie([pct_repeat, 100-pct_repeat], labels=['Terugkerend', 'Nieuw'],
                autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
    axes[0].set_title('Klantsegmentatie op history')

    axes[1].bar(['Terugkerend', 'Nieuw'], [100*rate_repeat, 100*rate_new],
                color=['skyblue', 'lightcoral'], edgecolor='black')
    axes[1].set_ylabel('Booking rate (%)')
    axes[1].set_title('Booking rate per klantsegment')
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save("12_visitor_history_segment.png")


def competitor_signal(train):
    """
    Booking rate wanneer Expedia goedkoper / gelijk / duurder dan comp1.
    Kwantificeert waarde van competitor-features ondanks hoge missingness.
    """
    print("\nCOMPETITOR SIGNAAL (comp1_rate):")
    if 'comp1_rate' not in train.columns or 'booking_bool' not in train.columns:
        print("  Overgeslagen")
        return

    with_comp = train[train['comp1_rate'].notna()]
    print(f"  Rijen met comp-data: {len(with_comp):,} ({100*len(with_comp)/len(train):.1f}%)")

    stats = with_comp.groupby('comp1_rate')['booking_bool'].agg(['mean', 'count'])
    labels = {-1: 'Expedia goedkoper', 0: 'Zelfde prijs', 1: 'Expedia duurder'}
    for val, row in stats.iterrows():
        print(f"  {labels.get(val, val)}: {100*row['mean']:.2f}% booking (n={int(row['count']):,})")

    fig, ax = plt.subplots(figsize=(8, 4))
    x = [labels.get(v, str(v)) for v in stats.index]
    ax.bar(x, stats['mean']*100, color=['#2a9d8f', '#aec6cf', '#e76f51'], edgecolor='black')
    ax.set_ylabel('Booking rate (%)')
    ax.set_title('Booking rate per competitor prijssignaal (comp1_rate)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    _save("13_competitor_signal.png")


def affinity_score_analysis(train):
    """
    Distributie van srch_query_affinity_score (log-probabiliteit).
    Correlatie met booking toont predictieve waarde ondanks 93% missingness.
    """
    print("\nAFFINITY SCORE ANALYSE:")
    if 'srch_query_affinity_score' not in train.columns:
        print("  Overgeslagen")
        return

    present = train['srch_query_affinity_score'].notna()
    missing_pct = 100 * (~present).mean()
    print(f"  Missing: {missing_pct:.1f}%")

    vals = train.loc[present, 'srch_query_affinity_score']
    print(f"  Bereik: [{vals.min():.2f}, {vals.max():.2f}], mean={vals.mean():.2f}")

    if 'booking_bool' in train.columns:
        corr = vals.corr(train.loc[present, 'booking_bool'])
        print(f"  Correlatie met booking_bool (non-null): {corr:.4f}")
        if abs(corr) > 0.03:
            print(f"  → Behoud ondanks missingness; voeg missing-indicator toe")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(vals, bins=40, color='mediumpurple', edgecolor='black')
    ax.set_xlabel('Affinity score (log-kans op klik)')
    ax.set_ylabel('Frequentie')
    ax.set_title(f'srch_query_affinity_score distributie '
                 f'(n={present.sum():,}, {100-missing_pct:.1f}% aanwezig)')
    plt.tight_layout()
    _save("14_affinity_score.png")


# ─── EXTRA ANALYSES ──────────────────────────────────────────────────────────

def location_score2_analysis(train):
    """
    prop_location_score2 missingness en booking rate aanwezig vs afwezig.
    Motiveert missing-indicator feature naast de score zelf.
    """
    print("\nPROP_LOCATION_SCORE2 ANALYSE:")
    if 'prop_location_score2' not in train.columns:
        print("  Overgeslagen: kolom niet gevonden")
        return

    present = train['prop_location_score2'].notna()
    missing_pct = 100 * (~present).mean()
    vals = train.loc[present, 'prop_location_score2']
    print(f"  Missing: {missing_pct:.1f}%")
    print(f"  Bereik: [{vals.min():.4f}, {vals.max():.4f}], mean={vals.mean():.4f}")

    if 'booking_bool' in train.columns:
        rate_present = train.loc[present,  'booking_bool'].mean()
        rate_missing = train.loc[~present, 'booking_bool'].mean()
        print(f"  Booking rate aanwezig: {100*rate_present:.2f}%")
        print(f"  Booking rate afwezig:  {100*rate_missing:.2f}%")
        diff = (rate_present - rate_missing) * 100
        print(f"  Verschil: {diff:+.2f}pp → "
              f"{'missing-indicator toevoegen' if abs(diff) > 0.1 else 'marginaal verschil'}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Distributie van de score zelf
    axes[0].hist(vals, bins=50, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('prop_location_score2')
    axes[0].set_ylabel('Frequentie')
    axes[0].set_title(f'Locatiescore 2 distributie\n({100-missing_pct:.1f}% aanwezig)')

    # Booking rate aanwezig vs afwezig
    if 'booking_bool' in train.columns:
        labels = ['Score aanwezig', 'Score afwezig (null)']
        rates  = [100*rate_present, 100*rate_missing]
        bars   = axes[1].bar(labels, rates, color=['#2a9d8f', '#e76f51'], edgecolor='black')
        axes[1].set_ylabel('Booking rate (%)')
        axes[1].set_title('Booking rate: aanwezig vs afwezig\n(motiveert missing-indicator)')
        axes[1].grid(True, alpha=0.3, axis='y')
        for bar, rate in zip(bars, rates):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{rate:.2f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    _save("15_location_score2.png")


def orig_destination_distance_analysis(train):
    """
    orig_destination_distance missingness en effect op booking rate.
    Nabijheid van zoeker tot hotel als gedragssignaal.
    """
    print("\nORIG_DESTINATION_DISTANCE ANALYSE:")
    if 'orig_destination_distance' not in train.columns:
        print("  Overgeslagen: kolom niet gevonden")
        return

    present = train['orig_destination_distance'].notna()
    missing_pct = 100 * (~present).mean()
    vals = train.loc[present, 'orig_destination_distance']
    print(f"  Missing: {missing_pct:.1f}%")
    print(f"  Bereik: [{vals.min():.1f}, {vals.max():.1f}] km, "
          f"mean={vals.mean():.1f}, median={vals.median():.1f}")

    if 'booking_bool' in train.columns:
        rate_present = train.loc[present,  'booking_bool'].mean()
        rate_missing = train.loc[~present, 'booking_bool'].mean()
        print(f"  Booking rate afstand bekend:  {100*rate_present:.2f}%")
        print(f"  Booking rate afstand onbekend: {100*rate_missing:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Log-distributie van afstand
    axes[0].hist(np.log1p(vals), bins=60, color='mediumpurple', edgecolor='black')
    axes[0].set_xlabel('log(afstand + 1) in km')
    axes[0].set_ylabel('Frequentie')
    axes[0].set_title(f'Afstand zoeker–hotel (log-schaal)\n({100-missing_pct:.1f}% aanwezig)')

    # Booking rate per afstandsquintiel
    if 'booking_bool' in train.columns:
        df_dist = train.loc[present, ['orig_destination_distance', 'booking_bool']].copy()
        df_dist['quintiel'] = pd.qcut(df_dist['orig_destination_distance'], q=5,
                                      labels=['Q1\n(dichtbij)', 'Q2', 'Q3', 'Q4', 'Q5\n(ver weg)'])
        quintiel_rates = df_dist.groupby('quintiel', observed=True)['booking_bool'].mean() * 100
        axes[1].bar(quintiel_rates.index, quintiel_rates.values,
                    color='mediumpurple', edgecolor='black', alpha=0.8)
        axes[1].set_xlabel('Afstandsquintiel')
        axes[1].set_ylabel('Booking rate (%)')
        axes[1].set_title('Booking rate per afstandsquintiel\n(dichtbij vs ver weg)')
        axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    _save("16_orig_destination_distance.png")


def search_context_analysis(train):
    """
    Booking rate per srch_length_of_stay bucket en srch_booking_window bucket.
    Last-minute vs vroegboeker gedrag als feature motivatie.
    """
    print("\nZOEKCONTEXT ANALYSE (verblijfsduur & boekingsvenster):")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plotted = 0

    if 'srch_length_of_stay' in train.columns and 'booking_bool' in train.columns:
        # Begrens op realistische waarden
        los = train[train['srch_length_of_stay'].between(1, 14)].copy()
        los_rates = los.groupby('srch_length_of_stay')['booking_bool'].agg(['mean', 'count'])
        print(f"  srch_length_of_stay: mean={train['srch_length_of_stay'].mean():.1f}, "
              f"median={train['srch_length_of_stay'].median():.0f}")

        axes[0].bar(los_rates.index, los_rates['mean']*100,
                    color='steelblue', edgecolor='black', alpha=0.8)
        axes[0].set_xlabel('Verblijfsduur (nachten, 1–14)')
        axes[0].set_ylabel('Booking rate (%)')
        axes[0].set_title('Booking rate per verblijfsduur\n(1 nacht = last-minute)')
        axes[0].grid(True, alpha=0.3, axis='y')
        plotted += 1

    if 'srch_booking_window' in train.columns and 'booking_bool' in train.columns:
        # Binning in buckets: last-minute / kort / middellang / lang vooruit
        df_bw = train[train['srch_booking_window'] >= 0].copy()
        bins   = [-1, 0, 7, 30, 90, df_bw['srch_booking_window'].max()]
        labels = ['Zelfde dag', '1–7 dagen', '8–30 dagen', '31–90 dagen', '>90 dagen']
        df_bw['window_bucket'] = pd.cut(df_bw['srch_booking_window'], bins=bins, labels=labels)
        bw_rates = df_bw.groupby('window_bucket', observed=True)['booking_bool'].mean() * 100
        print(f"  srch_booking_window: mean={train['srch_booking_window'].mean():.1f} dagen, "
              f"median={train['srch_booking_window'].median():.0f} dagen")

        axes[1].bar(bw_rates.index, bw_rates.values,
                    color='coral', edgecolor='black', alpha=0.8)
        axes[1].set_xlabel('Boekingsvenster')
        axes[1].set_ylabel('Booking rate (%)')
        axes[1].set_title('Booking rate per boekingsvenster\n(last-minute vs vroegboeker)')
        axes[1].tick_params(axis='x', rotation=20)
        axes[1].grid(True, alpha=0.3, axis='y')
        plotted += 1

    if plotted > 0:
        plt.tight_layout()
        _save("17_search_context.png")
    else:
        plt.close()
        print("  Overgeslagen: benodigde kolommen ontbreken")


def gross_booking_analysis(train):
    """
    Distributie van gross_booking_usd en correlatie met prop_starrating.
    Legt basis voor bias discussie (dure vs goedkope hotels).
    """
    print("\nGROSS_BOOKING_USD ANALYSE (bias-voorbereiding):")
    if 'gross_booking_usd' not in train.columns:
        print("  Overgeslagen: kolom niet gevonden (alleen in trainset)")
        return

    # Alleen rijen met een daadwerkelijke boeking
    booked = train[train['gross_booking_usd'] > 0]['gross_booking_usd']
    print(f"  Rijen met boeking: {len(booked):,}")
    print(f"  Transactiewaarde: mean=${booked.mean():.2f}, median=${booked.median():.2f}, "
          f"max=${booked.max():.2f}")
    print(f"  Totale omzet in dataset: ${booked.sum():,.0f}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Distributie op log-schaal
    axes[0].hist(np.log1p(booked), bins=60, color='gold', edgecolor='black')
    axes[0].set_xlabel('log(transactiewaarde + 1) USD')
    axes[0].set_ylabel('Frequentie')
    axes[0].set_title('Transactiewaarde geboekte hotels (log-schaal)')

    # Gemiddelde transactiewaarde per sterrenklasse
    if 'prop_starrating' in train.columns:
        booked_df = train[train['gross_booking_usd'] > 0].copy()
        star_revenue = booked_df.groupby('prop_starrating')['gross_booking_usd'].median()
        axes[1].bar(star_revenue.index.astype(str), star_revenue.values,
                    color='gold', edgecolor='black')
        axes[1].set_xlabel('Sterrenklasse (0 = onbekend)')
        axes[1].set_ylabel('Mediane transactiewaarde (USD)')
        axes[1].set_title('Transactiewaarde per sterrenklasse\n(basis voor bias analyse)')
        axes[1].grid(True, alpha=0.3, axis='y')
        print(f"  Mediane waarde per ster: " +
              ", ".join([f"{k}★=${v:.0f}" for k, v in star_revenue.items()]))
        print(f"  → 5★ hotels domineren omzet: potentieel algoritme-bias richting dure hotels")

    plt.tight_layout()
    _save("18_gross_booking.png")


# ─── FEATURE ENGINEERING CHECKLIST ──────────────────────────────────────────

def feature_engineering_checklist():
    """Print overzicht van te engineeren features op basis van EDA-bevindingen."""
    print("\n" + "="*60)
    print("FEATURE ENGINEERING CHECKLIST (volgende fase)")
    print("="*60)
    print("""
PROP_ID AGGREGATES (mean / std / median per prop_id):
  → prop_id_booking_rate      (met Bayesian shrinkage voor <5 verschijningen)
  → prop_id_click_rate
  → prop_id_price_mean/std
  → prop_id_location_score1_mean
  → [alle numerieke kolommen] × {mean, std, median}

DESTINATION AGGREGATES:
  → dest_booking_rate          (booking rate per srch_destination_id)
  → dest_price_mean            (gemiddelde prijs per destination)

WITHIN-QUERY RELATIEVE FEATURES:
  → price_rank_in_query        (rang van price_usd binnen srch_id)
  → starrating_rank_in_query   (rang van sterren binnen srch_id)
  → price_vs_prop_historical   (price_usd – prop_log_historical_price)
  → location_score2_rank_in_query

MISSING-INDICATOREN:
  → visitor_hist_missing       (bool: visitor_hist_starrating is null)
  → orig_distance_missing      (bool: orig_destination_distance is null)
  → affinity_missing           (bool: srch_query_affinity_score is null)
  → prop_review_missing        (bool: prop_review_score is null)

COMPETITOR AGGREGAAT:
  → n_comps_cheaper            (som comp_rate == -1 over comp1–8)
  → n_comps_pricier            (som comp_rate == 1 over comp1–8)
  → (null behandeld als 0 — elegante missingness-aanpak)
""")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def run(train_path, test_path=None):
    """
    Voer volledige EDA pipeline uit.
    Aanroepbaar vanuit main.py of standalone.
    """
    print("Loading data...")
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path) if test_path else None
    print(f"  Train: {train.shape}, Test: {test.shape if test is not None else 'niet geladen'}")

    # Phase 1
    basic_counts(train, test)
    missingness_overview(train)
    competitor_missingness_heatmap(train)
    range_and_type_checks(train)

    # Phase 2
    target_distribution(train)
    position_bias(train)
    temporal_analysis(train)
    price_distribution(train)
    destination_analysis(train)
    site_id_analysis(train)

    # Phase 3
    correlation_with_target(train)
    propid_aggregate_analysis(train)
    within_query_ranks(train)
    visitor_history_segment(train)
    competitor_signal(train)
    affinity_score_analysis(train)
    location_score2_analysis(train)
    orig_destination_distance_analysis(train)
    search_context_analysis(train)
    gross_booking_analysis(train)

    feature_engineering_checklist()

    print("\n" + "="*60)
    print("EDA KLAAR — plots opgeslagen in plots/eda/")
    print("="*60)


if __name__ == "__main__":
    # Pas paden aan naar jullie projectstructuur
    project_root = Path(__file__).parent.parent.parent
    run(
        train_path=str(project_root / "Data" / "Raw" / "training_set_VU_DM.csv"),
        test_path =str(project_root / "Data" / "Raw" / "test_set_VU_DM.csv"),
    )