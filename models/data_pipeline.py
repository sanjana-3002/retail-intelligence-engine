import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/online_retail_II.xlsx")  # download from archive.ics.uci.edu/dataset/502/online+retail+ii
PROCESSED_DIR = Path("data/processed")


def load_raw_data(filepath=RAW_DATA_PATH):
    print("loading year 2009-2010...")
    df1 = pd.read_excel(filepath, sheet_name="Year 2009-2010", dtype={"Customer ID": str})
    print("loading year 2010-2011...")
    df2 = pd.read_excel(filepath, sheet_name="Year 2010-2011", dtype={"Customer ID": str})

    df = pd.concat([df1, df2], ignore_index=True)

    df.rename(columns={
        "Invoice":     "invoice_no",
        "StockCode":   "stock_code",
        "Description": "description",
        "Quantity":    "quantity",
        "InvoiceDate": "invoice_date",
        "Price":       "unit_price",
        "Customer ID": "customer_id",
        "Country":     "country",
    }, inplace=True)

    df["invoice_date"] = pd.to_datetime(df["invoice_date"])

    missing = df["customer_id"].isna().sum()
    pct = missing / len(df) * 100

    print(f"\ntotal rows        : {len(df):,}")
    print(f"date range        : {df['invoice_date'].min().date()} -> {df['invoice_date'].max().date()}")
    print(f"missing customer_id : {missing:,}  ({pct:.1f}%)")

    return df


def extract_signal_before_cleaning(df):
    # total positive order lines across the whole dataset
    total_positive = df[df["quantity"] > 0].shape[0]

    # return rate — negative quantity rows count as a return event
    returns = (
        df[(df["quantity"] < 0) & df["customer_id"].notna()]
        .groupby("customer_id")
        .size()
        .reset_index(name="return_count")
    )
    returns["return_rate"] = returns["return_count"] / total_positive
    return_features = returns[["customer_id", "return_rate"]]

    # cancellation rate — invoices starting with 'C'
    total_invoices = df["invoice_no"].nunique()
    cancelled = (
        df[df["invoice_no"].astype(str).str.startswith("C") & df["customer_id"].notna()]
        .groupby("customer_id")["invoice_no"]
        .nunique()
        .reset_index(name="cancelled_invoices")
    )
    cancelled["cancellation_rate"] = cancelled["cancelled_invoices"] / total_invoices
    cancel_features = cancelled[["customer_id", "cancellation_rate"]]

    print(f"return_features shape  : {return_features.shape}")
    print(f"cancel_features shape  : {cancel_features.shape}")

    return return_features, cancel_features


def clean_data(df):
    df = df.copy()
    print(f"starting rows : {len(df):,}")

    # 1. add revenue column
    df["revenue"] = df["quantity"] * df["unit_price"]
    print(f"after adding revenue        : {len(df):,}")

    # 2. flag cancellation invoices
    df["is_cancelled"] = df["invoice_no"].astype(str).str.startswith("C")
    print(f"after adding is_cancelled   : {len(df):,}")

    # 3. drop negative quantities
    df = df[df["quantity"] >= 0]
    print(f"after removing quantity < 0 : {len(df):,}")

    # 4. drop cancellation rows
    df = df[~df["is_cancelled"]]
    print(f"after removing cancellations: {len(df):,}")

    # 5. drop zero or negative price
    df = df[df["unit_price"] > 0]
    print(f"after removing price <= 0   : {len(df):,}")

    # 6. drop non-product stock codes
    non_product = {
        "POST", "D", "M", "DOT", "CRUK", "C2",
        "BANK CHARGES", "PADS", "AMAZONFEE", "S", "ADJUST", "ADJUST2"
    }
    df = df[~df["stock_code"].astype(str).str.upper().isin(non_product)]
    print(f"after removing non-product  : {len(df):,}")

    # 7. split into customers-only and full sets
    df_customers = df[df["customer_id"].notna()].copy()
    df_all = df.copy()
    print(f"df_customers : {len(df_customers):,}  |  df_all : {len(df_all):,}")

    return df_customers, df_all


def calculate_rfm(df, snapshot_date=None):
    if snapshot_date is None:
        snapshot_date = df["invoice_date"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("customer_id")
        .agg(
            last_purchase=("invoice_date", "max"),
            first_purchase=("invoice_date", "min"),
            frequency=("invoice_no", "nunique"),
            monetary=("revenue", "mean"),       # mean, not sum — required for Gamma-Gamma
            total_revenue=("revenue", "sum"),
            total_items=("quantity", "sum"),
        )
        .reset_index()
    )

    rfm["recency"] = (snapshot_date - rfm["last_purchase"]).dt.days
    rfm["customer_tenure_days"] = (rfm["last_purchase"] - rfm["first_purchase"]).dt.days

    print("\nRFM summary:")
    print(rfm[["recency", "frequency", "monetary", "total_revenue"]].describe().round(2))

    return rfm


def engineer_features(df, rfm):
    df = df.copy()

    # 1. velocity_decay_ratio — are purchase gaps getting longer over time?
    def _velocity_decay(dates):
        dates = sorted(dates)
        if len(dates) < 4:
            return 1.0
        gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        mid = len(gaps) // 2
        first_avg = np.mean(gaps[:mid]) if mid > 0 else 1.0
        second_avg = np.mean(gaps[mid:]) if len(gaps[mid:]) > 0 else 1.0
        return second_avg / first_avg if first_avg > 0 else 1.0

    vdr = (
        df.groupby("customer_id")["invoice_date"]
        .apply(lambda x: _velocity_decay(list(x)))
        .reset_index(name="velocity_decay_ratio")
    )

    # 2. category_hhi — how concentrated is each customer's buying across categories?
    df["category"] = df["stock_code"].astype(str).str[:2]

    def _hhi(series):
        shares = series.value_counts(normalize=True)
        return (shares ** 2).sum()

    cat_hhi = (
        df.groupby("customer_id")["category"]
        .apply(_hhi)
        .reset_index(name="category_hhi")
    )

    # 3. spend_cv — coefficient of variation of spend per transaction
    spend_cv = (
        df.groupby("customer_id")["revenue"]
        .agg(lambda x: x.std() / x.mean() if x.mean() != 0 else 0.0)
        .reset_index(name="spend_cv")
    )
    spend_cv["spend_cv"] = spend_cv["spend_cv"].fillna(0)

    # 4. country_count — how many different countries has this customer shopped from?
    country_count = (
        df.groupby("customer_id")["country"]
        .nunique()
        .reset_index(name="country_count")
    )

    # 5. primary_country — the country this customer shops from most
    primary_country = (
        df.groupby("customer_id")["country"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index(name="primary_country")
    )

    # 6. avg_items_per_order — average basket size across all orders
    avg_items = (
        df.groupby(["customer_id", "invoice_no"])["quantity"]
        .sum()
        .reset_index()
        .groupby("customer_id")["quantity"]
        .mean()
        .reset_index(name="avg_items_per_order")
    )

    # merge everything onto the rfm table
    enriched = rfm.copy()
    for feat_df in [vdr, cat_hhi, spend_cv, country_count, primary_country, avg_items]:
        enriched = enriched.merge(feat_df, on="customer_id", how="left")

    print(f"enriched feature table: {enriched.shape[0]} customers, {enriched.shape[1]} columns")
    return enriched


def build_cohort_matrix(df):
    df = df.copy()

    # assign each transaction to a calendar month
    df["invoice_month"] = df["invoice_date"].dt.to_period("M")

    # find the first month each customer ever bought — that's their cohort
    cohort_map = (
        df.groupby("customer_id")["invoice_month"]
        .min()
        .reset_index()
        .rename(columns={"invoice_month": "cohort_month"})
    )
    df = df.merge(cohort_map, on="customer_id", how="left")

    # months since first purchase
    df["cohort_index"] = (df["invoice_month"] - df["cohort_month"]).apply(lambda x: x.n)

    pivot = (
        df.groupby(["cohort_month", "cohort_index"])["customer_id"]
        .nunique()
        .unstack(fill_value=0)
    )

    cohort_size = pivot[0]
    retention_matrix = pivot.divide(cohort_size, axis=0)

    print(f"cohort matrix: {retention_matrix.shape[0]} cohorts x {retention_matrix.shape[1]} months")
    return retention_matrix, cohort_size


def build_master_customer_table(df_customers, return_features, cancel_features):
    rfm = calculate_rfm(df_customers)
    master = engineer_features(df_customers, rfm)

    # bring in the pre-cleaning behavioural signals
    master = master.merge(return_features, on="customer_id", how="left")
    master = master.merge(cancel_features, on="customer_id", how="left")

    master = master.fillna(0)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master.to_csv(PROCESSED_DIR / "customer_features.csv", index=False)

    print(f"\nmaster table : {master.shape[0]} customers, {master.shape[1]} features")
    print(f"columns      : {list(master.columns)}")

    return master


if __name__ == "__main__":
    # 1. load both sheets
    df_raw = load_raw_data()

    # 2. extract return/cancel signals BEFORE we drop any rows
    return_features, cancel_features = extract_signal_before_cleaning(df_raw)

    # 3. clean
    df_customers, df_all = clean_data(df_raw)

    # 4. build the master customer feature table
    master = build_master_customer_table(df_customers, return_features, cancel_features)

    # 5. build and save cohort retention matrix
    retention_matrix, cohort_size = build_cohort_matrix(df_customers)
    retention_matrix.to_csv(PROCESSED_DIR / "cohort_retention.csv")
    print("saved cohort_retention.csv")

    # 6. save all cleaned transactions
    df_all.to_csv(PROCESSED_DIR / "all_transactions.csv", index=False)
    print("saved all_transactions.csv")

    # 7. final summary
    print(f"\nunique customers : {master['customer_id'].nunique():,}")
    print(f"feature count    : {master.shape[1]}")
    print("\nmedian RFM values:")
    print(master[["recency", "frequency", "monetary"]].median().round(2))
