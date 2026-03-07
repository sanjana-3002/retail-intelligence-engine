import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/online_retail_II.xlsx")
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
    df = df[df["is_cancelled"] == False]
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
