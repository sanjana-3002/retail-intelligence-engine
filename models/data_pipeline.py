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
