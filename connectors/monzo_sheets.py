"""Monzo spending from a CSV export or the Google Sheets auto-export.

**CSV (simplest):** in the Monzo app, *Account -> Export transactions -> CSV*, then
upload it on the Connections page (or drop it into ``data/inbox/monzo/``). Several
overlapping exports are fine - transactions are de-duplicated by Transaction ID.

**Google Sheets (automatic, Monzo Plus/Premium/Perks):**

1. In the Monzo app enable *Export to Google Sheets*; note the spreadsheet id
   (the long token in the sheet URL) -> ``MONZO_SHEET_ID``.
2. Create a Google Cloud service account, enable the Google Sheets + Drive APIs,
   download its JSON key -> ``GOOGLE_SERVICE_ACCOUNT_FILE``.
3. Share the spreadsheet (Viewer) with the service account's e-mail address.

Transactions are aggregated to ``(date, category)`` daily spend in GBP. Outgoing
amounts are positive. Refunds (incoming amounts in spending categories) are netted
against spend; internal pot transfers and excluded categories (income, savings,
transfers by default) are ignored.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from connectors.base import BaseConnector, ConnectorError, Frames
from connectors.utils import find_column, slugify

EXCLUDED_TYPES = {"pot transfer", "pot_transfer", "account interest"}


class MonzoSheetsConnector(BaseConnector[pd.DataFrame]):
    """Read Monzo transactions (CSV exports or Google Sheet) and aggregate daily spend by category."""

    name = "monzo"
    description = "Monzo transactions from CSV exports or the Google Sheets export, aggregated by category."
    tables = ("monzo_spending",)
    # Both sources contain full history; a day's category rows are recomputed each run.
    filter_to_range = False
    overwrite_nulls = True

    def csv_files(self) -> list[Path]:
        """Monzo exports (CSV, or the Google Sheet downloaded/synced as .xlsx) in the import folder."""
        folder = self.settings.monzo_import_dir
        if not folder.exists():
            return []
        return sorted(p for p in folder.iterdir()
                      if p.suffix.lower() in {".csv", ".xlsx"} and not p.name.startswith("~$"))

    def _sheets_configured(self) -> bool:
        key = self.settings.google_service_account_file
        return bool(self.settings.monzo_sheet_id and key and key.expanduser().exists())

    def missing_config(self) -> list[str]:
        """Either a CSV export or a sheet id + service-account key file is required."""
        if self.csv_files() and not self.settings.monzo_sheet_id:
            return []
        if not self.settings.monzo_sheet_id and not self.settings.google_service_account_file:
            return ["a Monzo CSV export (upload on the Connections page) or Google Sheets settings"]
        missing = []
        if not self.settings.monzo_sheet_id:
            missing.append("MONZO_SHEET_ID")
        key = self.settings.google_service_account_file
        if key is None:
            missing.append("GOOGLE_SERVICE_ACCOUNT_FILE")
        elif not key.expanduser().exists():
            missing.append(f"GOOGLE_SERVICE_ACCOUNT_FILE (not found: {key})")
        return missing

    def extract(self, start: date, end: date) -> pd.DataFrame:
        """Read CSV exports, or download all rows of the Monzo worksheet."""
        if not self._sheets_configured():
            frames = [read_monzo_file(path) for path in self.csv_files()]
            if not frames:
                raise ConnectorError("No Monzo exports found")
            combined = pd.concat(frames, ignore_index=True)
            id_col = find_column(combined, ["transaction_id", "id"])
            return combined.drop_duplicates(subset=[id_col], keep="last") if id_col else combined.drop_duplicates()
        try:
            import gspread
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise ConnectorError("gspread is not installed: pip install gspread google-auth") from exc

        assert self.settings.google_service_account_file is not None
        client = gspread.service_account(filename=str(self.settings.google_service_account_file.expanduser()))
        try:
            spreadsheet = client.open_by_key(self.settings.monzo_sheet_id)
        except gspread.exceptions.SpreadsheetNotFound as exc:
            raise ConnectorError(
                "Spreadsheet not found - share it with the service account e-mail and check MONZO_SHEET_ID"
            ) from exc
        worksheet = (
            spreadsheet.worksheet(self.settings.monzo_worksheet_name)
            if self.settings.monzo_worksheet_name
            else spreadsheet.get_worksheet(0)
        )
        records: list[dict[str, Any]] = worksheet.get_all_records(numericise_ignore=["all"])
        self.log.debug("Read %d Monzo transactions", len(records))
        return pd.DataFrame(records)

    def transform(self, raw: pd.DataFrame) -> Frames:
        """Aggregate transactions into ``monzo_spending`` rows."""
        daily = aggregate_monzo_transactions(
            raw,
            category_map=self.settings.monzo_category_map,
            excluded_categories=self.settings.monzo_excluded_categories,
        )
        return {"monzo_spending": daily if not daily.empty else self.empty_frame("monzo_spending")}


def read_monzo_file(path: Path) -> pd.DataFrame:
    """Read a Monzo CSV export, or every transactions tab of the Monzo Google Sheet saved as .xlsx.

    The sheet has several tabs (Live Feed, Budget, ...); the transaction tabs are
    named "... Transactions" (e.g. "Personal Account Transactions") and carry a
    ``Transaction ID`` column.
    """
    if path.suffix.lower() != ".xlsx":
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    workbook = pd.ExcelFile(path)
    tabs = [name for name in workbook.sheet_names if "transaction" in name.lower()] or workbook.sheet_names
    frames = []
    for name in tabs:
        frame = workbook.parse(name)
        if find_column(frame, ["transaction_id"]) is not None:
            frames.append(frame)
    if not frames:
        raise ConnectorError(f"No transactions tab with a 'Transaction ID' column found in {path.name}")
    return pd.concat(frames, ignore_index=True)


def aggregate_monzo_transactions(
    transactions: pd.DataFrame,
    category_map: dict[str, str],
    excluded_categories: list[str],
    net_refunds: bool = True,
) -> pd.DataFrame:
    """Aggregate raw Monzo export rows to daily spend per normalised category.

    Args:
        transactions: Rows from the Monzo export (columns ``Date``, ``Category``,
            ``Amount``, ``Type`` ...; matched case-insensitively).
        category_map: Monzo category (lower-case) -> normalised category slug.
            Unmapped categories are slugified as-is.
        excluded_categories: Normalised categories to drop entirely.
        net_refunds: Subtract incoming amounts in spending categories from spend.

    Returns:
        DataFrame with ``date``, ``category``, ``amount_gbp``, ``transaction_count``.

    Raises:
        ConnectorError: If the date or amount column cannot be found.
    """
    if transactions.empty:
        return pd.DataFrame(columns=["date", "category", "amount_gbp", "transaction_count"])

    date_col = find_column(transactions, ["date"])
    amount_col = find_column(transactions, ["amount", "money_out"])
    category_col = find_column(transactions, ["category"])
    type_col = find_column(transactions, ["type"])
    currency_col = find_column(transactions, ["currency"])
    if date_col is None or amount_col is None:
        raise ConnectorError(f"Monzo sheet is missing Date/Amount columns: {list(transactions.columns)}")

    frame = pd.DataFrame(
        {
            "date": (transactions[date_col].dt.date if pd.api.types.is_datetime64_any_dtype(transactions[date_col])
                     else pd.to_datetime(transactions[date_col], dayfirst=True, errors="coerce").dt.date),
            "amount": pd.to_numeric(
                transactions[amount_col].astype(str).str.replace(r"[£,\s]", "", regex=True), errors="coerce"
            ),
            "raw_category": transactions[category_col].astype(str).str.strip().str.lower()
            if category_col
            else "general",
            "type": transactions[type_col].astype(str).str.strip().str.lower() if type_col else "",
        }
    )
    if currency_col:
        # "Amount" is always in the account currency (GBP); keep only GBP accounts.
        frame = frame.loc[transactions[currency_col].astype(str).str.upper().isin(["GBP", ""])]

    frame = frame.dropna(subset=["date", "amount"])
    frame = frame.loc[~frame["type"].isin(EXCLUDED_TYPES)]
    frame["category"] = frame["raw_category"].map(lambda c: category_map.get(c, slugify(c) if c else "other"))
    frame = frame.loc[~frame["category"].isin({c.lower() for c in excluded_categories})]

    if net_refunds:
        frame["spend"] = -frame["amount"]
    else:
        frame = frame.loc[frame["amount"] < 0]
        frame["spend"] = -frame["amount"]

    daily = (
        frame.groupby(["date", "category"], as_index=False)
        .agg(amount_gbp=("spend", "sum"), transaction_count=("spend", "size"))
        .sort_values(["date", "category"])
    )
    daily["amount_gbp"] = daily["amount_gbp"].round(2)
    return daily.reset_index(drop=True)
