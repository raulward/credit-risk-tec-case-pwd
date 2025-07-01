import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from typing import Optional, Union
import polars as pl
import gc
import numpy as np

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from typing import List, Tuple, Dict

default_palette = sns.color_palette("tab10", n_colors=8)


class ChartMaker:

    def __init__(self, figure_size: Tuple[int] = (6,6), font: str = "sans-serif", palette: List[str] = default_palette):
        self._figure_size = figure_size
        self.font = font
        self.palette = default_palette

        sns.set_theme(
            context="talk",        # textos maiores para apresentação
            style="whitegrid",     # grade clara
            font=self.font,     # fonte limpa
        )

        pass

    def plot_bar_chart(self, X: pd.DataFrame, y: pd.Series, color_palette: List[str] = None) -> None:
        if color_palette == None:
            color_palette = self.palette
        fig, ax = plt.subplots(figsize=(self._figure_size))
        sns.barplot(x=X, y=y, palette=color_palette, ax=ax, hue=X, legend=False)
        for bar in ax.patches:
            x = bar.get_width() / 2 +  bar.get_x()
            y = bar.get_height()
            ax.text(
                x,
                y * 0.98,
                f"{y:2f}",
                va='top', ha='center',
                color='white'
            )
        plt.tight_layout()
        plt.show()
        return None

class HandleData:

    @staticmethod
    def unzip_files(path: Path,
                    dest_dir: Path,
                    file_type: str = None,
                    prefix: str = None) -> None:

        # aborta se o destino já existir e não estiver vazio
        if dest_dir.exists() and any(dest_dir.iterdir()):
            print("Destino já existe e não está vazio; nada a fazer.")
            return

        if not path.is_file():
            raise FileNotFoundError(f"ZIP não encontrado: {path}")

        dest_dir.mkdir(parents=True, exist_ok=True)
        file_type = file_type.lower().lstrip('.') if file_type else None

        with ZipFile(path, 'r') as z:
            to_extract = []
            for member in z.namelist():
                parts = Path(member).parts
                if len(parts) >= 2 and parts[1] == "train":
                    name = parts[-1]
                    if file_type and not name.lower().endswith(f".{file_type}"):
                        continue
                    if prefix and not name.startswith(prefix):
                        continue
                    to_extract.append(member)

            for member in to_extract:
                orig = Path(member).name
                new_name = orig[len(prefix):] if prefix and orig.startswith(prefix) else orig
                target = dest_dir / new_name
                with z.open(member) as src, open(target, 'wb') as dst:
                    dst.write(src.read())

        print(f"Extraídos {len(to_extract)} arquivos de 'train/' para '{dest_dir}'")


    @staticmethod
    def manage_memory(df: pl.DataFrame) -> pl.DataFrame:
        """
        Downcast integer columns to the smallest signed type that fits their range,
        convert Float64 to Float32 and Utf8 to Categorical.
        """
        for name, dtype in zip(df.columns, df.dtypes):
            if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                         pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                stats = df.select(
                    pl.col(name).min().alias("mn"),
                    pl.col(name).max().alias("mx")
                ).to_dicts()[0]
                mn, mx = stats["mn"], stats["mx"]
                if mn is None or mx is None:
                    continue

                # signed integers
                if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
                    if mn >= -2**7     and mx <=  2**7  - 1:
                        target = pl.Int8
                    elif mn >= -2**15  and mx <=  2**15 - 1:
                        target = pl.Int16
                    elif mn >= -2**31  and mx <=  2**31 - 1:
                        target = pl.Int32
                    else:
                        target = pl.Int64

                # unsigned integers
                else:
                    if mn >= 0 and mx <= 2**8  - 1:
                        target = pl.UInt8
                    elif mn >= 0 and mx <= 2**16 - 1:
                        target = pl.UInt16
                    elif mn >= 0 and mx <= 2**32 - 1:
                        target = pl.UInt32
                    else:
                        target = pl.UInt64

                if target != dtype:
                    df = df.with_columns(pl.col(name).cast(target).alias(name))

            elif dtype == pl.Float64:
                df = df.with_columns(pl.col(name).cast(pl.Float32).alias(name))

            elif dtype == pl.Utf8:
                df = df.with_columns(pl.col(name).cast(pl.Categorical).alias(name))

        return df


    @staticmethod
    def handle_dtypes(df: pl.DataFrame) -> pl.DataFrame:
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    @staticmethod
    def delta_dates(df: pl.DataFrame) -> pl.DataFrame:
        date_cols = [c for c, dt in df.schema.items() if str(dt).startswith("Date")]
        # Adiciona 'date_decision' se existir
        if "date_decision" in df.columns:
            df = df.with_columns([
                pl.col("date_decision").cast(pl.Date).cast(pl.Int64).alias("__date_decision_days")
            ])
        for c in date_cols:
            df = df.with_columns([
                pl.col(c).cast(pl.Date).cast(pl.Int64).alias(f"__{c}_days")
            ])
        df = df.with_columns([
            (pl.col("__date_decision_days") - pl.col(f"__{c}_days")).alias(f"{c}_dt_delta")
            for c in date_cols
        ])
        to_drop = (
            [f"__{c}_days" for c in date_cols]
            + date_cols
            + ["date_decision", "MONTH", "__date_decision_days"]
        )
        to_drop = [c for c in to_drop if c in df.columns]
        return df.drop(to_drop)

    @staticmethod
    def missing_pct(df: pl.DataFrame, thresh: float = 0.7) -> pl.DataFrame:
        n_rows = df.height
        null_counts = df.null_count().to_dicts()[0]
        cols_to_drop = [
            col
            for col, cnt in null_counts.items()
            if cnt / n_rows > thresh
        ]
        if cols_to_drop:
            print(f"Descartando colunas com >{thresh*100:.0f}% de missing:", cols_to_drop)
        return df.drop(cols_to_drop)


    @staticmethod
    def read_file(path: Path):
        df = pl.read_parquet(path)
        df = df.pipe(HandleData.handle_dtypes)
        df = df.pipe(HandleData.manage_memory)
        df = df.pipe(HandleData.missing_pct)
        df = df.pipe(HandleData.check_str_cardinality)
        return df


    @staticmethod
    def merge_tables(base, tables, on: str = "case_id", how: str = "left"):
        result = base
        for name, df_agg in tables.items():
            result = result.join(df_agg, on=on, how=how, suffix = f"_{name}")
            gc.collect()
        result = result.pipe(HandleData.delta_dates)

        del tables
        gc.collect()
        return result

    def check_str_cardinality(df):
        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()
                if (freq == 1) | (freq > 200):
                    df = df.drop(col)
        return df

class Aggregator:

    @staticmethod
    def num_agg(df):
        aggs = []
        for col in df.columns:
            if col.endswith(("P", "A",)):
                aggs.extend([
                    pl.col(col).max().alias(f"{col}_max"),
                    pl.col(col).sort().last().alias(f"{col}_last"),
                    pl.col(col).mean().alias(f"{col}_mean"),
                ])
        return aggs

    @staticmethod
    def date_agg(df):
        aggs = []
        for col in df.columns:
            if col.endswith(("D",)):
                aggs.extend([
                    pl.col(col).max().alias(f"{col}_max"),
                    pl.col(col).sort().last().alias(f"{col}_last"),
                    pl.col(col).mean().alias(f"{col}_mean"),
                ])
        return aggs


    @staticmethod
    def str_agg(df):
        aggs = []
        for col in df.columns:
            if col.endswith(("M",)):
                aggs.extend([
                    pl.col(col).sort().last().alias(f"{col}_lexmax"),
                    pl.col(col).last().alias(f"{col}_last"),
                ])
        return aggs

    @staticmethod
    def other_agg(df):
        aggs = []
        for col in df.columns:
            if col.endswith(("T", "L",)):
                aggs.extend([
                    pl.col(col).sort().last().alias(f"{col}_lexmax"),
                    pl.col(col).last().alias(f"{col}_last"),
                ])
        return aggs

    @staticmethod
    def aggregate(df):
        final_aggs = Aggregator.num_agg(df) + Aggregator.date_agg(df) + \
                    Aggregator.str_agg(df) + Aggregator.other_agg(df)

        result = df.group_by("case_id").agg(final_aggs)

        del df
        gc.collect()

        return result

class ModelUtils:

    @staticmethod
    def make_model(preprocessor, model, X_train, X_test, y_train, y_test):
        scoring = ["roc_auc", "accuracy", "precision", "recall", "f1"]
        pipe = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        pipe.fit(X_train, y_train)
        y_preds = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_res = cross_validate(
            pipe,
            X_train, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )
        cv_metrics = {m: cv_res[f"test_{m}"].mean() for m in scoring}
        test_metrics = {
            "roc_auc":  roc_auc_score(y_test, y_proba),
            "accuracy": accuracy_score(y_test, y_preds),
            "precision": precision_score(y_test, y_preds),
            "recall":    recall_score(y_test, y_preds),
            "f1":        f1_score(y_test, y_preds)
        }

        return {
            "pipeline": pipe,
            "cv_metrics":  cv_metrics,
            "test_metrics": test_metrics
        }
