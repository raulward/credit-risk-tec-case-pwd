import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from typing import Optional, Union
import polars as pl
import gc

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
    """
    Classe para descompactar arquivos ZIP e extrair somente arquivos de uma extensão específica,
    com destino configurável e diretório padrão no nível do projeto.
    """

    def __init__(self, default_dest: Optional[Union[str, Path]] = None):
        # Define diretório padrão: ou usuário especifica, ou ProjetRoot/data
        if default_dest:
            self.default_dest = Path(default_dest).resolve()
        else:
            # Assume este script em <project_root>/funcs/
            project_root = Path(__file__).resolve().parent.parent
            self.default_dest = project_root / "data"
        self.default_dest.mkdir(parents=True, exist_ok=True)

    def unzip_files(
        self,
        zip_path: Union[str, Path],
        extension: str,
        dest_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Extrai de um ZIP apenas arquivos com a extensão informada.

        Parameters
        ----------
        zip_path : str | Path
            Caminho para o arquivo .zip da competição.
        extension : str
            Extensão dos arquivos a extrair (por ex. '.parquet' ou 'csv').
        dest_dir : str | Path, optional
            Diretório de destino. Se None, usa o default configurado no init.
        """
        zip_path = Path(zip_path).resolve()
        # Definir destino final
        dest = Path(dest_dir).resolve() if dest_dir else self.default_dest
        dest.mkdir(parents=True, exist_ok=True)

        # Normaliza extensão (garantir ponto e lowercase)
        ext = extension if extension.startswith('.') else f'.{extension}'
        ext = ext.lower()

        with ZipFile(zip_path, 'r') as z:
            members = [m for m in z.namelist() if m.lower().endswith(ext) and "test" not in m.lower()]
            for member in members:
                with z.open(member) as src, open(dest / Path(member).name, 'wb') as dst:
                    dst.write(src.read())

        print(f"Extraídos {len(members)} arquivos '{ext}' em '{dest}'")


    @staticmethod
    def read_files(data_store: dict[pl.DataFrame]):
        pass

    @staticmethod
    def cast_types(data: pl.DataFrame) -> pl.DataFrame:
        for column in data.columns:
            if column[-1] in ("P", "A",):
                data = data.with_columns(pl.col(column).cast(pl.Float64).alias(column))
            elif column[-1] in ("D",):
                data = data.with_columns(pl.col(column).cast(pl.Date).alias(column))
            elif column in ("date_decision"):
                data = data.with_columns(pl.col(column).cast(pl.Date).alias(column))
            elif column[-1] in ("M",):
                data = data.with_columns(pl.col(column).cast(pl.Categorical).alias(column))
        return data


    @staticmethod
    def concatenate_files(
        source: Union[str, dict[str, pl.DataFrame]],
        prefix: Optional[str] = None,
        relaxed: bool = True,
    ) -> pl.DataFrame:
        """
        Lê todos os arquivos .parquet em `directory` (ou cujo nome comece com `prefix`)
        e os concatena verticalmente em um único DataFrame.

        Parameters
        ----------
        directory : Path
            Pasta onde estão seus .parquet.
        prefix : str, optional
            Se fornecido, só concatena arquivos cujo nome (stem) comece com esse prefixo.
        relaxed : bool
            Se True, usa `how="vertical_relaxed"` (aceita colunas faltantes e faz upcast automático).
            Se False, usa `how="vertical"` (requere esquema idêntico).

        Returns
        -------
        pl.DataFrame
        DataFrame resultante da concatenação.
        """

        pattern = f"{prefix}*.parquet" if prefix else "*.parquet"
        how = "vertical_relaxed" if relaxed else "vertical"

        if isinstance(source, dict):
            dfs = [
                df
                for name, df in source.items()
                if (prefix is None) or name.startswith(prefix)
            ]
            if not dfs:
                raise FileNotFoundError(f"Nenhum DataFrame em memória com prefixo='{prefix}'")
            return pl.concat(dfs, how=how)

        directory = Path(source)


        files = sorted(directory.glob(pattern))
        if not files:
            print(directory)
            raise FileNotFoundError(f"Nenhum arquivo encontrado no diretorio {directory} com prefixo = {prefix}")
        frames = [pl.scan_parquet(str(file)) for file in files]
        return pl.concat(frames, how=how).collect()

    @staticmethod
    def reduce_memory_usage(df: pl.DataFrame) -> pl.DataFrame:
        """
        Para cada coluna numérica inteira, faz downcast para o menor Int* que comporte o range de valores.
        Também converte Float64 → Float32 e Utf8 → Categorical.
        Retorna um novo DataFrame otimizado.
        """
        for name, dtype in zip(df.columns, df.dtypes):
            if dtype in (pl.Int64, pl.Int32, pl.Int16, pl.Int8,
                         pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8):
                # calcula min e max
                stats = df.select(
                    pl.col(name).min().alias("min"),
                    pl.col(name).max().alias("max")
                ).to_dicts()[0]
                mn, mx = stats["min"], stats["max"]
                if mn is None or mx is None:
                    continue
                target = None
                if dtype in (pl.UInt64, pl.UInt32, pl.UInt16, pl.UInt8):
                    if mx < 2**8:
                        target = pl.UInt8
                    elif mx < 2**16:
                        target = pl.UInt16
                    elif mx < 2**32:
                        target = pl.UInt32
                    else:
                        target = pl.UInt64
                else:
                    if mn >= -2**7 and mx < 2**7:
                        target = pl.Int8
                    elif mn >= -2**15 and mx < 2**15:
                        target = pl.Int16
                    elif mn >= -2**31 and mx < 2**31:
                        target = pl.Int32
                    else:
                        target = pl.Int64
                if target != dtype:
                    df = df.with_columns(pl.col(name).cast(target))
            elif dtype == pl.Float64:
                df = df.with_columns(pl.col(name).cast(pl.Float32))
            elif dtype == pl.Utf8:
                df = df.with_columns(pl.col(name).cast(pl.Categorical))

        return df


    @staticmethod
    def load_and_optimize_all(data_dir: Path, ids: list[int] = None) -> dict[str, pl.DataFrame]:
        """
        Lê todos os arquivos .parquet em data_dir (incluindo subpastas),
        aplica reduce_memory_usage e retorna dict[path_stem -> df].
        """
        optimized = {}
        for path in sorted(data_dir.rglob("*.parquet")):
            stem = path.stem
            # lazy scan + filter de case_id
            lf = pl.scan_parquet(str(path)).filter(pl.col("case_id").is_in(ids))
            # materializa e otimiza memória
            df = lf.collect()
            df = HandleData.reduce_memory_usage(df)
            optimized[stem] = df
            del df
            gc.collect()
        return optimized




class Aggregator:

    @staticmethod
    def aggregate_depth_tables(
        tables: Dict[str, pl.DataFrame],
        static_key: str
    ) -> Dict[str, pl.DataFrame]:
        aggregated: Dict[str, pl.DataFrame] = {}

        for name, df in tables.items():
            if name == static_key:
                continue

            df_norm = df  # garante definição antes de qualquer uso
            aggs = []

            for c in df_norm.columns:
                if c == "case_id":
                    continue

                if c.endswith(("P", "A")):
                    aggs += [
                        pl.col(c).max().alias(f"{c}_max"),
                        pl.col(c).last().alias(f"{c}_last"),
                        pl.col(c).mean().alias(f"{c}_mean"),
                    ]

                elif c.endswith("D"):
                    # primeiro garante que a coluna seja string
                    df_norm = df_norm.with_columns(
                        pl.col(c)
                          .cast(pl.Utf8)
                          .str.strptime(pl.Date, fmt="%Y-%m-%d")
                          .alias(c)
                    )
                    aggs += [
                        pl.col(c).max().alias(f"{c}_max"),
                        pl.col(c).last().alias(f"{c}_last"),
                        pl.col(c).mean().alias(f"{c}_mean"),
                    ]

                elif c.endswith("M"):
                    df_norm = df_norm.with_columns(
                        pl.col(c).cast(pl.Categorical)
                    )
                    aggs += [
                        pl.col(c).sort().last().alias(f"{c}_lexmax"),
                        pl.col(c).last().alias(f"{c}_last"),
                    ]

                else:
                    aggs += [
                        pl.col(c).max().alias(f"{c}_max"),
                        pl.col(c).last().alias(f"{c}_last"),
                    ]

            # faz a agregação lazy e coleta
            df_agg = df_norm.lazy().groupby("case_id").agg(aggs).collect()
            aggregated[name] = df_agg

        return aggregated

    @staticmethod
    def merge_all(
        tables: Dict[str, pl.DataFrame],
        aggregated: Dict[str, pl.DataFrame],
        static_key: str
    ) -> pl.DataFrame:
        base = tables[static_key]
        for df_agg in aggregated.values():
            base = base.join(df_agg, on="case_id", how="left")
        return base
