import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from zipfile import ZipFile
from typing import Optional, Union
import polars as pl

from typing import List, Tuple

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
            members = [m for m in z.namelist() if m.lower().endswith(ext)]
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
                    data = data.with_columns(pl.col(column))
            return data
