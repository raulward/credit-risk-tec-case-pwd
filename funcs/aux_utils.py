import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
