import pandas as pd
import numpy as np
from  typing import Union
import  seaborn as sns
import  matplotlib.pyplot as plt

def count_plot(
        *,
        df: pd.DataFrame,
        columns: list,
        nrows: int,
        ncols: int,
        figsize: tuple = (17, 10),
        hue: Union[None, str] = None,
        #   hspace: float = 0.3,
        #   wspace: float = 0.2,
        title_size: int = 15,
        xlabel_size: int = 15,
        ylabel_size: int = 15,
        rotation: Union[None, int] = None,
        x_labelsize: int = 12,
        color: str = "#6e917f",
) -> "Plot":
    """ This fuction returns multiple bar chart simultaneously
        ==========================================================
        Parameters
        -> df: The dataframe to be used
        -> data: list of columns to be use
        -> nrows: number of row
        -> n_cols: number of columns
           the product of the number of row and columns should not be less than the size of the data listed
        -> figzise: The size of the figures, width and height as tuple, by default is 17 width, 10 height
        -> hue: Additional information to the chart, it is like a legend, it is None by default
        -> hspace: height space, helps between the rows, by  0.3 by default
        -> wspace: width space, helps between the columns, by  0.1 by default
        -> xlabel_size: the size of the x label, 15 by default
        -> ylabel_size: the size of the y label, 15 by default
        -> rotation: the number of degrees to be rotated for x label, 0 by default
        -> labelsize: size of the label when it is set to rotate

    """

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize)
    for idx, var in enumerate(columns):
        if nrows > 1:
            ax = axs[
                (idx // ncols), (idx % ncols)
            ]  # calculating and finding the axis and  then store it
        else:
            ax = axs[idx]

        sns.countplot(data=df, x=var, ax=ax, color=color)
        ax.set_title(f"Distribution of {var!r}", size=title_size)
        ax.set_xlabel(var, size=xlabel_size)
        ax.set_ylabel("count", size=ylabel_size)
        #  plt.subplots_adjust(hspace=hspace, wspace=wspace)
        ax.tick_params(axis="x", rotation=rotation, labelsize=x_labelsize)

        # Annotate the chart
        for bar in ax.patches:
            x_val = bar.get_x() + bar.get_width() / 2  # x pos
            y_val = bar.get_height()  # y pos
            ax.annotate(
                text=y_val,  # text pos
                xy=(x_val, y_val),  # (x, y)
                xytext=(0, 6),  # text position
                ha="center",  # horizontal alignment
                va="center",  # vertical alignment
                size=10,  # text size
                textcoords="offset points",
            )
        plt.tight_layout()


def replace_question_mark(data: pd.DataFrame) -> pd.DataFrame:
    """" This function accept one parameter, a datafram.
         It checks to see if a variable contains contains ? and then replace the label with mode of the variable
         """
    for i in data.select_dtypes(include="O").columns:
        # get the mode
        get_mode = data[i].mode()[0]
        data[i] = np.where(data[i].str.contains("\?"), get_mode, data[i])

    # return data


def descritization(
    *, data: pd.DataFrame, variable: str, bins: int, labels: list[Union[str, int]],
) -> pd.DataFrame:
    """
    this function accepts four parameters and return the decritized variable
    ==== parameters ====
    -> data = the dataframe
    -> variabe = the column to be descritize
    -> bins = number of bins to used
    -> labels = label to used
    """
    data[f"{variable}_binned"] = pd.qcut(
        x=data[variable], q=bins, labels=labels, duplicates="drop"
    )

    # dropping the old column
    data.drop(columns=[variable], inplace=True)


# return data

def replacing_values(
    *, data: pd.DataFrame, variable: str, search: list, replace: str
) -> pd.DataFrame:
    """
    this function accepts three parameters and return the decritized variable
    ==== parameters ====
    -> variabe = the column to be replace
    -> search = the label(s) to search in a list
    -> replace = label to repace
    """

    data[variable] = np.where(data[variable].isin(search), replace, data[variable])
    # return data