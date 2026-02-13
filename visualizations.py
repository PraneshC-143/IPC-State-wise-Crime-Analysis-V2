import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def create_bar_chart(data, x_column, y_column, title, x_label, y_label):
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x=x_column, y=y_column)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


def create_line_chart(data, x_column, y_column, title, x_label, y_label):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x=x_column, y=y_column)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()


# Example usage
# df = pd.DataFrame({'Year': [2020, 2021, 2022], 'Crime Rate': [200, 250, 300]})
# create_bar_chart(df, 'Year', 'Crime Rate', 'Crime Rate over Years', 'Year', 'Crime Rate')
# create_line_chart(df, 'Year', 'Crime Rate', 'Crime Rate over Years', 'Year', 'Crime Rate')