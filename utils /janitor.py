import pandas as pd
import re


class Janitor:
    def __init__(self):
        pass

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Apply all cleaning procedure in sequence
        df = df.copy()  # First make a copy to preserve integrity of the old df
        df = self.drop_duplicates(df)
        df = self.snake_case_columns(df)
        df = self.fix_none(df)
        df = self.fix_datatypes(df)
        df = self.dropna_target(df)
        df = df.reset_index(drop=True)  # Fix index
        return df

    def drop_duplicates(self, df):
        return df.drop_duplicates() if df.duplicated().sum() > 0 else df

    def snake_case_columns(self, df):
        pattern = r'(?<!^)(?=[A-Z][a-z])|(?<=[a-z])(?=[A-Z])'
        df.columns = [re.sub(pattern, '_', column).lower()
                      for column in df.columns]
        return df

    def fix_none(self, df):
        def replace_none(value):
            like_nan = {'none', ''}
            if pd.isnull(value) or (isinstance(value, str) and (value.lower().strip() in like_nan)):
                value = pd.NA
            return value

        return df.map(replace_none)

    def fix_datatypes(self, df):
        columns_int = ['prg', 'pl', 'pr', 'sk', 'ts', 'age']
        columns_float = ['m11', 'bd2']
        col_to_fix = {col for col in columns_int+columns_float}
        if col_to_fix.issubset(df.columns):
            df[columns_int] = df[columns_int].astype(int)
            df[columns_float] = df[columns_float].astype(float)
        return df

    # Drop rows with missing values in target column and reset index
    def dropna_target(self, df):
        return df.dropna(subset='sepsis') if 'sepsis' in df.columns else df
