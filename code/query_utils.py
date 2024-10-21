import pandas as pd
import numpy as np
import ast

def convert_list_columns_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts columns in the DataFrame that contain lists (or arrays) of dictionaries
    into strings. The function joins the 'label' key from each dictionary in the list
    into a comma-separated string.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame that may contain columns with lists or arrays of dictionaries.

    Returns:
    --------
    pd.DataFrame
        The modified DataFrame where list columns have been converted to strings.

    Example:
    --------
    If a column contains values like [{'label': 'A'}, {'label': 'B'}], it will be
    converted to a string like 'A, B'.
    """
    df = df.copy()
    for column in df.columns:
        # Check if the first value in the column is a list or numpy array
        if isinstance(df[column].values[0], (list, np.ndarray)):
            # Convert the list of dictionaries into a comma-separated string
            df[column] = df[column].apply(
                lambda x: ", ".join([item.get("label", "") if isinstance(item, dict) else "" for item in x])
            )
    return df


def from_collection_datasets_to_markdown(datasets: pd.DataFrame) -> str:
    base_url = "https://cellxgene.cziscience.com/e/"
    datasets["id"] = datasets["id"].apply(lambda id: f"[{id}]({base_url}/{id}.cxg)")
    table = datasets[
        [
            "id",
            "name",
            "disease",
            "organism",
            "primary_cell_count",
            "sex",
            "tissue",
        ]
    ].pipe(convert_list_columns_to_str)
    return table.to_markdown(index=False)


def extract_dictionary_from_response(result: str) -> dict:
    try:
        # Remove the '```python' part if it's there
        result = result.replace("```python", "```")

        # Check if there are triple backticks indicating code
        if "```" in result:
            # Split and get the part between triple backticks
            result = result.split("```")[1].strip()

        # Use ast.literal_eval for safe evaluation of the string
        params = ast.literal_eval(result)

        # Ensure the result is a dictionary
        if isinstance(params, dict):
            return params
        else:
            raise ValueError("Extracted content is not a dictionary.")
    
    except (SyntaxError, ValueError, IndexError) as e:
        # Handle cases where the input is malformed or not a valid dictionary
        print(f"Error while extracting dictionary: {e}")
        return {}