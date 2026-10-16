
import pandas as pd

def load_flight_data(filepath):
    """
    Load flight data from a CSV file.

    Parameters:
    filepath (str): The path to the CSV file containing flight data.

    Returns:
    pd.DataFrame: A DataFrame containing the flight data.
    """
    df = pd.read_csv(filepath)
    df = df[["Year", "Month", "Flt"]]

    df["Flt"] = (
        df["Flt"]
        .str.replace(",", "", regex=False)
        .astype(int)
    )

    df["Date"] = pd.to_datetime(
        dict(
            year=df["Year"],
            month=df["Month"],
            day=1
        )
    )

    df = df.sort_values("Date").reset_index(drop=True)
    df = df[["Date", "Flt"]]
 
    return df
