import pandas as pd
from config import DATASET_PATH, SHEET_NAME

def load_dataset():
    df = pd.read_excel(DATASET_PATH, sheet_name=SHEET_NAME)
    df = df.fillna(-1)
    review_column = "Comments"
    category_columns = df.columns[4:14]  # Columns E to M contain categories

    labeled_data = []
    for _, row in df.iterrows():
        
        review = row[review_column]
        categories = {col: int(row[col]) for col in category_columns if int(row[col]) > -1}  # Keep non-zero scores
        
        if categories:
            labeled_data.append({"review": review, "categories": list(categories.keys()), "scores": list(categories.values())})
    
    return labeled_data, df.iloc[200:]  # Return labeled data and unlabeled reviews
