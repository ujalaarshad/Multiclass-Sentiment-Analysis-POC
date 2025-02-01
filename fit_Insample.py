from data_loader import load_dataset
from vector_store import store_embeddings
from prediction import update_dataframe_with_predictions_insample
import pandas as pd
import os
df = pd.read_excel(os.environ['DATASET_PATH'], sheet_name=os.environ['SHEET_NAME']).iloc[:200]
labeled_data = update_dataframe_with_predictions_insample(df)

# Save results
labeled_data.to_excel('In_sample_labeled_patient_reviews.xlsx', index=False)
print("Labeling complete. File saved as In_sample_labeled_patient_reviews.xlsx")