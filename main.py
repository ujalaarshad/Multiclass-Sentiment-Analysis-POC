
from data_loader import load_dataset
from vector_store import store_embeddings
from prediction import update_dataframe_with_predictions
import pandas as pd

# Load dataset
labeled_data, unlabeled_df = load_dataset()

# Store labeled data embeddings in Pinecone
# store_embeddings(labeled_data)

# # Update the DataFrame with predictions
unlabeled_df = update_dataframe_with_predictions(unlabeled_df)

# Save results
unlabeled_df.to_excel('labeled_patient_reviews.xlsx', index=False)
print("Labeling complete. File saved as labeled_patient_reviews.xlsx")