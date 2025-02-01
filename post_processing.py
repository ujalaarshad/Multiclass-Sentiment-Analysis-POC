import pandas as pd

def process_excel_file(file_path, sheet_name=0):
    
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    categories = [
        "Research labs", "Schedule exams", "Receive orientation to prepare for the exam",
        "Visit to the lab", "Parking the car", "Checking in and waiting",
        "The time waiting to do the exam", "Doing the collection on material or the exam",
        "Making the payment", "Receiving the results of the exam"
    ]
    
    for category in categories:
        if category in df.columns:
            # Replace -1 with an empty string
            df[category] = df[category].replace(-1, "")
        else:
            # If the column does not exist, create it with default empty string values
            df[category] = ""
    
    
    return df

# Example usage:
if __name__ == "__main__":
    input_file = "labeled_patient_reviews.xlsx"  # Replace with your actual file path
    output_file = "processed_excel_file.xlsx"
    
    processed_df = process_excel_file(input_file)
    processed_df.to_excel(output_file, index=False)
    print(f"Processed Excel file saved as {output_file}")
