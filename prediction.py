import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from pydantic import BaseModel, Field, validator
from vector_store import vectorstore
from config import GOOGLE_API_KEY, OPENAI_API_KEY
from typing import List
import pandas as pd
# Initialize LLM
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash-exp",
#     google_api_key=GOOGLE_API_KEY,
#     temperature=0.3,
#     max_retries=2
# )
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4,
    max_retries=2,
    api_key=OPENAI_API_KEY
)

# Define output validation model
class CategoryPrediction(BaseModel):
    categories: List[str] = Field(description="List of Predicted categories for the comments")
    scores: List[float] = Field(description="List of Sentiment scores between 0-10 for the comments")
    translation: str = Field(description="English translation of the comment")

    @validator('scores', each_item=True)
    def score_range(cls, v):
        if v < 0 or v > 10:
            raise ValueError('Score must be between 0 and 10')
        return v

# Initialize output parser with retry
parser = PydanticOutputParser(pydantic_object=CategoryPrediction)
retry_parser = RetryWithErrorOutputParser.from_llm(
    parser=parser,
    llm=llm,
    max_retries=1
)


prompt_template = ChatPromptTemplate.from_template(
    """
    # PORTUGUESE COMMENT ANALYZER ROLE
    You are a specialized AI that analyzes Brazilian Portuguese comments across three dimensions: 
    1. Service Category Prediction
    2. Integer Sentiment Scoring (0-10)
    3. Cultural-Aware Translation

    ## ANALYSIS CONTEXT
    **Relevant Context & Metadata:**
    {context}

    ## INPUT COMMENT
    **Target for Analysis:**
    "{input_comment}"

    ## PROCESSING PIPELINE
    
    ### 1. CATEGORY DETECTION
    - Mandatory Multi-Category Selection:
      * Always select ALL applicable categories from metadata
      * Prioritize maximum relevant categories
      * Never merge categories
    
    ### 2. SENTIMENT SCORING (STRICT 0-10 INTEGERS)
    
    **Scoring Rules:**
    - Base score on metadata patterns
    - Each category gets separate integer score
    - Match score count to category count
    
    ### 3. TRANSLATION REQUIREMENTS
    - Preserve cultural context
    - Maintain original emotional tone
    - Use service-industry specific terminology

    ## OUTPUT REQUIREMENTS
    **Format Instructions:** 
    {format_instructions}

    **Validation Checks:**
    [✓] Exactly 1 score per category
    [✓] All scores 0-10 integers
    [✓] No extra categories beyond metadata
    [✓] Add all the categories in the metadata and score them
    ## STRICT OUTPUT FORMAT (CSV): 
    categories:List, sentiment scores:List, translation:str
    """
)

def predict_category_score_translation(comment: str) -> dict:
    # Retrieve relevant documents
    relevant_docs = vectorstore.similarity_search(comment, k=5)
    contexts = "\n".join([doc.page_content for doc in relevant_docs])
    metadata_list = [doc.metadata for doc in relevant_docs]
    
#     print(metadata_list)
    contexts = f"CONTEXT{contexts}\n\n METADATA{metadata_list}\n\n"
    # Format prompt with context and comment
    formatted_prompt = prompt_template.format_messages(
        context=contexts,
        input_comment=comment,
        format_instructions=parser.get_format_instructions()
    )
    # print(formatted_prompt)
    try:
        # Get initial prediction
        response = llm.invoke(formatted_prompt)
        
        # Parse and validate with retry
        parsed = retry_parser.parse_with_prompt(response.content, formatted_prompt)
        
        return parsed.dict()
    except Exception as e:
        return {"error": f"Failed to process comment: {str(e)}"}


def update_dataframe_with_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Updates the DataFrame with predictions and translations, handling multiple categories per comment.
    Adds a 2-second delay after processing each row to comply with API rate limits.
    """
    # Initialize category columns with -1 (default value)
    categories = [
        "Research labs", "Schedule exams", "Receive orientation to prepare for the exam",
        "Visit to the lab", "Parking the car", "Checking in and waiting",
        "The time waiting to do the exam", "Doing the collection on material or the exam",
        "Making the payment", "Receiving the results of the exam"
    ]
    
    # Initialize all category columns if they don't exist
    for category in categories:
        if category not in df.columns:
            df[category] = -1
    
    # Initialize English Translation column
    if "English Translation" not in df.columns:
        df["English Translation"] = ""

    # Process each row with rate limiting
    for index, row in df.iterrows():
        comment = row["Comments"]
        result = predict_category_score_translation(comment)
        
        if "error" not in result:
            # Process multiple categories and scores
            for category, score in zip(result["categories"], result["scores"]):
                if category in categories:
                    df.at[index, category] = int(score)
                else:
                    print(f"Warning: Unexpected category '{category}' at index {index}")
            
            # Update translation
            df.at[index, "English Translation"] = result["translation"]
        

        print(f"Processed index {index+1}/{len(df)} - {result}")
    
    return df




import time
import pandas as pd

def update_dataframe_with_predictions_insample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Updates the DataFrame with predictions and translations for in-sample data.
    Handles multiple categories and scores, and adds a 2-second delay after processing each row
    to comply with API rate limits.

    Args:
        df (pd.DataFrame): Input DataFrame containing a 'Comments' column.

    Returns:
        pd.DataFrame: Updated DataFrame with predictions, scores, and translations.
    """
    # Define all possible categories
    categories = [
        "Research labs", "Schedule exams", "Receive orientation to prepare for the exam",
        "Visit to the lab", "Parking the car", "Checking in and waiting",
        "The time waiting to do the exam", "Doing the collection on material or the exam",
        "Making the payment", "Receiving the results of the exam"
    ]

    # Initialize category columns with -1 (default value for unmentioned categories)
    for category in categories:
        if category not in df.columns:
            df[category] = -1

    # Add new columns for predictions and translation
    df["predicted_categories"] = ""  # Stores all predicted categories as a list
    df["predicted_scores"] = ""      # Stores all predicted scores as a list
    df["English Translation"] = ""   # Stores the English translation

    # Apply predictions to each comment
    for index, row in df.iterrows():
        comment = row["Comments"]
        result = predict_category_score_translation(comment)

        if "error" not in result:
            # Extract results
            predicted_categories = result.get("categories", [])
            predicted_scores = result.get("scores", [])
            translation = result.get("translation", "")

            # Update category scores in their respective columns
            for category, score in zip(predicted_categories, predicted_scores):
                if category in categories:
                    df.at[index, category] = int(score)  # Ensure scores are integers

            # Update the new columns for predictions and translation
            df.at[index, "predicted_categories"] = predicted_categories
            df.at[index, "predicted_scores"] = predicted_scores
            df.at[index, "English Translation"] = translation

        # Print progress
        print(f"\nProcessed Index: {index + 1} with result: {result}\n")


    return df