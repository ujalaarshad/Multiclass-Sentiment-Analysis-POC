import streamlit as st
import time
import os
import nltk
nltk.download('punkt_tab')  # Ensure NLTK resource is available

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from pydantic import BaseModel, Field, validator
from vector_store import vectorstore
from config import GOOGLE_API_KEY, OPENAI_API_KEY
from typing import List

# Allowed Categories
allowed_categories = [
    "Research labs", "Schedule exams", "Receive orientation to prepare for the exam",
    "Visit to the lab", "Parking the car", "Checking in and waiting",
    "The time waiting to do the exam", "Doing the collection on material or the exam",
    "Making the payment", "Receiving the results of the exam"
]

llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.1,
    max_retries=2,
    api_key=OPENAI_API_KEY
)

class CategoryPrediction(BaseModel):
    categories: List[str] = Field(description="List of Predicted categories for the comments")
    scores: List[float] = Field(description="List of Sentiment scores between 0-10 for the comments")
    translation: str = Field(description="English translation of the comment")

    @validator('scores', each_item=True)
    def score_range(cls, v):
        if v < 0 or v > 10:
            raise ValueError('Score must be between 0 and 10')
        return v

parser = PydanticOutputParser(pydantic_object=CategoryPrediction)
retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=llm, max_retries=1)

prompt_template = ChatPromptTemplate.from_template(
    """
    You are a specialized AI that analyzes Brazilian Portuguese comments along three dimensions: 
    1. Service Category Prediction  
    2. Integer Sentiment Scoring (0-10)  
    3. Cultural-Aware Translation

    ## ANALYSIS CONTEXT
    **Context & Metadata:**
    {context}

    ## INPUT COMMENT
    **Comment:**
    "{input_comment}"

    ## PROCESSING
    - Detect all applicable service categories by using the metadata and the context.
    - Provide a separate integer sentiment score (0-10) per category by analyzing the metadata of similar context and the english translation as well.
    - DO NOT OVERLAP THE CONTEXT WITH THE TRANSLATION OF THE COMMENT , ITS JUST GIVEN FOR SCORE RELEVANCE ANALYSIS
    - Translate the comment into English, preserving cultural nuances.
    - DO NOT ADD ANY OTHER CATEGORY OTHER THAN CATEGORIES PRESENT IN THE METADATA AND CONTEXT
    - ANALYZE CLOSELY THE METADATA AND CATEGORIES PRESENT IN THE METADATA AND CONTEXT
    **Format Instructions:** 
    {format_instructions}
    
    EXAMPLE OUTPUT: 
    categories: ['Receive orientation to prepare for the exam', 'Visit to the lab', 'Doing the collection on material or the exam', 'Making the payment'],scores:[3, 5, 3, 5], translation: "English translation here" 
    
    OUTPUT FORMAT (CSV): 
    categories:List, sentiment scores:List, translation:str
    """
)

def predict_category_score_translation(comment: str) -> tuple:
    docs = vectorstore.similarity_search(comment, k=5)
    contexts = "\n".join([doc.page_content for doc in docs])
    metadata_list = [doc.metadata for doc in docs]
    context_str = f"CONTEXT:\n{contexts}\n\nMETADATA:\n{str(metadata_list)}\n\n"
    formatted_prompt = prompt_template.format_messages(
        context=context_str,
        input_comment=comment,
        format_instructions=parser.get_format_instructions()
    )
    try:
        response = llm.invoke(formatted_prompt)
        if isinstance(response, list):
            content = response[0].content if hasattr(response[0], 'content') else str(response[0])
        else:
            content = response.content
        parsed = retry_parser.parse_with_prompt(content, formatted_prompt)
        
        # **Filter only allowed categories**
        filtered_categories = [
            category for category in parsed.categories if category in allowed_categories
        ]
        filtered_scores = [
            parsed.scores[i] for i in range(len(parsed.categories)) if parsed.categories[i] in allowed_categories
        ]

        return {
            "categories": filtered_categories,
            "scores": filtered_scores,
            "translation": parsed.translation
        }, context_str

    except Exception as e:
        return {"error": f"Failed to process comment: {str(e)}"}, context_str

st.title("Portuguese Comment Analyzer")
query = st.text_area("Enter your comment in Portuguese:")

if st.button("Analyze") and query:
    with st.spinner("Analyzing..."):
        result, context_str = predict_category_score_translation(query)
        time.sleep(2)
    st.divider()
    if "error" in result:
        st.error(result["error"])
    else:
        # Layout: Middle = Predicted Output, Right = Context & Metadata
        middle_col, right_col = st.columns([2, 1])
        
        with middle_col:
            st.subheader("Predicted Output")
            st.markdown("**Translation:**")
            st.write(result["translation"])
            st.divider()
            st.markdown("**Categories and Sentiment Scores:**")
            col1, col2 = st.columns(2)
            with col1:
                for category in result["categories"]:
                    st.write(f"- {category}")
            with col2:
                for score in result["scores"]:
                    color = "green" if score >= 7 else "orange" if score >= 4 else "red"
                    st.markdown(
                        f"<div style='display:inline-block; margin:5px; background-color:{color}; border-radius:50%; width:30px; height:30px; line-height:30px; text-align:center; color:white;'>{int(score)}</div>",
                        unsafe_allow_html=True
                    )
        
        with right_col:
            st.subheader("Context & Metadata")
            with st.expander("Show Full Context & Metadata"):
                st.text(context_str)
