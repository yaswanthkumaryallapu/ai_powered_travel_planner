import os
import streamlit as st
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from PIL import Image

# ✅ Set Up API Key
os.environ["GOOGLE_API_KEY"] = "API_Key_Here"

# ✅ Load Banner Image
banner = Image.open("banner.jpg")

# ✅ Initialize LangChain Model with Google Gemini
llm = GoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=os.environ["GOOGLE_API_KEY"])

# ✅ Define Prompt Template for LangChain
travel_prompt = PromptTemplate(
    input_variables=["from_location", "to_location"],
    template="""
    You are an AI-powered travel planner. Plan a trip from {from_location} to {to_location}.
    Include:
    - first give the estimated more accurate total budget
    - Travel options (flight/train/bus) with estimated costs must be more accuracte
    - Hotel recommendations and estimated prices
    - Must-visit attractions and activities
    - Total budget breakdown

    Format the response in a structured and professional way.
    """
)

# ✅ Create LangChain Travel Planning Chain
travel_chain = travel_prompt | llm | RunnablePassthrough()

# ✅ Streamlit UI Setup
st.set_page_config(page_title="AI Travel Planner", layout="wide")

# ✅ Display Banner Image
st.image(banner, use_container_width=True)

st.markdown("<h1 style='text-align: center; color: #0078D7;'>🌍 AI-Powered Travel Planner</h1>", unsafe_allow_html=True)

# ✅ Input Fields
col1, col2 = st.columns(2)
with col1:
    from_location = st.text_input("🏠 From (Location):", placeholder="Enter your starting location")

with col2:
    to_location = st.text_input("📍 To (Destination):", placeholder="Enter your destination")

# ✅ Predict Button
if st.button("🚀 Generate Travel Plan"):
    if from_location and to_location:
        with st.spinner("Generating your travel plan..."):
            result = travel_chain.invoke({"from_location": from_location, "to_location": to_location})
        st.success("Here is your travel plan:")
        st.write(result)
    else:
        st.error("Please enter both locations before generating the plan.")
