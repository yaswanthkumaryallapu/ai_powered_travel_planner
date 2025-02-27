# **AI-Powered Travel Planner**  

An AI-powered travel planning application that helps users find optimal travel options between two locations. It utilizes **LangChain** and **Google Gemini AI** to generate travel recommendations, including flights, trains, buses, and estimated costs.  

---

## **🚀 Features**  
✅ **AI-Powered Travel Suggestions** – Provides multiple travel options (flight, train, bus, cab) with estimated costs.  
✅ **Accommodation Recommendations** – Suggests hotels with price estimates.  
✅ **Attractions & Activities** – Lists top places to visit.  
✅ **Budget Breakdown** – Estimates total travel costs.  
✅ **User-Friendly Interface** – Built with **Streamlit** for an intuitive experience.  

---

## **🛠 Tech Stack**  
- **Programming Language**: Python  
- **User Interface**: Streamlit  
- **Framework**: LangChain  
- **AI Model**: Google Gemini AI  
- **Image Processing**: Pillow  

---

## **📂 Folder Structure**  
```
/AI_Travel_Planner
│-- app.py               # Main Streamlit app
│-- banner.jpg           # UI banner image
│-- requirements.txt     # Dependencies list
│-- README.md            # Project documentation
```

---

## **📦 Installation**  
### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/yaswanthkumaryallapu/ai_powered_travel_planner.git
cd ai_powered_travel_planner
```

### **2️⃣ Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3️⃣ Set Up Google Gemini API Key**  
1. Get your **API key** from [Google AI Studio](https://aistudio.google.com/).  
2. Add it to your environment variables:  
   ```bash
   export GOOGLE_API_KEY="your-api-key"
   ```

---

## **🚀 Running the Application**  
```bash
streamlit run app.py
```
The app will open in your **browser**.

---

## **🖼️ User Interface Preview**  
![Banner](banner.jpg)

---

## **📜 License**  
This project is licensed under the **MIT License**.
