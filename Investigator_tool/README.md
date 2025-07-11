# 🕵️‍♀️ Financial Crime Investigation Tool

This is a **Streamlit-based AI assistant** for identifying and classifying suspicious customer behavior in financial transactions.  
It uses **OpenAI GPT-3.5** to analyze transaction patterns, detect potential **money laundering** activities, and classify them under **known AML typologies**.

---

## 🚨 Features

🔎 **Financial Crime Detection**  
- Detects suspicious vs. genuine transaction patterns  
- Classifies customers under AML typologies like:
  - Smurfing  
  - Mule account  
  - Layering  
  - Human trafficking  
  - Tax evasion  
  - No crime  

📊 **Risk Indicators & Justification Summary**  
- Extracts **risk indicators** from transaction behavior  
- Provides a **detailed bullet-point summary** explaining why a crime was flagged

🧠 **Powered by OpenAI GPT-3.5**  
- Uses a carefully crafted prompt that simulates expert financial crime investigator logic

💻 **Interactive Streamlit App**  
- Upload or paste transaction details  
- Get instant classification & reasoning in JSON and tabular format

---

## 🏗️ Final interface with results

<img width="1919" height="990" alt="image" src="https://github.com/user-attachments/assets/163e5cf4-eefe-4979-ad60-71c7c850a01e" />

## Instructions to run

- To use this model, use GPT-3.5-Turbo openai API key
- Download required packages - json, openai==0.2, streamlit, pandas 
- To get the app running - On the terminal streamlit run <folder_path>/openai_helper.py


