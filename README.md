Personalized Health Recommendation System

A full-stack AI platform that empowers patients and healthcare administrators with personalized, document-aware medical insights using cutting-edge NLP, clustering, and retrieval-based technologies.

⸻

🚀 Overview

This system blends structured health inputs (age, BMI, vitals) with unstructured medical documents (PDFs) to:
<pre>
	•	Generate AI-powered health reports using Google Gemini 1.5
	•	Enable natural language medical Q&A via a RAG (Retrieval-Augmented Generation) pipeline
	•	Offer a semantic search-enabled admin dashboard for querying health records
</pre>



⸻

🎯 Key Features

👤 Patient-Side
<pre>
	•	Secure Account Creation: Register with demographic and medical data
	•	Health Data Form: Input structured vitals and symptoms
	•	PDF Uploads: Attach lab reports or prescriptions for AI parsing
	•	AI-Generated Reports: Get interpretive reports personalized to your data
	•	RAG Chatbot: Ask questions like “What did my May report say about cholesterol?” and receive document-grounded answers
	•	General Health Assistant: Chat with Gemini for lifestyle or condition-related advice
	•	Past Report Viewer: Timeline with ability to download or delete individual reports
</pre>

🛠️ Admin-Side
<pre>
	•	Role-Based Login
	•	User Management: View/edit user health profiles
	•	Report Timeline Access
	•	Semantic Search: Run plain-English queries like “List users with high BMI”
	•	Statistics Dashboard: Platform-level health analytics (age, BMI, risk clusters)
</pre>

⸻

🧠 AI/ML & Deep Learning Components

📝 Gemini 1.5 for Report Generation
<pre>
	•	Fusion of structured metrics + parsed PDF data
	•	Personalized and readable
	•	Categories: Vitals Summary, Insights, Recommendations, Risk Category
</pre>

🗂️ Clustering (KMeans-style)
<pre>
	•	Risk stratification: Low, Moderate, High
	•	Inputs: Symptoms, BMI, HR, BP, etc.
	•	Used in report summaries + admin analytics
</pre>

🤖 Retrieval-Augmented Generation (RAG)
<pre>
	•	Embeds PDF chunks using MiniLM
	•	Semantic retrieval with SentenceTransformers
	•	Gemini generates grounded responses from relevant report sections
</pre>

🧵 Semantic Search (Admin)
<pre>
	•	All reports parsed and indexed
	•	Plain queries like “Find users with high BP and low hemoglobin”
	•	Powered by cosine similarity + metadata tagging
</pre>

⸻

🧪 Evaluation Metrics
<pre>
Metric	Value/Insight
BLEU-4	0.62 – lexical overlap with gold answers
ROUGE-L	0.71 – recall of critical segments
Precision@K	0.87 – relevance of top-3 retrieved chunks
Faithfulness	82% correct (manual review, no hallucination)
</pre>

⸻

💡 Sample Use Cases
<pre>
Query	Response Type
What was my BP last year?	Date-filtered stat summary
Compare my last two tests	Multi-report synthesis
Do I show signs of diabetes?	RAG-based explanation
Find users with high BMI	Admin semantic search
</pre>

⸻

🛠️ Tech Stack
<pre>
	•	Frontend: Streamlit (interactive dashboards + chat)
	•	Backend: Python
	•	Database: MongoDB (user profiles + report timeline)
	•	AI/LLM: Google Gemini 1.5
	•	Embeddings: SentenceTransformers (MiniLM)
	•	Clustering: KMeans-like symptom-based categorization
	•	PDF Parsing: PyPDF2
	•	Visualization: Plotly
</pre>

⸻

📁 Folder Structure

<pre>
DLProject/
├── app.py               # Main Streamlit frontend
├── clustering.py        # Clustering logic
├── llm_utils.py         # Gemini and RAG handlers
├── pdf_utils.py         # PDF parsing logic
├── .env                 # API Keys
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
├── /reports             # Saved health reports (PDFs)
└── /screenshots         # Screenshots for documentation
```
</pre>


⸻

📷 Screenshots

<img width="1405" alt="Screenshot 2025-05-14 at 7 04 11 PM" src="https://github.com/user-attachments/assets/97624ed2-511c-496b-a16e-92b2b27996b3" />

<img width="1170" alt="Screenshot 2025-05-14 at 1 00 55 PM" src="https://github.com/user-attachments/assets/28a1c5f4-a7ac-428b-808d-5ab19707b6f0" />

<img width="1111" alt="Screenshot 2025-05-14 at 1 01 27 PM" src="https://github.com/user-attachments/assets/20e02e68-877a-4ebe-a395-81c588874fb1" />


<img width="1121" alt="Screenshot 2025-05-14 at 1 04 58 PM" src="https://github.com/user-attachments/assets/343b230f-bf2f-4ba6-8dee-c5136454c7e2" />


<img width="1138" alt="Screenshot 2025-05-14 at 1 01 41 PM" src="https://github.com/user-attachments/assets/8adaa5a4-6145-4a5e-ae8a-41c1a6c3f69f" />




⸻

🧾 Installation & Run Instructions

# 1. Clone the repository
(https://github.com/DevMewada1299/Personalized-health-recommendation-System.git)
cd health-rag

# 2. Set up Python virtual environment
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Gemini API key in a .env file
GOOGLE_API_KEY=your_api_key_here

# 5. Run the app
streamlit run app.py


⸻



🔄 RAG + Gemini Prompt Example

Context:
- Glucose: 210 mg/dL (elevated)
- TSH: 4.8 uIU/mL

User Question:
"Was my sugar high in May?"

Gemini Response:
"Your May report shows glucose at 210 mg/dL, which is borderline high. Consider lifestyle changes and consult a doctor."


⸻

📌 Roadmap
<pre>
	•	Add PDF-to-report generation
	•	Implement user RAG Q&A
	•	Semantic Search (Admin)
	•	Visual timeline trends (user & admin)
	•	Multi-modal record summarization
	•	Integrate vitals from wearable APIs
</pre>


⸻

📄 License

This project is for educational and non-commercial research purposes only.
