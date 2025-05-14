🏥 Personalized Health Recommendation System – RAG + Gemini

A full-stack AI health dashboard that empowers patients and admins to track medical data, analyze risk, and query past records using Retrieval-Augmented Generation (RAG) with Large Language Models (LLMs). Built with Streamlit, MongoDB, Google Gemini API, and Sentence-Transformers.

⸻

🚀 Features

👨‍⚕️ Patient Side
	•	Account Creation with detailed personal and clinical information
	•	PDF Medical Record Upload with automatic text extraction
	•	BMI, Blood Pressure, Heart Rate Calculations and category classification
	•	Clustering-based Risk Assessment using KMeans-style prediction
	•	Personalized Recommendations from Gemini (LLM)
	•	Longitudinal Report Storage with timeline of past health evaluations
	•	RAG-Based Report Querying – “What was my cholesterol trend?”
	•	Delete + Download Past Reports

🛠 Admin Panel
	•	Admin-only access via login form
	•	Global Platform Metrics: Average BMI, HR, Age
	•	Distributions: BMI, Age, Risk Category Pie Chart
	•	Semantic Search Across All Reports – “List users with signs of kidney failure”
	•	Per-User Report Browsing
	•	Per-User Risk Clustering & Medical History

🧠 Built-in NLP/ML Intelligence
	•	RAG with Sentence-Transformers + Gemini
	•	Trend Summarization across multiple time-series PDF reports
	•	Abbreviation Expansion for better LLM context


⸻

🧰 Tech Stack

Layer	Tools/Frameworks
Frontend	Streamlit
Backend	Python + MongoDB
LLM	Google Gemini 1.5 Flash
RAG	SentenceTransformers (MiniLM-L6)
NLP	SciSpacy, PyPDF2
Viz	Plotly + Streamlit Charts


⸻

📁 Folder Structure

📦 health-rag
├── app.py                # Main application logic
├── llm_utils.py          # Gemini interaction and prompts
├── clustering.py         # Risk prediction logic
├── pdf_utils.py          # PDF text extraction logic
├── requirements.txt      # Dependencies
├── .env                  # API key securely loaded
├── screenshots/          # Add screenshots here for README
└── sample_reports/       # Test PDFs for users


⸻

🔑 Setup & Installation

# Clone the repo


# Create & activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add Gemini API Key to .env
GOOGLE_API_KEY=your_api_key_here

# Run the app
streamlit run app.py


⸻

🔍 Example Queries (RAG)

User Query	Response Type
“What was my BP in Jan and May?”	Date-based trend extraction
“Was I ever anemic?”	Condition-based answer
“Compare my last two lab reports”	Summary comparison
“Any critical alerts in my history?”	Risk-based detection


⸻

🧪 Evaluation (BLEU, ROUGE, Precision@K)

Metric	Description
BLEU	Overlap with human-generated answers
ROUGE	Recall-style n-gram match
Precision@K	Top-K relevant report retrieval
Faithfulness	Answer grounded in context (LLM eval)

Run eval_script.py to compute metrics for sample QA pairs.

⸻

📌 Roadmap
	•	User + Admin panel
	•	RAG with Gemini + Semantic Search
	•	Named Entity Recognition for medical terms
	•	Per-user historical reports
	•	Export report as PDF with chart
	•	CI/CD with unit tests

⸻


🧾 License

MIT License. Use for academic, educational, or clinical research purposes with attribution.
