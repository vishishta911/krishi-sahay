# 🌾 Kisan Call Centre Query Assistant

**An AI-powered agricultural Q&A system for instant farmer support.**

Combines fast semantic search with optional LLM enhancement to deliver offline-first answers backed by official Kisan Call Centre data.

---

## 🎯 Project Objective

Help farmers get accurate, instant answers to agricultural questions using:
- **Offline Mode:** Search pre-indexed Q&A knowledge base (no internet required)
- **Online Mode:** Enhance answers with AI-powered explanations (optional, requires API)

**Key Benefits:**
- ✅ Fast responses (~100ms for offline search)
- ✅ Works without internet connection
- ✅ Grounds answers in verified agricultural data
- ✅ Optional AI enhancement for deeper explanations
- ✅ Clean web interface for easy access

---

## 🔄 Offline vs Online Workflow

### 📚 Offline Mode (Semantic Search)
1. User submits agricultural question
2. System searches pre-built FAISS index
3. Retrieves 5 most similar Q&A pairs from knowledge base
4. Displays relevant answers with relevance scores
5. **No internet, no costs, instant response**

### 🤖 Online Mode (RAG Enhancement)
1. Offline search runs first (always)
2. Retrieved answers used as context
3. LLM generates personalized explanation based on context
4. Both offline and AI answers displayed together
5. **Optional, requires IBM Watsonx credentials**

### Recommended Workflow
```
User Query
    ↓
[FAISS Offline Search] ← Fast, no internet
    ↓
[Display Offline Results] ← Always shown
    ↓
[Optional: LLM Enhancement] ← If enabled
    ↓
[Display AI Response] ← If LLM available
```

---

## 💻 Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector DB** | FAISS | Fast semantic similarity search |
| **Embeddings** | Sentence-Transformers | Convert text to semantic vectors |
| **LLM** | IBM Watsonx Granite | AI-powered answer generation |
| **Web UI** | Streamlit | Clean, interactive interface |
| **Data Processing** | Pandas, NumPy | Clean and prepare Q&A data |
| **Language** | Python 3.9+ | Core implementation |

---

## 📁 Folder Structure

```
kisan-call-centre-assistant/
├── README.md                    # This file
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables
│
├── data/                       # Q&A data files
│   ├── raw_kcc.csv
│   ├── clean_kcc.csv
│   └── kcc_qa_pairs.json
│
├── embeddings/                 # Vector embeddings
│   └── kcc_embeddings.pkl
│
├── vector_store/               # FAISS index + metadata
│   ├── faiss.index
│   └── meta.pkl
│
├── models/                     # LLM client
│   └── granite_llm.py
│
├── services/                   # Core processing
│   ├── preprocess_data.py
│   ├── generate_embeddings.py
│   └── semantic_search.py
│
├── ui/                         # Web interface
│   └── app.py
│
└── utils/                      # Utilities
    └── text_cleaning.py
```

---

## 🚀 Quick Start (5 minutes)

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Add Data (Optional)
Place your Q&A data in `data/raw_kcc.csv`:
```csv
question,answer
"What is crop rotation?","Crop rotation is growing different crops seasonally..."
"How to prevent pests?","Integrated Pest Management (IPM) involves..."
```

### 3️⃣ Run Setup Pipeline
```bash
# Clean data and create embeddings
python services/preprocess_data.py
python services/generate_embeddings.py
python services/semantic_search.py
```
⏳ Takes 5-10 minutes on first run (downloads embedding model)

### 4️⃣ Configure LLM (Optional)
Edit `.env` and add IBM Watsonx credentials:
```env
WATSONX_API_KEY=your_key_here
WATSONX_PROJECT_ID=your_project_id
```

### 5️⃣ Launch App
```bash
streamlit run ui/app.py
```
✅ Opens at `http://localhost:8501`

---

## 📝 Sample Queries

Try these questions in the web interface:

| Query | Mode | Expected Result |
|-------|------|-----------------|
| "How to prevent crop disease?" | Offline | Retrieves disease prevention Q&As |
| "What fertilizer for wheat?" | Offline | Shows fertilizer recommendations |
| "Best time to plant rice?" | Both | Offline + AI-enhanced timing advice |
| "How do I improve soil fertility?" | Both | Knowledge base answer + personalized tips |
| "What is integrated pest management?" | Offline | Definition and techniques from database |

---

## 🔧 Detailed Setup Steps

### Step 1: Data Preparation
```bash
python services/preprocess_data.py
# Outputs: data/clean_kcc.csv, data/kcc_qa_pairs.json
```
- Removes duplicates and empty entries
- Cleans text formatting and special characters
- Creates JSON with metadata

### Step 2: Generate Embeddings
```bash
python services/generate_embeddings.py
# Outputs: embeddings/kcc_embeddings.pkl
```
- Converts Q&A text to semantic vectors
- Uses "all-MiniLM-L6-v2" model (~80MB)
- Creates 384-dimensional embeddings

### Step 3: Build Vector Index
```bash
python services/semantic_search.py
# Outputs: vector_store/faiss.index, vector_store/meta.pkl
```
- Creates FAISS IndexFlatL2
- Enables fast similarity search (<100ms)
- Includes metadata for result display

### Step 4: Run Web App
```bash
streamlit run ui/app.py
```
- Loads cached model and index (lazy loading)
- Shows offline results immediately
- Optionally calls LLM if enabled

---

## 💻 Using the App

**Interface Overview:**
1. **Query Box** - Enter your agricultural question
2. **Settings Sidebar** - Adjust search results, enable/disable LLM
3. **Offline Results** - Best matches from knowledge base
4. **Online Results** - AI-enhanced answer (if LLM enabled)
5. **Statistics** - Show result count and response time

**Settings:**
- **Number of results** (1-10) - How many references to retrieve
- **AI Mode** - Toggle LLM enhancement on/off
- **Temperature** (0.0-1.0) - Controls response creativity
- **Max tokens** (50-500) - Response length

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| "Embeddings file not found" | Run: `python services/preprocess_data.py && python services/generate_embeddings.py` |
| "FAISS index not found" | Run: `python services/semantic_search.py` |
| "LLM API key not set" | Add `WATSONX_API_KEY` to `.env`, or disable LLM in sidebar |
| "Streamlit won't start" | Run: `pip install streamlit==1.28.1` |
| "Slow embedding generation" | First run downloads model (~2GB) - normal. Cached locally after. |

---

## 📊 Performance

| Component | Response Time | Details |
|-----------|---------------|---------|
| Offline Search | <100ms | Pure vector similarity |
| Embeddings Gen | 1-5 min | One-time setup, depends on dataset size |
| LLM Response | 2-5 sec | Includes token fetch & generation |
| Full Query | <5 sec | Offline + optional LLM |

---

## 🎓 Architecture Highlights

**Separation of Concerns:**
- **Backend:** `SemanticSearch.get_answers()` + `GraniteLLMClient`
- **UI:** Streamlit displays formatted results

**Why This Approach?**
- FAISS: Fast, offline-first, no server needed
- Sentence-Transformers: Semantic understanding for agriculture
- RAG: Grounds LLM in real data, reduces hallucinations
- Lazy Loading: Fast startup, efficient resource use

---

## 📄 License & Disclaimer

This project is open-source for educational and agricultural purposes.

**⚠️ Disclaimer:**  
This assistant provides advisory information only. Farmers should consult local agricultural officers before application of any advice.

---

## 🤝 Contributing

Improvements welcome! Consider:
- Better text cleaning for agricultural terms
- Support for multiple languages
- Fine-tuning embedding model for agriculture domain
- Adding evaluation metrics
- Caching for faster repeated queries

---

**Happy Farming! 🌾**
