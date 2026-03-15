# ResumeIQ — AI Resume Matcher

An AI-powered resume scoring tool built with **FastAPI**, **spaCy**, and **scikit-learn**.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the Server
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Open in Browser
```
http://localhost:8000
```

---

## 📁 Project Structure
```
resume-matcher/
├── main.py              # FastAPI application + all scoring logic
├── requirements.txt     # Python dependencies
├── static/
│   └── index.html       # Frontend UI
└── README.md
```

---

## 🔌 API Endpoints

### `POST /api/analyze`
Analyze a resume against a job description.

**Form Data:**
| Field             | Type   | Description              |
|-------------------|--------|--------------------------|
| `resume`          | File   | PDF resume file          |
| `job_description` | string | Plain text job description |

**Response:**
```json
{
  "status": "success",
  "filename": "resume.pdf",
  "score": 72.45,
  "breakdown": {
    "tfidf_cosine": 68.3,
    "skill_overlap": 80.0,
    "experience_match": 100.0,
    "keyword_density": 45.2
  },
  "skills": {
    "matched": ["python", "aws", "docker"],
    "missing": ["kubernetes", "react"],
    "resume_total": 8,
    "jd_total": 5
  }
}
```

### `GET /api/health`
Health check endpoint.

---

## ⚙️ Scoring Algorithm

The final score is a **weighted composite** of four metrics:

| Metric          | Weight | Description                              |
|-----------------|--------|------------------------------------------|
| TF-IDF Cosine   | 40%    | Semantic similarity via TF-IDF vectors   |
| Skill Overlap   | 25%    | Matching tech skills from a curated list |
| Experience Match| 20%    | Numeric experience year comparison        |
| Keyword Density | 15%    | Common vocabulary overlap percentage     |

### Score Interpretation
| Score | Rating      |
|-------|-------------|
| 75–100 | Excellent Match |
| 55–74  | Good Match      |
| 35–54  | Moderate Match  |
| 0–34   | Low Match       |

---

## 🛠 Tech Stack
- **FastAPI** — REST API framework
- **spaCy** — NLP lemmatization & text preprocessing  
- **scikit-learn** — TF-IDF vectorization & cosine similarity
- **PyPDF2** — PDF text extraction
- **Vanilla JS** — Interactive frontend (no framework needed)
