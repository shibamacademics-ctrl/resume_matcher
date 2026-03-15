from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import PyPDF2
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import os
import uvicorn

app = FastAPI(title="Resume Matcher API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load spaCy model
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')


# ─── Core Logic Functions ────────────────────────────────────────────────────

def extract_resume_text(file_bytes: bytes) -> str:
    text = ""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def clean_text(text: str) -> str:
    text_lower = text.lower()
    pattern = r"[^a-zA-Z0-9+/#.\- ]"
    match = re.sub(pattern, "", text_lower)
    return preprocess(match)


def preprocess(text: str) -> str:
    doc = nlp(text)
    filtered_text = []
    for token in doc:
        if token.is_punct:
            continue
        filtered_text.append(token.lemma_)
    return " ".join(filtered_text)


SKILLS_LIST = [
    "python", "java", "c++", "sql", "aws", "gcp", "azure",
    "docker", "kubernetes", "react", "node", "microservices",
    "machine", "learning", "data", "analysis", "system", "design",
    "distributed", "backend", "frontend", "cyber security",
    "cloud computing", "blockchain", "tensorflow", "pytorch",
    "mongodb", "postgresql", "redis", "kafka", "spark",
    "scala", "golang", "rust", "typescript", "graphql",
    "ML", "NLP", "Natural Language Processing"
]


def skill_overlap_score(resume: str, jd: str) -> float:
    resume_words = set(resume.split())
    jd_words = set(jd.split())
    jd_skills = [w for w in jd_words if w in SKILLS_LIST]
    if not jd_skills:
        return 0
    matched = [s for s in jd_skills if s in resume_words]
    return len(matched) / len(jd_skills)


def experience_score(resume: str, jd: str) -> float:
    resume_numbers = re.findall(r"\d+", resume)
    jd_numbers = re.findall(r"\d+", jd)
    if not jd_numbers:
        return 0.5
    resume_exp = max([int(n) for n in resume_numbers], default=0)
    jd_exp = max([int(n) for n in jd_numbers], default=0)
    if resume_exp >= jd_exp:
        return 1
    return resume_exp / jd_exp if jd_exp else 0


def calc_similarity(resume: str, jd: str) -> float:
    v = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
    vectors = v.fit_transform([resume, jd])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])
    return float(similarity[0][0])


def keyword_density_score(resume: str, jd: str) -> float:
    resume_words = resume.split()
    jd_words = jd.split()
    overlap = set(resume_words).intersection(set(jd_words))
    return len(overlap) / len(set(jd_words)) if jd_words else 0


def match_skills(text: str) -> set:
    found = []
    for skill in SKILLS_LIST:
        if skill in text:
            found.append(skill)
    return set(found)


def calculate_final_score(resume: str, jd: str) -> dict:
    cosine   = calc_similarity(resume, jd)
    skill    = skill_overlap_score(resume, jd)
    experience = experience_score(resume, jd)
    density  = keyword_density_score(resume, jd)

    final_score = (
        0.40 * cosine +
        0.25 * skill +
        0.20 * experience +
        0.15 * density
    ) * 100

    return {
        "final_score": round(final_score, 2),
        "breakdown": {
            "tfidf_cosine":      round(cosine * 100, 2),
            "skill_overlap":     round(skill * 100, 2),
            "experience_match":  round(experience * 100, 2),
            "keyword_density":   round(density * 100, 2),
        }
    }


# ─── API Routes ──────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("static/index.html", "r") as f:
        return f.read()


@app.post("/api/analyze")
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    # Validate file type
    if not resume.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Resume must be a PDF file.")

    if not job_description.strip():
        raise HTTPException(status_code=400, detail="Job description cannot be empty.")

    # Read and process resume
    resume_bytes = await resume.read()
    try:
        raw_resume = extract_resume_text(resume_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {str(e)}")

    if not raw_resume.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from PDF.")

    # Clean both texts
    resume_clean = clean_text(raw_resume)
    jd_clean     = clean_text(job_description)

    # Compute scores
    scores = calculate_final_score(resume_clean, jd_clean)

    # Skills analysis
    resume_skills = match_skills(resume_clean)
    jd_skills     = match_skills(jd_clean)
    matched       = resume_skills & jd_skills
    missing       = jd_skills - resume_skills

    return JSONResponse({
        "status": "success",
        "filename": resume.filename,
        "score": scores["final_score"],
        "breakdown": scores["breakdown"],
        "skills": {
            "matched": sorted(list(matched)),
            "missing": sorted(list(missing)),
            "resume_total": len(resume_skills),
            "jd_total": len(jd_skills),
        }
    })


@app.get("/api/health")
async def health():
    return {"status": "ok", "model": "en_core_web_sm"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
