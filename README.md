<p align="left">
  <a href="https://github.com/rahmay1/dynamic-resume-highlighter/actions">
    <img alt="CI" src="https://img.shields.io/github/actions/workflow/status/rahmay1/dynamic-resume-highlighter/ci.yml?label=CI&logo=github">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-blue.svg">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue">
  <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg">
</p>

# Dynamic Resume Keyword Highlighter

Highlight the exact skills a job description is asking for **directly in your LaTeX resume**.  
No static keyword list — the script builds a tech vocabulary on the fly from public sources, expands acronyms from context (e.g., `Artificial Intelligence (AI)`), and matches with hybrid **exact / fuzzy / semantic** logic.

> Works great for **mass‑applying**: point it at your LaTeX resume folder and a JD, and it writes a new folder with the skills bolded (and can compile a PDF if you want).

---

## ✨ Features

- **Dynamic skill DB (no hardcoding):** pulls languages / packages / frameworks from GitHub Linguist, Stack Overflow, PyPI, and curated lists.
- **Contextual acronym expansion:** detects patterns like `Artificial Intelligence (AI)` and `AI (Artificial Intelligence)` and expands *near actual acronym mentions*.  
  *Guards against nonsense like “ai is / and ip”.*
- **Smart matching pipeline:**
  - Exact + fuzzy (RapidFuzz) + optional semantic (Sentence‑BERT).
  - JS‑aware: treats `Node.js`, `Express.js`, `Next.js`, etc., as children of `JavaScript` and handles `.js` bases (`node` ↔ `node.js`).
  - Multi‑word phrase shadowing: won’t highlight `learning` when `Machine Learning` is the real skill.
- **LaTeX‑aware bolding:** skips URLs and already bolded regions; protects `\href{…}{…}`; prefers longer phrases first.
- **Section/header awareness:** dynamically detects typical headers and avoids treating them as skills. You can safely keep `Education`, `Projects`, etc., in your `.tex` files.
- **Performance minded:** caching for external lists; prunes tech terms to what actually appears in your JD+resume; optional `--quick`/`--no-semantic` fast paths.

---

## 📦 Requirements

- Python 3.8+
- `pip install -r requirements.txt` (includes `rapidfuzz`, `aiohttp`, `sentence-transformers` (optional), etc.)
- A LaTeX toolchain if you want PDF output (`pdflatex`, `xelatex`, `lualatex`, or `tectonic`).

> Semantic matching is optional. If you don’t want to download the SBERT model, run with `--no-semantic`.

---

## 🗂️ Project Layout

```text
.
├─ .skill_cache/                 # optional cache (safe to delete)
├─ .venv/                        # your virtualenv (not required)
├─ output_highlighted/           # generated LaTeX + optional PDF
├─ resume/
│  └─ src/                       # preferred (falls back to resume/ root)
│     ├─ experience.tex
│     ├─ projects.tex
│     ├─ skills.tex
│     └─ education.tex
├─ jd.txt                        # job description (plain text)
├─ highlight_resume_dynamic.py   # the script (v5/v6 names also fine)
├─ requirements.txt
└─ README.md
```

> The script looks for sections in `resume/src/` first; if that directory doesn’t exist it falls back to `resume/` itself.

---

## 🚀 Quick Start

```bash
# 1) Create and activate a venv (optional but recommended)
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Put your LaTeX resume files under ./resume/src/
#    (or directly under ./resume/ and the script will fall back to that)

# 4) Put a job description in jd.txt
#    (copy/paste from a posting, remove images/HTML)

# 5) Run
python highlight_resume_dynamic.py \
  --jd jd.txt \
  --root ./resume \
  --out output_highlighted \
  --try-pdf \
  --semantic-threshold 0.70 \
  --score-threshold 0.65
```

On success you’ll see a folder `output_highlighted/` that mirrors your resume’s `.tex` files, with matched skills wrapped in `\textbf{…}`. If `--try-pdf` is set and a LaTeX engine is available, a PDF is built too.

---

## 🙋 No `resume/` folder yet? (Starter Template)

You can scaffold a minimal layout in a minute. Copy the snippets below into files under `resume/src/`:

**`resume/src/experience.tex`**
```tex
\section*{Experience}
\resumeItem{Built REST APIs with Node.js/Express and PostgreSQL on GCP.}
\resumeItem{Led migration from monolith to microservices; wrote CI/CD workflows in GitHub Actions.}
```

**`resume/src/projects.tex`**
```tex
\section*{Projects}
\resumeItem{Real-time chat app with WebSocket + TypeScript + Docker.}
\resumeItem{ML pipeline (Python) for text classification; exposed a Flask API.}
```

**`resume/src/skills.tex`**
```tex
\section*{Skills}
JavaScript (Node.js, Express), TypeScript, Python, SQL, PostgreSQL, Docker, Git
```

**`resume/src/education.tex`**
```tex
\section*{Education}
Some University — relevant coursework: Algorithms, Databases, Distributed Systems
```

> Use your existing LaTeX macros if you have them; the script only looks for text between commands and handles common wrappers like `\resumeItem{…}`.

---

## 🧰 CLI Options (most used)

```
--jd PATH                     # JD text file
--root PATH                   # resume folder (containing src/ or .tex files)
--out PATH                    # output folder to write bolded files
--sections-dir NAME           # default: src
--try-pdf                     # try to compile PDF using any available engine
--engine-order simple|latexmk-first
--score-threshold FLOAT       # default 0.60
--semantic-threshold FLOAT    # default 0.70 (higher = stricter)
--no-semantic                 # disable SBERT for faster runs
--quick                       # use only what’s already cached
--no-cache                    # disable on-disk cache
--verbose                     # print full list of matched terms
```

---

## 🧠 Matching & Bolding Rules (short version)

- Terms are **pruned to those that appear** in JD+resume to minimize noise.
- **Acronym expansion** is conservative (windowed & TitleCase): `GCP` ⇄ `Google Cloud Platform`, `AI` ⇄ `Artificial Intelligence`, etc.
- **Header suppression**: lines that look like section headers are ignored as skills. The script further suppresses common education lines (e.g., GPA / degree titles) from bolding.
- **Phrase‑over‑token preference**: if a multi‑word phrase exists, its component words aren’t highlighted separately (e.g., prefer `Machine Learning` over `learning`).

---

## 🧪 Examples

```bash
# Faster run (no SBERT)
python highlight_resume_dynamic.py --jd jd.txt --root ./resume --out output_highlighted --no-semantic

# Force refresh external lists
python highlight_resume_dynamic.py --jd jd.txt --root ./resume --out output_highlighted --force
```

---

## 🛠 Troubleshooting

- **“No LaTeX engine found”**: install `texlive` (Linux), `MacTeX` (macOS), or MiKTeX/TeX Live (Windows). Or omit `--try-pdf` and compile later.
- **SBERT download too slow**: run with `--no-semantic` or keep the model cached between runs.
- **Too many matches**: raise `--score-threshold` and/or `--semantic-threshold`; or run with `--verbose` to see what’s being matched and why.

---

## Example (Try it in 30 seconds)

This repo includes a **minimal, fake** resume and job description so anyone can dry‑run the highlighter without your files.

**Project layout of the example:**

```text
examples/
├─ jd_sample.txt
└─ resume/
   ├─ resume.tex
   └─ src/
      ├─ experience.tex
      ├─ projects.tex
      ├─ skills.tex
      └─ education.tex
```

**Run the sample:**

```bash
# From the repo root
python highlight_resume_dynamic.py \
  --jd examples/jd_sample.txt \
  --root examples/resume \
  --out output_highlighted \
  --try-pdf
```

Notes:
- The LaTeX files mimic common resume structure and include a simple `\resumeItem{...}` macro so PDF compilation works.
- Education lines such as “Bachelor … | Minor in Computer Science | GPA …” are present to demonstrate suppression of generic terms (not highlighted).
- The sample JD asks for Node.js, Express, REST APIs, SQL, PostgreSQL, Angular, TypeScript, HTML/CSS, and Git.

**Download the example bundle:** [examples_bundle.zip](sandbox:/mnt/data/examples_bundle.zip)

If you don’t want to compile a PDF, omit `--try-pdf` and inspect the `.tex` in `output_highlighted/` to see bolded matches.

---

## 📈 Roadmap (nice‑to‑haves)

- Simple web front‑end to run in a browser (Pyodide/wasm).
- Optional export to DOCX.

---

## 📝 License

MIT — see `LICENSE`.

---

## 🤝 Contributing

Issues and PRs welcome. Please include a JD snippet and a minimal `.tex` snippet when reporting matching problems so we can reproduce quickly.
