#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic Resume Keyword Highlighter - Enhanced Version
Automatically recognizes technical skills without hardcoded lists
Includes dynamic abbreviation expansion and improved semantic matching
"""

import os, re, json, argparse, sys, hashlib
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timedelta
import asyncio
import aiohttp
from typing import List, Dict, Set, Optional

# ---- Optional dependencies ----
HAVE_SBERT = True
try:
    from sentence_transformers import SentenceTransformer, util as sbert_util
except Exception:
    HAVE_SBERT = False
    print("Warning: sentence-transformers not found. Semantic matching disabled.", file=sys.stderr)

try:
    from rapidfuzz import fuzz, process
except Exception:
    print("ERROR: rapidfuzz is required. pip install rapidfuzz", file=sys.stderr)
    sys.exit(2)

# ----------------- Dynamic Abbreviation Expander -----------------
class AbbreviationExpander:
    """Dynamically expands abbreviations based on context"""

    def build_from_context(self, text: str) -> Dict[str, List[str]]:
        """Build abbreviation mappings from the text itself"""
        mappings = defaultdict(list)

        # Pattern 1: "Artificial Intelligence (AI)"
        pattern1 = re.findall(r'([A-Z][a-z]+(?: [A-Z][a-z]+)+)\s*\(([A-Z]+)\)', text)
        for full_form, abbrev in pattern1:
            mappings[abbrev.lower()].append(full_form.lower())
            mappings[full_form.lower()].append(abbrev.lower())

        # Pattern 2: "AI (Artificial Intelligence)"
        pattern2 = re.findall(r'([A-Z]+)\s*\(([A-Z][a-z]+(?: [A-Z][a-z]+)+)\)', text)
        for abbrev, full_form in pattern2:
            mappings[abbrev.lower()].append(full_form.lower())
            mappings[full_form.lower()].append(abbrev.lower())

        # Pattern 3: infer acronyms from Title Case multi-words (only 2–4 tokens)
        multi_word_terms = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', text)
        for term in multi_word_terms:
            words = term.split()
            if 2 <= len(words) <= 4:
                abbrev = ''.join(w[0] for w in words).upper()
                if 2 <= len(abbrev) <= 4:
                    mappings[term.lower()].append(abbrev.lower())
                    mappings[abbrev.lower()].append(term.lower())

        # Pattern 4: windowed, TitleCase, stopword-aware expansions around each acronym
        potential_abbrevs = set(re.findall(r'\b([A-Z]{2,4})\b', text))
        try:
            from spacy.lang.en.stop_words import STOP_WORDS as _SW  # optional
            _HAVE_SW = True
        except Exception:
            _SW = set()
            _HAVE_SW = False

        for abbr in potential_abbrevs:
            for m in re.finditer(r'\b' + re.escape(abbr) + r'\b', text):
                L = max(0, m.start() - 100)
                R = min(len(text), m.end() + 100)
                window = text[L:R]

                parts = [fr'{c}[a-z]+' for c in abbr]
                patt = re.compile(r'\b(' + r'\s+'.join(parts) + r')\b')

                for phrase in patt.findall(window):
                    tokens = phrase.split()
                    if len(tokens) != len(abbr):
                        continue
                    if not all(t[:1].isupper() and t[1:].islower() for t in tokens):
                        continue
                    if not all(t.isalpha() and len(t) >= 3 for t in tokens):
                        continue
                    if max(len(t) for t in tokens) < 4:
                        continue
                    if _HAVE_SW:
                        sw_ratio = sum(t.lower() in _SW for t in tokens) / len(tokens)
                        if sw_ratio > 0.34:
                            continue

                    k1 = abbr.lower()
                    k2 = phrase.lower()
                    if k2 not in mappings[k1]:
                        mappings[k1].append(k2)
                    if k1 not in mappings[k2]:
                        mappings[k2].append(k1)

        return dict(mappings)

    def expand_term(self, term: str, context_mappings: Dict[str, List[str]]) -> List[str]:
        """Get all possible expansions of a term (exact mappings only)"""
        term_lower = term.lower()
        out = {term}
        if term_lower in context_mappings:
            out.update(context_mappings[term_lower])
        return list(out)

# ----------------- Cache Manager -----------------
class SkillCache:
    """Manages local file-based caching for API responses"""
    def __init__(self, cache_dir: Optional[str] = ".skill_cache"):
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.ttl_hours = 168  # 1 week
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True)

    def _get_cache_path(self, key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def get(self, key: str, force_refresh: bool = False) -> Optional[Dict]:
        if force_refresh or not self.cache_dir:
            return None
        cache_path = self._get_cache_path(key)
        if not cache_path or not cache_path.exists():
            return None
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=self.ttl_hours):
                return None
            return cache_data['data']
        except Exception:
            return None

    def set(self, key: str, data: Dict, ttl_hours: Optional[int] = None):
        if not self.cache_dir:
            return
        cache_path = self._get_cache_path(key)
        if not cache_path:
            return
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'ttl_hours': ttl_hours or self.ttl_hours,
            'data': data
        }
        with open(cache_path, 'w') as f:
            json.dump(cache_data, f)

# ----------------- API Clients -----------------
class SkillAPIClient:
    """Aggregates multiple skill recognition APIs"""
    def __init__(self, cache: SkillCache):
        self.cache = cache
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_github_languages(self, force_refresh: bool = False) -> List[str]:
        cache_key = "github_languages"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        languages = []
        try:
            url = "https://raw.githubusercontent.com/github/linguist/master/lib/linguist/languages.yml"
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    for line in content.split('\n'):
                        if line.strip() and not line.startswith('#'):
                            m = re.match(r'^([A-Za-z0-9+#.\- ]+):', line)
                            if m:
                                lang = m.group(1).strip()
                                if len(lang) > 1:
                                    languages.append(lang)
                    alias_matches = re.findall(r'aliases:\s*\[(.*?)\]', content)
                    for alias_group in alias_matches:
                        aliases = [a.strip().strip('"\'') for a in alias_group.split(',')]
                        languages.extend(aliases)
            self.cache.set(cache_key, languages, ttl_hours=24)
        except Exception as e:
            print(f"Warning: Failed to fetch GitHub languages: {e}", file=sys.stderr)
        return languages

    async def fetch_stackoverflow_tags(self, min_count: int = 1000, force_refresh: bool = False) -> List[Dict]:
        cache_key = f"stackoverflow_tags_{min_count}"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        tags = []
        try:
            url = "https://api.stackexchange.com/2.3/tags"
            params = {'site': 'stackoverflow', 'pagesize': 100, 'sort': 'popular', 'order': 'desc'}
            if min_count > 0:
                params['min'] = min_count
            async with self.session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    tags = [{'name': item['name'], 'count': item['count']} for item in data.get('items', [])]
            self.cache.set(cache_key, tags, ttl_hours=24)
        except Exception as e:
            print(f"Warning: Failed to fetch Stack Overflow tags: {e}", file=sys.stderr)
        return tags

    async def fetch_pypi_packages(self, force_refresh: bool = False) -> List[str]:
        cache_key = "pypi_packages"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        packages = []
        try:
            url = "https://hugovk.github.io/top-pypi-packages/top-pypi-packages-30-days.min.json"
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    packages = [row['project'] for row in data.get('rows', [])[:500]]
            self.cache.set(cache_key, packages, ttl_hours=24)
        except Exception as e:
            print(f"Warning: Failed to fetch PyPI packages: {e}", file=sys.stderr)
        return packages

    async def fetch_npm_packages(self, force_refresh: bool = False) -> List[str]:
        cache_key = "npm_packages"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        packages = []
        try:
            url = "https://raw.githubusercontent.com/sindresorhus/awesome/main/readme.md"
            async with self.session.get(url, timeout=10) as response:
                if response.status == 200:
                    content = await response.text()
                    matches = re.findall(r'\[([a-z0-9\-\.]+)\]', content, re.I)
                    packages = [m for m in matches if 2 < len(m) < 30][:200]
            self.cache.set(cache_key, packages, ttl_hours=24)
        except Exception as e:
            print(f"Warning: Failed to fetch NPM packages: {e}", file=sys.stderr)
        return packages

    async def fetch_common_tech_terms(self, force_refresh: bool = False) -> List[str]:
        cache_key = "common_tech_terms"
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached:
                return cached
        terms = set()
        try:
            skill_db_url = "https://raw.githubusercontent.com/linkedin/skill-sample-data/main/skills-data.json"
            try:
                async with self.session.get(skill_db_url, timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        try:
                            data = json.loads(content)
                            def extract_strings(obj, result=None):
                                if result is None: result = set()
                                if isinstance(obj, dict):
                                    for v in obj.values(): extract_strings(v, result)
                                elif isinstance(obj, list):
                                    for item in obj: extract_strings(item, result)
                                elif isinstance(obj, str) and 1 < len(obj) < 50:
                                    result.add(obj)
                                return result
                            terms.update(extract_strings(data))
                        except json.JSONDecodeError:
                            pass
            except Exception:
                pass
            markdown_sources = [
                "https://raw.githubusercontent.com/kamranahmedse/developer-roadmap/master/readme.md",
                "https://raw.githubusercontent.com/sindresorhus/awesome/main/readme.md",
            ]
            for url in markdown_sources:
                try:
                    async with self.session.get(url, timeout=5) as response:
                        if response.status == 200:
                            content = await response.text()
                            heading_matches = re.findall(r'^[ \t]*#{1,6}[ \t]+(.+)$', content, flags=re.MULTILINE)
                            for heading in heading_matches:
                                clean = re.sub(r'[^\w\s\-+#./]', ' ', heading)
                                clean = re.sub(r'\s+', ' ', clean).strip()
                                if 2 < len(clean) < 80:
                                    terms.add(clean)
                            link_matches = re.findall(r'\[([^\]]+)\]', content)
                            for link_text in link_matches:
                                t = link_text.strip()
                                if 2 < len(t) < 50:
                                    terms.add(t)
                except Exception:
                    continue
            terms_list = sorted(terms)
            self.cache.set(cache_key, terms_list, ttl_hours=168)
        except Exception as e:
            print(f"Warning: Failed to fetch common tech terms: {e}", file=sys.stderr)
            terms_list = []
        return terms_list

    async def fetch_technology_terms(self, force_refresh: bool = False) -> Set[str]:
        all_terms = set()
        tasks = [
            self.fetch_github_languages(force_refresh),
            self.fetch_stackoverflow_tags(force_refresh),
            self.fetch_pypi_packages(force_refresh),
            self.fetch_npm_packages(force_refresh),
            self.fetch_common_tech_terms(force_refresh)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        if isinstance(results[0], list): all_terms.update(results[0])
        if isinstance(results[1], list): all_terms.update(tag['name'] for tag in results[1])
        if isinstance(results[2], list): all_terms.update(results[2])
        if isinstance(results[3], list): all_terms.update(results[3])
        if isinstance(results[4], list): all_terms.update(results[4])
        return all_terms

# ----------------- NLP Skill Extractor -----------------
class DynamicSkillExtractor:
    """Extracts skills using NLP without predefined lists"""

    def __init__(self):
        self.nlp = None   # spaCy intentionally not loaded (fast path)
        self.tech_terms = set()
        self.abbreviation_expander = AbbreviationExpander()

    def set_tech_terms(self, terms: Set[str]):
        self.tech_terms = {t.lower() for t in terms}
        for term in list(self.tech_terms):
            if '.' in term:
                self.tech_terms.add(term.replace('.', ''))
            if '-' in term:
                self.tech_terms.add(term.replace('-', ' '))
                self.tech_terms.add(term.replace('-', ''))
            if term.endswith('.js'):
                base = term[:-3]
                if len(base) >= 2:
                    self.tech_terms.add(base)

    def _is_adjectival_nontech(self, token: str, text: str, multi_word_phrases: Set[str]) -> bool:
        if not token.isalpha() or ' ' in token:
            return False
        if any(token in phrase.split() for phrase in multi_word_phrases):
            return False
        # regex fallback (fast): adjacent lowercase word ⇒ likely adjectival
        fwd = len(re.findall(r'\b' + re.escape(token) + r'\s+[a-z][a-z\-]{2,}\b', text))
        back = len(re.findall(r'\b[a-z][a-z\-]{2,}\s+' + re.escape(token) + r'\b', text))
        return (fwd + back) >= 1

    def extract_skills(self, text: str, expand_abbreviations: bool = True,
                       suppress_section_headers: bool = False,
                       section_headers: Set[str] = None) -> List[Dict]:
        skills = []

        context_mappings = {}
        if expand_abbreviations:
            context_mappings = self.abbreviation_expander.build_from_context(text)

        multi_word_phrases = set()
        component_words_in_phrases = set()

        for n in [2, 3, 4]:
            words = re.findall(r'\b[\w+#.\-/]+\b', text)
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n]).lower()
                if phrase in self.tech_terms:
                    multi_word_phrases.add(phrase)
                    for w in phrase.split():
                        component_words_in_phrases.add(w)

        # Only shadow when the TitleCase phrase itself is a known tech term
        titlecase_phrases = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b', text)
        for phrase in titlecase_phrases:
            lower = phrase.lower()
            if lower in self.tech_terms:  # gate on actual tech phrase
                multi_word_phrases.add(lower)
                for w in lower.split():
                    component_words_in_phrases.add(w)

        # NEW: Shadow-only phrases from degree/title lines so single words like
        # "science" and "engineering" are *not* highlighted alone.
        # Examples: "Master of Science in Computer Science", "B.S. in Computer Engineering"
        for line in text.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # Heuristic: degree lines tend to have "Master", "Bachelor", "B.S.", "M.S.", etc.,
            # and a trailing "in <Title Case phrase>"
            if re.search(r'\b(Master|Bachelor|M\.?S\.?|B\.?S\.?)\b', line_stripped):
                m = re.search(r'\bin\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\b', line_stripped)
                if m:
                    deg_phrase = m.group(1).lower()
                    # Use this phrase only for shadowing components; do not add it as a skill.
                    for w in deg_phrase.split():
                        component_words_in_phrases.add(w)

        # NEW: Coursework phrase harvesting.
        # Lines like "Relevant Coursework: Software Development, Data Structures & Algorithms, ..."
        for line in text.splitlines():
            if re.search(r'(?i)\bRelevant\s+Coursework\b', line):
                after_colon = line.split(':', 1)[1] if ':' in line else line
                for chunk in re.split(r'[;,/]', after_colon):
                    ph = re.sub(r'\s+', ' ', chunk.strip())
                    if len(ph.split()) >= 2:
                        lower = ph.lower()
                        # 1) shadow component words (so "development" alone won’t be highlighted)
                        multi_word_phrases.add(lower)
                        for w in lower.split():
                            component_words_in_phrases.add(w)
                        # 2) emit the full coursework phrase as a real skill candidate
                        skills.append({
                            'name': ph,
                            'score': 1.0,
                            'type': 'coursework',
                            'expansions': self.abbreviation_expander.expand_term(
                                ph, context_mappings
                            ) if expand_abbreviations else []
                        })

        text_lower = text.lower()

        words_raw = re.findall(r'\b[\w+#.\-/]+\b', text)
        words = []
        for w in words_raw:
            words.append(w)

            # Split "Python/Node" or "C/C++"
            if '/' in w:
                parts = [p for p in re.split(r'/+', w) if p]
                words.extend(parts)

            # NEW: split "*.js" frameworks into base + "js" (e.g., "Express.js" -> "Express", "js")
            m = re.match(r'^([A-Za-z][A-Za-z0-9\-]*)\.js$', w, re.I)
            if m:
                words.append(m.group(1))
                words.append('js')

            # NEW: split CamelCase “…JS/…Js/…js” (e.g., "ExpressJS" -> "Express", "js")
            m2 = re.match(r'^([A-Za-z][A-Za-z0-9\-]*)(?:JS|Js|js)$', w)
            if m2:
                words.append(m2.group(1))
                words.append('js')

        token_set = {w.lower() for w in words}

        # single tokens
        for word in words:
            wl = word.lower()
            if suppress_section_headers and section_headers:
                if wl in section_headers or word in section_headers:
                    continue
            if wl in component_words_in_phrases:
                continue
            if wl in self.tech_terms:
                if not self._is_adjectival_nontech(wl, text, multi_word_phrases):
                    if not any(s['name'].lower() == wl for s in skills):
                        skills.append({
                            'name': word, 'score': 1.0, 'type': 'tech_exact',
                            'expansions': self.abbreviation_expander.expand_term(word, context_mappings) if expand_abbreviations else []
                        })
                continue
            wn = wl.rstrip('s')
            if wn in self.tech_terms and not self._is_adjectival_nontech(wn, text, multi_word_phrases):
                if not any(s['name'].lower() == wl for s in skills):
                    skills.append({
                        'name': word, 'score': 0.9, 'type': 'tech_normalized',
                        'expansions': self.abbreviation_expander.expand_term(word, context_mappings) if expand_abbreviations else []
                    })

        # multi-word phrases
        for n in [2, 3, 4]:
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                pl = phrase.lower()
                if pl in self.tech_terms:
                    if not any(s['name'].lower() == pl for s in skills):
                        skills.append({
                            'name': phrase, 'score': 1.1, 'type': 'tech_phrase',
                            'expansions': self.abbreviation_expander.expand_term(phrase, context_mappings) if expand_abbreviations else []
                        })
                phrase_and = phrase.replace(' & ', ' and ')
                if phrase_and.lower() in self.tech_terms:
                    if not any(s['name'].lower() == pl for s in skills):
                        skills.append({
                            'name': phrase, 'score': 1.1, 'type': 'tech_phrase_and',
                            'expansions': self.abbreviation_expander.expand_term(phrase, context_mappings) if expand_abbreviations else []
                        })

        # fallback pattern search
        for tech_term in self.tech_terms:
            tl = tech_term.lower()
            if suppress_section_headers and section_headers and tl in section_headers:
                continue
            if ' ' not in tl and tl.isalpha() and tl not in token_set:
                continue
            if ' ' not in tl and tl.isalpha():
                if tl in component_words_in_phrases:
                    continue
                if self._is_adjectival_nontech(tl, text, multi_word_phrases):
                    continue
            if re.search(r'\b' + re.escape(tl) + r's?\b', text_lower, re.I):
                if not any(s['name'].lower() == tl for s in skills):
                    m = re.search(r'\b' + re.escape(tl) + r's?\b', text, re.I)
                    if m:
                        skills.append({
                            'name': m.group(), 'score': 0.8, 'type': 'tech_found',
                            'expansions': self.abbreviation_expander.expand_term(m.group(), context_mappings) if expand_abbreviations else []
                        })

        # dedupe keep highest score
        seen = {}
        for s in skills:
            key = s['name'].lower()
            if key not in seen or s['score'] > seen[key]['score']:
                seen[key] = s
        return list(seen.values())

# ----------------- LaTeX Processing -----------------
def find_brace_span(s: str, start: int):
    if start >= len(s) or s[start] != '{':
        return None
    i, depth = start, 0
    while i < len(s):
        if s[i] == '{': depth += 1
        elif s[i] == '}':
            depth -= 1
            if depth == 0: return i
        i += 1
    return None

def strip_latex_commands(s: str) -> str:
    s = re.sub(r"\\href{[^{}]*}{([^{}]*)}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+{([^{}]*)}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    return s

def detect_section_headers(tex_files: Dict[str, str]) -> Set[str]:
    headers = set()
    filename_based_headers = {
        'education.tex': ['education'],
        'experience.tex': ['experience', 'professional experience', 'work experience'],
        'projects.tex': ['projects'],
        'skills.tex': ['skills', 'technical skills']
    }
    for filename, content in tex_files.items():
        if filename.lower() in filename_based_headers:
            headers.update(filename_based_headers[filename.lower()])
        for match in re.findall(r'\\section\*?\{([^}]+)\}', content):
            clean = strip_latex_commands(match).strip()
            if clean:
                headers.add(clean.lower())
                words = clean.split()
                if len(words) <= 2:
                    for w in words:
                        if w.lower() in ['education','experience','projects','skills','professional','technical','leadership','work']:
                            headers.add(w.lower())
    return headers

def extract_resume_text(tex: str, suppress_headers: bool = True,
                        section_headers: Set[str] = None) -> str:
    items, i = [], 0
    while i < len(tex):
        if tex.startswith('\\resumeItem{', i):
            j = tex.find('{', i)
            if j != -1:
                k = find_brace_span(tex, j)
                if k is not None:
                    items.append(strip_latex_commands(tex[j+1:k]))
                    i = k + 1
                    continue
        i += 1
    cleaned = strip_latex_commands(tex)
    for line in cleaned.split('\n'):
        line = line.strip()
        if line and not line.startswith('%') and not line.startswith('\\'):
            if suppress_headers and section_headers:
                ll = line.lower()
                if ll in section_headers:
                    continue
                words = line.split()
                if len(words) <= 2 and any(w.lower() in section_headers for w in words):
                    continue
            items.append(line)
    text = "\n".join(items).replace('&', ' and ')
    if "Data Structures" in text and "Algorithms" in text:
        text += "\nData Structures\nAlgorithms"
    return text

def protected_ranges(tex: str):
    ranges, i = [], 0
    while i < len(tex):
        if tex.startswith('\\textbf{', i):
            brace = tex.find('{', i)
            if brace != -1:
                end = find_brace_span(tex, brace)
                if end is not None:
                    ranges.append((i, end+1)); i = end + 1; continue
        elif tex.startswith('\\url{', i) or tex.startswith('\\href{', i):
            brace = tex.find('{', i)
            if brace != -1:
                end = find_brace_span(tex, brace)
                if end is not None:
                    if tex.startswith('\\href{', i):
                        second = tex.find('{', end+1)
                        if second != -1:
                            end2 = find_brace_span(tex, second)
                            if end2 is not None:
                                ranges.append((i, end2+1)); i = end2 + 1; continue
                    else:
                        ranges.append((i, end+1)); i = end + 1; continue
        i += 1
    return ranges

def apply_bold(tex: str, keywords: List[str], expansion_map: Dict[str, List[str]] = None) -> str:
    prot = protected_ranges(tex)
    def overlaps(a, b):
        for (s, e) in prot:
            if a < e and b > s:
                return True
        return False

    expanded_keywords = set(keywords)
    if expansion_map:
        for kw in keywords:
            k = kw.lower()
            if k in expansion_map:
                for exp in expansion_map[k]:
                    expanded_keywords.add(exp)

    multi_word_terms = [kw for kw in expanded_keywords if ' ' in kw]
    component_words = set()
    for term in multi_word_terms:
        component_words.update(term.lower().split())

    filtered_keywords = []
    for kw in expanded_keywords:
        if ' ' in kw or kw.lower() not in component_words:
            filtered_keywords.append(kw)

    keywords_sorted = sorted(filtered_keywords, key=lambda x: (-len(x), x))

    ops = []
    for kw in keywords_sorted:
        kw_esc = re.escape(kw)
        patterns = []
        if ' ' not in kw:
            if kw.isupper() and len(kw) <= 4:
                patterns.append(re.compile(r'(?i)(\b' + kw_esc + r's?\b)'))
                patterns.append(re.compile(r'(?i)(?:^|\s|&\s*)(' + kw_esc + r's?)(?:\s*&|\s|$|[,}])'))
            else:
                patterns.append(re.compile(r'(?i)(?<![-/])(\b' + kw_esc + r's?\b)(?![-/])'))
                
                # NEW: if the resume spells it as "Express.js" or "ExpressJS", bold that too
                patterns.append(re.compile(r'(?i)(\b' + kw_esc + r'(?:\.?js)\b)'))
        else:
            parts = kw.split()
            parts_esc = [re.escape(p) for p in parts]

            # exact spacing
            patterns.append(re.compile(r'(?i)(\b' + r'\s+'.join(parts_esc) + r'\b)'))
            # with " & " variants
            patterns.append(re.compile(r'(?i)(\b' + r'\s*&\s*'.join(parts_esc) + r'\b)'))

            # If the last token is an ALL-CAPS acronym (API/SDK/CPU...), accept an optional trailing 's'
            last = parts[-1]
            if last.isupper() and 2 <= len(last) <= 6:
                parts_esc_plural = parts_esc[:-1] + [parts_esc[-1] + r's?']
                patterns.append(re.compile(r'(?i)(\b' + r'\s+'.join(parts_esc_plural) + r'\b)'))

        for pat in patterns:
            for m in pat.finditer(tex):
                a, b = m.start(1), m.end(1)
                if not overlaps(a, b) and '\\textbf' not in tex[max(0,a-10):min(len(tex),b+10)]:
                    ops.append((a, b, '\\textbf{' + tex[a:b] + '}'))

    ops.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    filtered_ops, last_end = [], -1
    for a, b, rep in ops:
        if a >= last_end:
            filtered_ops.append((a, b, rep)); last_end = b
    filtered_ops.sort(key=lambda x: -x[0])

    out = tex
    for a, b, rep in filtered_ops:
        out = out[:a] + rep + out[b:]
    return out

# ----------------- Matching Algorithm -----------------
class SkillMatcher:
    """Hybrid exact/fuzzy matching with scoring"""

    def __init__(self, exact_boost: float = 1.5, semantic_threshold: float = 0.7, enable_semantic: bool = True):
        self.exact_boost = exact_boost
        self.semantic_threshold = semantic_threshold
        self.model = None
        if enable_semantic and HAVE_SBERT:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                print("SBERT model loaded for semantic matching", file=sys.stderr)
            except Exception as e:
                print(f"Warning: SBERT model loading failed: {e}", file=sys.stderr)

    def match_skills(self, jd_text: str, resume_skills: List[str],
                     tech_terms: Set[str], resume_text: str = None,
                     jd_mappings=None, resume_mappings=None) -> List[Dict]:
        matches = []

        extractor = DynamicSkillExtractor()
        extractor.set_tech_terms(tech_terms)
        jd_skills = extractor.extract_skills(jd_text, expand_abbreviations=True)

        expander = AbbreviationExpander()
        jd_mappings = jd_mappings or expander.build_from_context(jd_text)
        resume_mappings = resume_mappings or expander.build_from_context(resume_text or ' '.join(resume_skills))
        all_mappings = {**jd_mappings, **resume_mappings}

        def _is_short_acronym(s: str) -> bool:
            return bool(re.fullmatch(r'[A-Za-z]{2,4}', str(s)))

        def _ambiguous_acronym_token(acr: str) -> bool:
            acr = acr.lower()
            jd_opts = set(map(str.lower, jd_mappings.get(acr, [])))
            res_opts = set(map(str.lower, resume_mappings.get(acr, [])))
            # ambiguous if both sides have expansions but none intersect
            return jd_opts and res_opts and not (jd_opts & res_opts)

        if not jd_skills:
            print("Warning: No skills found in job description", file=sys.stderr)
            return [
                {'skill': s, 'score': 0.5, 'type': 'fallback', 'jd_match': 'general'}
                for s in resume_skills
            ]

        jd_normalized = {}
        for s in jd_skills:
            jd_normalized[s['name'].lower()] = s
            for exp in s.get('expansions', []):
                if exp.lower() not in jd_normalized:
                    jd_normalized[exp.lower()] = {'name': exp, 'score': s['score'] * 0.9, 'type': 'expansion', 'original': s['name']}

        # Accept base form for JD ".js" skills (e.g., "Express.js" → "Express")
        for k, js_skill in list(jd_normalized.items()):
            if k.endswith('.js'):
                base = k[:-3]
                if base and base not in jd_normalized:
                    jd_normalized[base] = {
                        'name': base,
                        'score': js_skill['score'] * 0.95,
                        'type': 'js_base',
                        'original': js_skill.get('original', js_skill['name'])
                    }

        resume_normalized = {}
        for s in resume_skills:
            resume_normalized[s.lower()] = s

        if resume_text:
            rt = resume_text.lower()
            for term, exps in resume_mappings.items():
                forms = [term.lower(), *[e.lower() for e in exps]]
                present = [f for f in forms if f in rt]
                if not present:
                    continue
                canonical = max(present, key=lambda x: (len(x), x))
                for alias in forms:
                    resume_normalized[alias] = canonical

        for s in resume_skills:
            for exp in expander.expand_term(s, all_mappings):
                resume_normalized.setdefault(exp.lower(), s)

        if 'javascript' in jd_normalized:
            js_parent = jd_normalized['javascript']
            js_like = []
            for k in list(resume_normalized.keys()):
                if re.search(r'(?:\.js|\bjs)\b$', k) and k not in ('js',):
                    js_like.append(k)
            for child in js_like:
                cl = child.lower()
                if cl not in jd_normalized:
                    jd_normalized[cl] = {
                        'name': child, 'score': js_parent['score'] * 0.95,
                        'type': 'js_child', 'original': js_parent.get('original', js_parent['name'])
                    }

        for alias in list(resume_normalized.keys()):
            if alias.endswith('.js'):
                base = alias[:-3]
                if base and base not in resume_normalized:
                    resume_normalized[base] = resume_normalized[alias]

        # --- SQL family bridging: if JD mentions "SQL", allow vendor/dialect tokens to count ---
        # (Place AFTER resume_normalized is built.)
        if 'sql' in jd_normalized:
            sql_parent = jd_normalized['sql']
            for k in list(resume_normalized.keys()):
                kl = k.lower()
                # allow e.g., postgresql, mysql, pl/sql, tsql, mssql; exclude "nosql"
                if 'nosql' in kl:
                    continue
                if re.search(r'(?i)(?:^|[\s\-/+_])[a-z0-9\-\+]*sql\b', kl) or re.search(r'(?i)[a-z0-9\-\+]*sql\b', kl):
                    if kl not in jd_normalized:
                        jd_normalized[kl] = {
                            'name': resume_normalized[k],
                            'score': sql_parent['score'] * 0.95,
                            'type': 'sql_child',
                            'original': sql_parent.get('original', sql_parent['name'])
                        }

        equivalence_classes = defaultdict(set)
        for term, exps in all_mappings.items():
            class_terms = {term} | set(exps)
            canonical = max(class_terms, key=lambda x: (len(x), x))
            for t in class_terms:
                equivalence_classes[t.lower()].add(canonical)

        def _promote_longer_titlecase(candidate: str, resume_text: str) -> str:
            parts = candidate.split()
            if not parts or not all(p[:1].isalpha() for p in parts):
                return candidate
            if not all(p[:1].isupper() for p in parts):
                return candidate
            patt = re.compile(r'\b' + re.escape(candidate) + r'(?:\s+[A-Z][a-z]+){1,2}\b')
            best = None
            for m in patt.finditer(resume_text):
                ph = m.group(0)
                if best is None or len(ph) > len(best):
                    best = ph
            return best or candidate

        def get_best_label(matched_term: str, jd_term: str, resume_term: str) -> str:
            candidates = {matched_term, jd_term, resume_term}
            for t in [matched_term, jd_term, resume_term]:
                candidates.update(equivalence_classes.get(t.lower(), set()))
            resume_candidates = [c for c in candidates if c.lower() in resume_normalized]

            if resume_candidates:
                specific, generic = [], []
                for cand in resume_candidates:
                    cl = cand.lower()
                    if (' ' in cand and ('cloud' in cl or 'azure' in cl)):
                        words = cand.split()
                        if any(w and w[0].isupper() for w in words):
                            specific.append(cand)
                        else:
                            generic.append(cand)
                    else:
                        generic.append(cand)
                if specific:
                    label = max(specific, key=lambda x: (len(x), x))
                elif generic:
                    label = max(generic, key=lambda x: (len(x), x))
                else:
                    label = matched_term
                return _promote_longer_titlecase(label, resume_text or '')
            else:
                return matched_term

        # exact
        for jd_key, jd_skill in jd_normalized.items():
            if jd_key in resume_normalized:
                original = resume_normalized[jd_key]
                best_label = get_best_label(original, jd_skill['name'], original)
                if not any(m['skill'].lower() == best_label.lower() for m in matches):
                    if not (_is_short_acronym(jd_skill['name']) and _ambiguous_acronym_token(jd_skill['name'])):
                        matches.append({'skill': best_label, 'score': jd_skill['score'] * self.exact_boost,
                                        'type': 'exact', 'jd_match': jd_skill.get('original', jd_skill['name'])})

        # fuzzy on remaining
        unmatched_jd = [s for k, s in jd_normalized.items()
                        if not any(m['jd_match'].lower() == s.get('original', s['name']).lower() for m in matches)]
        for jd_skill in unmatched_jd:
            for match_key, score, _ in process.extract(jd_skill['name'], list(resume_normalized.keys()),
                                                       scorer=fuzz.token_set_ratio, limit=3):
                if score >= 70:
                    original = resume_normalized[match_key]
                    best_label = get_best_label(original, jd_skill['name'], original)
                    if not any(m['skill'].lower() == best_label.lower() for m in matches):
                        if not (_is_short_acronym(jd_skill['name']) and _ambiguous_acronym_token(jd_skill['name'])):
                            matches.append({'skill': best_label, 'score': (score / 100) * jd_skill['score'],
                                            'type': 'fuzzy', 'jd_match': jd_skill.get('original', jd_skill['name'])})

        # semantic (opt-in / default on)
        if self.model and jd_skills and resume_skills:
            try:
                original_jd = [s for s in jd_skills if s.get('type') != 'expansion']
                jd_texts = [s['name'] for s in original_jd]
                resume_texts = list(set(resume_skills))
                jd_emb = self.model.encode(jd_texts, convert_to_tensor=True)
                res_emb = self.model.encode(resume_texts, convert_to_tensor=True)
                sims = sbert_util.cos_sim(jd_emb, res_emb)
                for i, jd_skill in enumerate(original_jd):
                    for j, r in enumerate(resume_texts):
                        sim = float(sims[i][j])
                        if sim >= self.semantic_threshold:
                            len_ratio = len(r) / len(jd_skill['name'])
                            if 0.3 < len_ratio < 3.0:
                                best_label = get_best_label(r, jd_skill['name'], r)
                                if not any(m['skill'].lower() == best_label.lower() for m in matches):
                                    matches.append({'skill': best_label, 'score': sim * jd_skill['score'] * 0.8,
                                                    'type': 'semantic', 'jd_match': jd_skill['name']})
            except Exception as e:
                print(f"Warning: Semantic matching failed: {e}", file=sys.stderr)

        # dedupe highest score
        best = {}
        for m in matches:
            k = m['skill'].lower()
            if k not in best or m['score'] > best[k]['score']:
                best[k] = m
        return sorted(best.values(), key=lambda x: x['score'], reverse=True)

# ----------------- PDF Building -----------------
def _which(prog):
    from shutil import which
    return which(prog) is not None

def build_pdf(out_dir, engine_order="simple", quiet=True):
    import subprocess
    DEVNULL = subprocess.DEVNULL
    run_kw = {} if not quiet else {"stdout": DEVNULL, "stderr": DEVNULL}
    print("\n== Building PDF (trying available engines) ==")
    cwd = os.getcwd()
    try:
        os.chdir(out_dir)
        if engine_order=="latexmk-first" and _which("latexmk"):
            try:
                subprocess.run(["latexmk","-pdf","-interaction=nonstopmode","resume.tex"], check=True, **run_kw)
                print("Built with latexmk."); return True
            except Exception as e:
                print(f"latexmk failed: {e}")
        if _which("tectonic"):
            try:
                subprocess.run(["tectonic","-X","compile","resume.tex"], check=True, **run_kw)
                print("Built with tectonic."); return True
            except Exception as e:
                print(f"tectonic failed: {e}")
        for eng in ["pdflatex","lualatex","xelatex"]:
            if _which(eng):
                try:
                    subprocess.run([eng,"-interaction=nonstopmode","resume.tex"], check=True, **run_kw)
                    subprocess.run([eng,"-interaction=nonstopmode","resume.tex"], check=True, **run_kw)
                    print(f"Built with {eng}."); return True
                except Exception as e:
                    print(f"{eng} failed: {e}")
        if engine_order!="latexmk-first" and _which("latexmk"):
            try:
                subprocess.run(["latexmk","-pdf","-interaction=nonstopmode","resume.tex"], check=True, **run_kw)
                print("Built with latexmk."); return True
            except Exception as e:
                print(f"latexmk failed: {e}")
        print("No LaTeX engine found or all failed."); return False
    finally:
        os.chdir(cwd)

# ----------------- Main Application -----------------
async def main():
    parser = argparse.ArgumentParser(
        description='Dynamic Resume Keyword Highlighter - Enhanced with abbreviation expansion'
    )
    parser.add_argument('--jd', required=True, help='Job description file')
    parser.add_argument('--root', required=True, help='LaTeX resume directory')
    parser.add_argument('--sections-dir', default='src', help='Sections subdirectory')
    parser.add_argument('--out', required=True, help='Output directory')
    parser.add_argument('--force', action='store_true', help='Force refresh cached data')
    parser.add_argument('--no-cache', action='store_true', help='Disable caching')
    parser.add_argument('--quick', action='store_true', help='Quick mode - use cached data only')
    parser.add_argument('--score-threshold', type=float, default=0.6, help='Minimum score threshold (0-1)')
    parser.add_argument('--semantic-threshold', type=float, default=0.7, help='Minimum semantic similarity threshold (0-1)')
    parser.add_argument('--max-highlights', type=int, default=50, help='Maximum keywords to highlight')
    parser.add_argument('--try-pdf', action='store_true', help='Compile PDF')
    parser.add_argument('--engine-order', choices=["simple","latexmk-first"], default="simple", help='LaTeX engine priority order')
    parser.add_argument('--verbose-pdf', action='store_true', help='Show full LaTeX build logs')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--no-semantic', action='store_true', help='Disable SBERT semantic matching')

    args = parser.parse_args()

    # Cache init (safe when --no-cache)
    cache = SkillCache(cache_dir=None) if args.no_cache else SkillCache()

    # JD
    jd_text = Path(args.jd).read_text(encoding='utf-8', errors='ignore')

    # Tech terms
    print("🔍 Fetching technology database from APIs...", file=sys.stderr)
    tech_terms = None
    if args.quick and not args.force:
        c1 = cache.get("github_languages"); c2 = cache.get("stackoverflow_tags_1000")
        c3 = cache.get("pypi_packages"); c4 = cache.get("npm_packages"); c5 = cache.get("common_tech_terms")
        if all([c1, c2, c3, c4, c5]):
            tech_terms = set(c1) | set(c3) | set(c4) | set(c5) | {t['name'] for t in c2}
    if tech_terms is None:
        async with SkillAPIClient(cache) as client:
            tech_terms = await client.fetch_technology_terms(force_refresh=args.force)
    print(f"✅ Loaded {len(tech_terms)} technology terms", file=sys.stderr)

    # Resume text
    print("📄 Processing resume LaTeX files...", file=sys.stderr)
    resume_text, tex_files = "", {}
    sections_path = Path(args.root) / args.sections_dir
    if not sections_path.exists():
        sections_path = Path(args.root)
    for filename in ['experience.tex', 'projects.tex', 'skills.tex', 'education.tex']:
        fp = sections_path / filename
        if fp.exists():
            tex_files[filename] = fp.read_text(encoding='utf-8', errors='ignore')
        else:
            print(f"  Warning: {filename} not found", file=sys.stderr)

    section_headers = detect_section_headers(tex_files)
    for _, content in tex_files.items():
        resume_text += extract_resume_text(content, suppress_headers=True, section_headers=section_headers) + "\n"
    if not resume_text.strip():
        print("ERROR: No resume content extracted. Check your LaTeX files.", file=sys.stderr)
        sys.exit(1)

    combo = (jd_text + "\n" + resume_text).lower()
    def seen(term: str) -> bool:
        t = term.lower()
        return (t in combo) or ('.' in t and t.replace('.', '') in combo) or ('-' in t and t.replace('-', ' ') in combo)
    tech_terms = {t for t in tech_terms if seen(t)}
    print(f"⚡ Pruned terms to {len(tech_terms)} that actually appear in JD/resume", file=sys.stderr)

    # 🔧 Seed extra candidates directly from the JD so prunes can't drop them.
    # We only keep seeds that also occur somewhere in the resume to avoid noise.
    jd_seeds = set()

    # *.js frameworks (e.g., Express.js, Next.js)
    jd_seeds |= {
        m.group(0).lower()
        for m in re.finditer(r'\b[A-Za-z][A-Za-z0-9\-]*\.js\b', jd_text)
    }

    # Single TitleCase tokens (e.g., Angular, React, Django)
    jd_seeds |= {
        m.group(0).lower()
        for m in re.finditer(r'\b[A-Z][a-z]{2,}\b', jd_text)
    }

    # Keep only seeds that actually appear in the resume text
    jd_seeds = {
        s for s in jd_seeds
        if re.search(r'\b' + re.escape(s) + r'\b', resume_text, re.I)
    }

    # Add to tech_terms so both JD and resume extractors can see them
    tech_terms |= jd_seeds

    # Abbreviation maps
    expander = AbbreviationExpander()
    expansion_map_jd = expander.build_from_context(jd_text)
    expansion_map_resume = expander.build_from_context(resume_text)

    # Extract resume skills
    print("🧠 Extracting skills from resume...", file=sys.stderr)
    extractor = DynamicSkillExtractor()
    extractor.set_tech_terms(tech_terms)
    resume_skills_data = extractor.extract_skills(resume_text, expand_abbreviations=True,
                                                  suppress_section_headers=True, section_headers=section_headers)
    resume_skills = [s['name'] for s in resume_skills_data]
    if not resume_skills:
        print("Warning: No skills extracted from resume", file=sys.stderr)
        for term in tech_terms:
            if term.lower() in resume_text.lower():
                resume_skills.append(term)

    if args.verbose:
        print(f"\n📋 Found {len(resume_skills)} skills in resume:", file=sys.stderr)
        for skill in sorted(resume_skills)[:20]:
            print(f"  • {skill}", file=sys.stderr)

    # Match
    print("🎯 Matching skills with job description...", file=sys.stderr)
    matcher = SkillMatcher(semantic_threshold=args.semantic_threshold, enable_semantic=not args.no_semantic)
    matches = matcher.match_skills(jd_text, resume_skills, tech_terms,
                                   resume_text=resume_text, jd_mappings=expansion_map_jd, resume_mappings=expansion_map_resume)

    # Build a dynamic unigram tech vocabulary from current tech_terms
    tech_unigrams = {
        tok
        for term in tech_terms
        for tok in re.findall(r'[a-z]+', str(term).lower())
        if len(tok) > 1
    }

    def _is_headerish_phrase(s: str) -> bool:
        """Short, generic phrase made only of non-tech unigrams -> header-ish (drop)."""
        if not s:
            return False
        s_lower = s.lower()
        # Keep if the full phrase itself is a known tech term
        if s_lower in {t.lower() for t in tech_terms}:
            return False
        toks = re.findall(r'[a-z]+', s_lower)
        # Short (≤3 tokens), alphabetic, and none of its tokens are present in any tech term
        return 1 <= len(toks) <= 3 and all(t not in tech_unigrams for t in toks)

    def _harvest_header_like_vocab(txt: str) -> set:
        """
        Dynamically collect 'header-like' tokens from short TitleCase lines.
        Catches lines such as 'Leadership Experience', 'Work Experience', etc.
        """
        vocab = set()
        for line in txt.splitlines():
            line = line.strip()
            if 0 < len(line) <= 60 and re.match(r'^[A-Z][A-Za-z/&\s]+$', line):
                toks = re.findall(r'[A-Za-z]+', line)
                if 1 <= len(toks) <= 4:
                    cap_ratio = sum(t[0].isupper() for t in toks) / len(toks)
                    if cap_ratio >= 0.5:
                        vocab.update(t.lower() for t in toks)
        return vocab

    # 1) Header-token vocabulary from the resume sections
    header_vocab = set()
    for h in section_headers:
        header_vocab.update(re.findall(r'[A-Za-z]+', h.lower()))

    # 2) Plus dynamic tokens from JD + resume text
    header_vocab |= _harvest_header_like_vocab(resume_text)
    header_vocab |= _harvest_header_like_vocab(jd_text)

    def _header_only_phrase(s: str) -> bool:
        toks = re.findall(r'[A-Za-z]+', s.lower())
        return toks and all(t in header_vocab for t in toks)

    # FINAL FILTER:
    # Drop if either the displayed label OR its source JD term is header-only OR header-ish.
    matches = [
        m for m in matches
        if not _header_only_phrase(m['skill'])
        and not _header_only_phrase(m.get('jd_match', ''))
        and not _is_headerish_phrase(m['skill'])
        and not _is_headerish_phrase(m.get('jd_match', ''))
    ]
    
    # Filter/limit
    filtered = [m for m in matches if m['score'] >= args.score_threshold]
    final_keywords = [m['skill'] for m in filtered[:args.max_highlights]]

    # Expand for bolding (resume-only), with short-initialism JD guard
    expanded_final = set(final_keywords)
    jd_lower = jd_text.lower()
    for kw in final_keywords:
        k = kw.lower()
        if k in expansion_map_resume:
            for exp in expansion_map_resume[k]:
                exp_l = exp.lower()
                if re.fullmatch(r'[a-z]{2,4}', exp_l):
                    if not re.search(r'\b' + re.escape(exp_l) + r'\b', jd_lower):
                        continue
                # NEW: only accept the acronym if JD's expansion set intersects resume's expansion set
                jd_opts = set(map(str.lower, expansion_map_jd.get(exp_l, [])))
                resume_opts = set(map(str.lower, expansion_map_resume.get(exp_l, [])))
                if jd_opts and resume_opts and not (jd_opts & resume_opts):
                    continue  # ambiguous meaning (e.g., JD "CS"=Computer Science, resume "CS"=Cloud Storage)
                expanded_final.add(exp)
                
    # robust singularization (APIs -> API, SDKs -> SDK), case-insensitive but only for acronyms
    # Only singularize ALL-CAPS acronyms that end with a **lowercase** 's' (APIs, SDKs, GPUs),
    # so we don't turn 'CSS' into 'CS'.
    for kw in list(expanded_final):
        if re.fullmatch(r'[A-Z]{2,6}s', kw):  # e.g., APIs, SDKs, GPUs
            expanded_final.add(kw[:-1])       # add API, SDK, GPU
            
    final_keywords = list(expanded_final)

    if not final_keywords:
        print("Warning: No keywords matched above threshold. Using top skills.", file=sys.stderr)
        final_keywords = [m['skill'] for m in matches[:min(10, len(matches))]]

    # Synthesize "X API" phrases that actually appear in the resume
    api_phrases = set()
    for m in re.finditer(r'\b([A-Z][A-Za-z0-9+./-]{2,})\s+API(?:s)?\b', resume_text):
        x = m.group(1)           # e.g., REST, Web, GraphQL, gRPC, etc.
        phrase = f"{x} API"
        # Only add if either X or API/APIs is already a selected keyword
        if any(k.lower() == x.lower() for k in final_keywords) or any(k.upper().startswith('API') for k in final_keywords):
            api_phrases.add(phrase)

    final_keywords = list(set(final_keywords) | api_phrases)
    
    # --- Ensure plain "SQL" appears if JD asks for it and resume only shows dialects ---
    if re.search(r'\bsql\b', jd_lower) and not any(k.lower() == 'sql' for k in final_keywords):
        # Evidence: final keywords or resume text contain a dialect ending with "sql", but not "nosql"
        dialect_hit = any(re.search(r'(?i)\b(?!no)[a-z0-9\-\+]*sql\b', k) for k in final_keywords) \
                      or re.search(r'(?i)\b(?!no)[a-z0-9\-\+]*sql\b', resume_text)
        if dialect_hit:
            final_keywords.append('SQL')

    print(f"\n✨ Highlighting {len(final_keywords)} matched skills:", file=sys.stderr)
    display_limit = len(final_keywords) if args.verbose else 15
    for i, kw in enumerate(sorted(final_keywords)[:display_limit], 1):
        item = next((m for m in filtered if m['skill'] == kw), None)
        if item:
            print(f"  {i:2}. {kw} (score: {item['score']:.2f}, type: {item['type']})", file=sys.stderr)
        else:
            # expansion from resume
            src = None
            for abbr, exps in expansion_map_resume.items():
                if kw.lower() in [e.lower() for e in exps]:
                    src = abbr; break
            if src:
                print(f"  {i:2}. {kw} (expanded from {src.upper()} in resume)", file=sys.stderr)
            else:
                print(f"  {i:2}. {kw} (derived match)", file=sys.stderr)

    if args.verbose and expansion_map_resume:
        print("\n📚 Abbreviation expansions detected in resume:", file=sys.stderr)
        for key, values in expansion_map_resume.items():
            if any(key == kw.lower() for kw in final_keywords):
                print(f"  {key} → {', '.join(values)}", file=sys.stderr)

    # Write LaTeX with bolding
    print("\n📝 Applying bold formatting to LaTeX files...", file=sys.stderr)
    os.makedirs(args.out, exist_ok=True)
    for root, _, files in os.walk(args.root):
        for file in files:
            if file.endswith('.tex'):
                src_path = Path(root) / file
                rel_path = src_path.relative_to(args.root)
                dst_path = Path(args.out) / rel_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                content = src_path.read_text(encoding='utf-8', errors='ignore')
                if file in ['experience.tex','projects.tex','skills.tex','education.tex']:
                    content = apply_bold(content, final_keywords, expansion_map_resume)
                dst_path.write_text(content, encoding='utf-8')

    print(f"✅ Output written to {args.out}", file=sys.stderr)

    if args.try_pdf:
        build_pdf(args.out, args.engine_order, quiet=(not args.verbose_pdf))

if __name__ == "__main__":
    if sys.version_info < (3, 7):
        print("Python 3.7+ required", file=sys.stderr)
        sys.exit(1)
    asyncio.run(main())
