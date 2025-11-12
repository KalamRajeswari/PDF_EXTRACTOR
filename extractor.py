import fitz  # PyMuPDF
import re
import spacy
import os
from collections import defaultdict
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# Load large SpaCy model for structure & heading NLP
nlp = spacy.load("en_core_web_lg")

# Load small, fast semantic model for accurate similarity
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")


class HeadingExtractor:
    def __init__(self):
        self.min_font_occurrences = 3
        self.max_heading_words = 12

    def extract_from_pdf(self, pdf_path):
        """Main function: Extract structured headings from PDF."""
        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()
            if toc and len(toc) > 2:
                return self._extract_from_toc(doc, toc)
            return self._extract_with_nlp(doc)
        except Exception as e:
            print(f"Error extracting from PDF: {e}")
            return {"title": "Unknown", "outline": []}

    def _extract_from_toc(self, doc, toc):
        title = doc.metadata.get("title", "")
        outline = []
        for item in toc:
            level, heading_text, page = item
            if level <= 3:
                outline.append({
                    "level": f"H{level}",
                    "text": heading_text.strip(),
                    "page": page
                })
        if not title and outline:
            title = outline[0]["text"]
        return {"title": title, "outline": sorted(outline, key=lambda x: x["page"])}

    def _extract_with_nlp(self, doc):
        font_stats = self._analyze_fonts(doc)
        font_stats["spans"] = self._merge_close_spans(font_stats["spans"])
        candidates = self._extract_candidate_headings(doc, font_stats)
        headings = self._classify_headings_with_nlp(candidates)
        outline = self._assign_heading_levels(headings)
        outline = self._deduplicate_outline(outline)
        for item in outline:
            item["page"] += 1
        outline.sort(key=lambda x: x["page"])

        title = doc.metadata.get("title", "")
        if not title and outline:
            title = outline[0]["text"]
        return {"title": title, "outline": outline}

    def _analyze_fonts(self, doc):
        fonts = defaultdict(int)
        font_details = []
        for page_num, page in enumerate(doc):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        size = round(span.get("size", 0), 1)
                        is_bold = span.get("flags", 0) & 2 > 0
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        fonts[size] += 1
                        font_details.append({
                            "text": text,
                            "size": size,
                            "bold": is_bold,
                            "page": page_num,
                            "y_pos": span.get("bbox")[1],
                            "bbox": span.get("bbox")
                        })
        common_fonts = sorted(
            [(size, count) for size, count in fonts.items() if count >= self.min_font_occurrences],
            key=lambda x: x[0], reverse=True
        )
        return {"common_fonts": common_fonts, "spans": font_details}

    def _merge_close_spans(self, spans, y_threshold=1.5):
        merged = []
        current = None
        for span in sorted(spans, key=lambda x: (x['page'], x['y_pos'])):
            if current and abs(current['y_pos'] - span['y_pos']) < y_threshold and current['page'] == span['page']:
                current['text'] += ' ' + span['text']
            else:
                if current:
                    merged.append(current)
                current = span.copy()
        if current:
            merged.append(current)
        return merged

    def _extract_candidate_headings(self, doc, font_stats):
        candidates = []
        common_fonts = font_stats["common_fonts"]
        spans = font_stats["spans"]
        spans_by_page = defaultdict(list)
        for span in spans:
            spans_by_page[span["page"]].append(span)
        for span in spans:
            text = span["text"]
            size = span["size"]
            is_bold = span["bold"]
            page = span["page"]
            if len(text.split()) > self.max_heading_words:
                continue
            features = {
                "font_size": size,
                "is_bold": is_bold,
                "word_count": len(text.split()),
                "has_number_prefix": bool(re.match(r'^\d+(\.\d+)*\.?\s', text)),
                "is_all_caps": text.isupper(),
                "ends_with_colon": text.endswith(':'),
                "at_page_top": self._is_at_page_top(span, spans_by_page[page]),
                "standalone_line": self._is_standalone(span, spans_by_page[page])
            }
            score = self._score_candidate(features, common_fonts)
            if score > 0.5:
                candidates.append({
                    "text": text,
                    "page": page,
                    "features": features,
                    "score": score,
                    "size": size,
                    "is_bold": is_bold,
                    "y_pos": span["y_pos"],
                    "bbox": span["bbox"],
                })
        return candidates

    def _is_at_page_top(self, span, page_spans):
        if not page_spans:
            return False
        sorted_spans = sorted(page_spans, key=lambda x: x["y_pos"])
        return span["y_pos"] <= sorted_spans[0]["y_pos"] + 0.15 * (sorted_spans[-1]["y_pos"] - sorted_spans[0]["y_pos"])

    def _is_standalone(self, span, page_spans):
        bbox = span["bbox"]
        for other in page_spans:
            if other == span:
                continue
            y_overlap = max(0, min(bbox[3], other["bbox"][3]) - max(bbox[1], other["bbox"][1]))
            if y_overlap > 0:
                return False
        return True

    def _score_candidate(self, features, common_fonts):
        score = 0
        for i, (font_size, _) in enumerate(common_fonts[:3]):
            if abs(features["font_size"] - font_size) < 0.5:
                score += 0.3 - (i * 0.1)
                break
        if features["is_bold"]:
            score += 0.2
        if features["has_number_prefix"]:
            score += 0.3
        if features["is_all_caps"]:
            score += 0.15
        if features["ends_with_colon"]:
            score += 0.15
        if features["at_page_top"]:
            score += 0.25
        if features["standalone_line"]:
            score += 0.2
        if features["word_count"] > 8:
            score -= 0.1 * (features["word_count"] - 8)
        return min(1.0, max(0.0, score))

    def _classify_headings_with_nlp(self, candidates):
        if not candidates:
            return []
        texts = [c["text"] for c in candidates]
        docs = list(nlp.pipe(texts, disable=["ner"]))
        filtered = []
        for candidate, doc in zip(candidates, docs):
            text = candidate["text"]
            if not text[0].isupper():
                continue
            wc = len(text.split())
            if wc < 3 or wc > 12:
                continue
            has_verb = any(token.pos_ == "VERB" for token in doc)
            if has_verb and wc > 5:
                continue
            if '.' in text:
                continue
            score = candidate["score"]
            pos_pattern = " ".join([token.pos_ for token in doc])
            if re.search(r"^(DET )?(ADJ )*(NOUN|PROPN)", pos_pattern):
                score += 0.2
            if score >= 0.7:
                candidate["score"] = score
                filtered.append(candidate)
        return filtered

    def _assign_heading_levels(self, headings):
        if not headings:
            return []
        sorted_headings = sorted(headings, key=lambda x: x["score"], reverse=True)
        size_counts = defaultdict(int)
        for h in sorted_headings:
            size_counts[h["size"]] += 1
        common_sizes = sorted(size_counts.items(), key=lambda x: x[1], reverse=True)
        size_to_level = {size: f"H{i+1}" for i, (size, _) in enumerate(common_sizes[:3])}
        outline = []
        for h in sorted_headings:
            level = size_to_level.get(h["size"], "H3")
            if h["score"] < 0.75 and level == "H3":
                continue
            outline.append({
                "level": level,
                "text": h["text"],
                "page": h["page"],
                "y_pos": h["y_pos"]
            })
        return outline

    def _deduplicate_outline(self, outline):
        seen = set()
        result = []
        for item in outline:
            key = (item["text"].strip().lower(), item["page"])
            if key not in seen:
                seen.add(key)
                result.append(item)
        return result


# -------------------------
# Helper Functions
# -------------------------

def extract_refined_text(pdf_path, page_number, heading_y_pos, max_sentences=3):
    """Extract a few sentences after a heading."""
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    blocks = page.get_text("dict")["blocks"]

    candidate_blocks = [b for b in blocks if b.get("bbox") and b["bbox"][1] > heading_y_pos + 2]
    candidate_blocks = sorted(candidate_blocks, key=lambda b: b["bbox"][1])

    accumulated_text = ""
    sentence_count = 0
    for block in candidate_blocks:
        block_text = " ".join(span.get("text", "") for line in block.get("lines", []) for span in line.get("spans", []))
        block_text = block_text.strip()
        sentences = re.split(r'(?<=[.!?])\s+', block_text)
        for sent in sentences:
            if sentence_count < max_sentences:
                accumulated_text += sent + " "
                sentence_count += 1
            else:
                break
        if sentence_count >= max_sentences:
            break
    if sentence_count >= max_sentences:
        accumulated_text = accumulated_text.strip() + "..."
    return accumulated_text


def get_pdf_text(pdf_path, max_pages=3):
    """Extract text from first few pages for similarity check."""
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text("text") for i, page in enumerate(doc) if i < max_pages])
    return text.strip()


def check_similarity(text1, text2):
    """Compute semantic similarity between two texts."""
    if not text1.strip() or not text2.strip():
        return 0.0
    emb1 = semantic_model.encode(text1, convert_to_tensor=True)
    emb2 = semantic_model.encode(text2, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()
    return round(float(similarity), 3)


def is_generic_heading(title):
    generic_words = ["introduction", "summary", "conclusion", "overview", "preface", "contents", "references"]
    return any(word in title.lower() for word in generic_words)


def section_score(section):
    level_weight = {"H1": 3, "H2": 2, "H3": 1}
    return level_weight.get(section.get("level", "H3"), 1)
_CONVERSATIONAL_EXAMPLES = [
    "how are you",
    "how do you do",
    "what are you doing",
    "how's it going",
    "hi",
    "hello",
    "hey",
    "what's up",
    "how have you been",
    "how r u",
    "how are u"
]
_CONV_THRESHOLD = 0.70 
def is_conversational_input(text, min_tokens=2):
    """
    Return True if text appears conversational/irrelevant:
     - short and includes conversational words OR
     - semantic similarity to conversational examples above threshold
    """
    if not text or not text.strip():
        return True

    txt = text.strip().lower()
    # quick token-based guards
    tokens = [t.text for t in nlp(txt) if not t.is_punct and not t.is_space]
    if len(tokens) < min_tokens:
        return True

    # if it contains clear conversational phrases
    for ex in _CONVERSATIONAL_EXAMPLES:
        if ex in txt:
            return True

    # semantic similarity against examples
    emb_txt = semantic_model.encode(txt, convert_to_tensor=True)
    ex_embs = semantic_model.encode(_CONVERSATIONAL_EXAMPLES, convert_to_tensor=True)
    sims = util.cos_sim(emb_txt, ex_embs).cpu().tolist()[0]
    max_sim = max(sims) if sims else 0.0
    if max_sim >= _CONV_THRESHOLD:
        return True

    # heuristic: lots of first/second-person pronouns with short text -> conversational
    pronouns = {"you", "i", "we", "they", "he", "she", "your", "my"}
    pron_count = sum(1 for t in tokens if t.lower() in pronouns)
    if pron_count >= 1 and len(tokens) <= 6:
        return True

    return False
