from flask import Flask, render_template, request
import os
from datetime import datetime
from extractor import (
    HeadingExtractor,
    get_pdf_text,
    check_similarity,
    extract_refined_text,
    is_generic_heading,
    section_score,
    is_conversational_input,
)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        persona = request.form.get("persona", "").strip()
        job_to_be_done = request.form.get("job_to_be_done", "").strip()
        folder_name = request.form.get("folder_name", "").strip()

        # validate
        if not persona or not job_to_be_done or not folder_name:
            return render_template("index.html", error="⚠️ Please enter persona, job, and folder name.")

        combined_context = f"{persona} {job_to_be_done}".strip()
        # conversational detector (transformer + rules)
        if is_conversational_input(combined_context):
            return render_template(
                "result.html",
                result="⚠️ The input seems conversational or irrelevant. Please describe the persona and job clearly (e.g., 'Travel planner planning a 4-day trip for 10 friends').",
                output_text=None,
                pdf_scores=None
            )

        # folder checks
        folder_path = os.path.join("static", folder_name)
        if not os.path.exists(folder_path):
            return render_template("index.html", error=f"⚠️ Folder '{folder_name}' not found inside static/")

        pdf_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        if not pdf_paths:
            return render_template("index.html", error=f"⚠️ No PDF files found in '{folder_path}'")

        extractor = HeadingExtractor()
        all_sections = []
        pdf_similarity_scores = []

        # compute transformer similarity between each PDF and persona+job context
        for pdf_path in pdf_paths:
            pdf_text = get_pdf_text(pdf_path)
            score = check_similarity(pdf_text, combined_context)
            pdf_similarity_scores.append({"pdf": os.path.basename(pdf_path), "score": round(score, 2)})

        avg_similarity = sum(p["score"] for p in pdf_similarity_scores) / len(pdf_similarity_scores)

        # adaptive threshold
        context_word_count = len(combined_context.split())
        if context_word_count <= 5:
            BASE_THRESHOLD = 0.20
        elif context_word_count <= 15:
            BASE_THRESHOLD = 0.15
        else:
            BASE_THRESHOLD = 0.10
        dynamic_cutoff = max(BASE_THRESHOLD, avg_similarity - 0.05)

        relevant_pdfs = [p["pdf"] for p in pdf_similarity_scores if p["score"] >= dynamic_cutoff]
        # ensure at least 1-2 PDFs used; prefer multiple if available
        if len(relevant_pdfs) < 1:
            relevant_pdfs = [p["pdf"] for p in sorted(pdf_similarity_scores, key=lambda x: -x["score"])[:2]]

        # extract headings from all relevant PDFs
        for pdf_path in pdf_paths:
            pdf_name = os.path.basename(pdf_path)
            if pdf_name not in relevant_pdfs:
                continue
            res = extractor.extract_from_pdf(pdf_path)
            for section in res.get("outline", []):
                all_sections.append({
                    "document": pdf_name,
                    "section_title": section["text"],
                    "page_number": section["page"],
                    "y_pos": section.get("y_pos", 0),
                    "level": section.get("level", "H3")
                })

        # filter generic and score/sort
        filtered_sections = [s for s in all_sections if not is_generic_heading(s["section_title"])]
        if not filtered_sections:
            filtered_sections = all_sections[:10]  # fallback

        filtered_sections.sort(key=lambda s: (section_score(s), -s["page_number"]), reverse=True)

        # dedupe and pick top N
        seen = set()
        final_sections = []
        for s in filtered_sections:
            key = s["section_title"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            final_sections.append(s)
            if len(final_sections) >= 12:
                break

        # extract refined text per section
        final_notes = []
        for s in final_sections:
            refined = extract_refined_text(os.path.join(folder_path, s["document"]), s["page_number"], s.get("y_pos", 0))
            final_notes.append(f"📘 **{s['section_title']}** (from {s['document']}, page {s['page_number']})\n\n{refined}\n")

        readable_notes = "\n\n".join(final_notes)
        readable_notes += f"\n\n---\n🧩 Average Similarity: {avg_similarity:.2f}\n📅 Processed on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return render_template("result.html", result="✅ Notes Generated Successfully!", output_text=readable_notes, pdf_scores=pdf_similarity_scores)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
