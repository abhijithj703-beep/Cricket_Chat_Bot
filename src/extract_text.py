import pdfplumber
import re

PDF_PATH = "../data/laws_of_cricket.pdf"
OUTPUT_PATH = "../data/laws_of_cricket.txt"

with pdfplumber.open(PDF_PATH) as pdf:
    all_text = []
    for page in pdf.pages:
        text = page.extract_text()
        all_text.append(text)

full_text = "\n".join(all_text)

# Remove the repeating footer line (e.g. "Laws of Cricket 2017 Code (3rd Edition - 2022) 20")
full_text = re.sub(r"Laws of Cricket 2017 Code \(3rd Edition - 2022\) \d+", "", full_text)

# Remove the copyright line if it appears
full_text = re.sub(r"©\s*Marylebone Cricket Club", "", full_text)

# Collapse 3+ blank lines down to 2 (cosmetic cleanup)
full_text = re.sub(r"\n{3,}", "\n\n", full_text)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(full_text.strip())

print(f"Cleaned text saved to {OUTPUT_PATH}")
print(f"Total characters: {len(full_text)}")