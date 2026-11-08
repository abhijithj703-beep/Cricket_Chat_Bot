import re
import json

INPUT_PATH = "../data/laws_of_cricket.txt"
OUTPUT_PATH = "../data/chunks.json"

# text to chunks -- test branch
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    text = f.read()

pattern = r"(?m)^(LAW \d+ [A-Z].*|(?:APPENDIX|Appendix) [A-Z](?!\.\d)\b.*)$"
parts = re.split(pattern, text)

chunks = []
seen_law_42 = False

preamble = parts[0].strip()
if preamble:
    chunks.append({
        "law_number": None,
        "appendix": None,
        "title": "Preamble - The Spirit of Cricket",
        "content": preamble
    })

for i in range(1, len(parts), 2):
    heading = parts[i].strip()
    content = parts[i + 1].strip() if i + 1 < len(parts) else ""

    law_match = re.match(r"LAW (\d+) (.*)", heading)
    appendix_match = re.match(r"(?:APPENDIX|Appendix) ([A-Z])[:\s]*(.*)", heading)

    if law_match:
        law_num = int(law_match.group(1))
        if law_num == 42:
            seen_law_42 = True
        chunks.append({
            "law_number": law_num,
            "appendix": None,
            "title": law_match.group(2),
            "content": content
        })
    elif appendix_match:
        if not seen_law_42:
            # False positive - this "Appendix" heading appeared before Law 42,
            # so it's actually part of the Contents list, not a real appendix.
            # Merge the heading text itself plus its content back into the
            # previous chunk instead of starting a new one.
            chunks[-1]["content"] += "\n" + heading + "\n" + content
            continue

        letter = appendix_match.group(1)
        title = appendix_match.group(2).strip()
        chunks.append({
            "law_number": None,
            "appendix": letter,
            "title": title if title else f"Appendix {letter}",
            "content": content
        })

print(f"Created {len(chunks)} chunks")

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(chunks, f, indent=2)

print(f"Saved to {OUTPUT_PATH}")