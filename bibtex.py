import re

# --- CONFIG ---
input_bib = "biblatex_entries.bib"  # your current BibLaTeX .bib
output_bib = "icml_bibtex.bib"  # output BibTeX file

# Allowed fields for ICML/BibTeX
allowed_fields = [
    "author", "title", "journal", "booktitle", "year",
    "volume", "number", "pages", "publisher", "address", "note"
]

# Fields to move into note
move_to_note = ["doi", "eprint"]


# Function to protect capital letters in title
def protect_capitals(title):
    def repl(match):
        return "{" + match.group(0) + "}"

    # Protect all uppercase letters and acronyms
    # Skip letters already in {}
    return re.sub(r'(?<!\{)(\b[A-Z][A-Z0-9]*\b)(?!\})', repl, title)


with open(input_bib, "r", encoding="utf-8") as f:
    content = f.read()

# Split entries
entries = re.split(r'(@\w+\{)', content)
new_entries = []

for i in range(1, len(entries), 2):
    entry_type = entries[i].strip()
    entry_body = entries[i + 1].strip()
    # Reconstruct entry text
    full_entry = entry_type + entry_body
    # Extract key
    key_match = re.match(r'\w+\{([^,]+),', entry_body)
    if not key_match:
        continue
    key = key_match.group(1)

    # Extract fields
    fields = re.findall(r'(\w+)\s*=\s*\{(.*?)\}', entry_body, flags=re.DOTALL)
    field_dict = {}
    notes = []
    for field, value in fields:
        field = field.lower()
        if field in allowed_fields:
            if field == "title":
                value = protect_capitals(value)
            field_dict[field] = value
        elif field in move_to_note:
            notes.append(f"{field.upper()}: {value}")
        else:
            # ignore other fields
            pass
    if notes:
        note_text = "; ".join(notes)
        if "note" in field_dict:
            field_dict["note"] += "; " + note_text
        else:
            field_dict["note"] = note_text

    # Reconstruct entry
    entry_lines = [f"{entry_type}{key},"]
    for k, v in field_dict.items():
        entry_lines.append(f"  {k} = {{{v}}},")
    entry_lines[-1] = entry_lines[-1].rstrip(",")  # remove comma on last field
    entry_lines.append("}")
    new_entries.append("\n".join(entry_lines))

# Write output
with open(output_bib, "w", encoding="utf-8") as f:
    f.write("\n\n".join(new_entries))

print(f"Converted {len(new_entries)} entries to ICML-compatible BibTeX: {output_bib}")
