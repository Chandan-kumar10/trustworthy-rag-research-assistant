import re
import fitz  # PyMuPDF

HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(abstract|introduction|related work|background|methods?|methodology|"
    r"approach|experiments?|results?|discussion|conclusions?|limitations|references)\s*$",
    re.IGNORECASE,
)


def parse_pdf(path):
    """Returns a list of {"section", "page", "text"} dicts. Stops at References."""
    doc = fitz.open(path)
    sections = []
    state = {"name": "Front Matter", "page": 1, "buf": []}

    def flush():
        text = " ".join(state["buf"]).strip()
        if text:
            sections.append({
                "section": state["name"],
                "page": state["page"],
                "text": text,
            })
        state["buf"] = []

    for page_no, page in enumerate(doc, start=1):
        for line in page.get_text().split("\n"):
            line = line.strip()
            if not line:
                continue
            m = HEADING_RE.match(line)
            if m:
                name = m.group(1).title()
                flush()
                if name == "References":
                    doc.close()
                    return sections
                state["name"] = name
                state["page"] = page_no
                continue
            state["buf"].append(line)

    flush()
    doc.close()
    return sections


if __name__ == "__main__":
    import sys
    secs = parse_pdf(sys.argv[1])
    for s in secs:
        print(f"[{s['section']}] page {s['page']}: {s['text'][:150]}...\n")