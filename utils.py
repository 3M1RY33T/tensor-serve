import re
from libzim.reader import Archive

def iterate_articles(zim_path):
    archive = Archive(zim_path)

    for entry in archive.iter_entries():
        if entry.is_redirect:
            continue
        
        try:
            item = entry.get_item()
            content = item.content.tobytes().decode("utf-8", errors="ignore")

            yield {
                "title": entry.title,
                "text": content
            }

        except Exception:
            continue

def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)  # remove HTML
    text = re.sub(r"\s+", " ", text)      # normalize spaces
    return text.strip()