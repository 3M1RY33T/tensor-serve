import re

from libzim.reader import Archive


def iterate_articles(zim_path):
    """
    Iterate over all HTML articles in a ZIM file.

    Uses the _get_entry_by_id / all_entry_count API which is available in
    libzim >= 3.x.  Only text/html entries are yielded; CSS, JavaScript,
    images, JSON and internal metadata entries are skipped.
    """
    archive = Archive(zim_path)
    try:
        zim_title = archive.get_metadata("Title")
    except Exception:
        try:
            zim_title = archive.get_metadata("Name")
        except Exception:
            zim_title = None
    if isinstance(zim_title, bytes):
        zim_title = zim_title.decode("utf-8", errors="ignore")

    for i in range(archive.all_entry_count):
        try:
            entry = archive._get_entry_by_id(i)
        except Exception:
            continue

        if entry.is_redirect:
            continue

        try:
            item = entry.get_item()

            if not item.mimetype.startswith("text/html"):
                continue

            content = item.content.tobytes().decode("utf-8", errors="ignore")

            yield {
                "title": entry.title or entry.path,
                "text": content,
                "zim_title": zim_title,
                "zim_path": zim_path,
            }

        except Exception:
            continue


def clean_text(text):
    text = re.sub(r"<[^>]+>", " ", text)  # strip HTML tags
    text = re.sub(r"\s+", " ", text)  # normalise whitespace
    return text.strip()
