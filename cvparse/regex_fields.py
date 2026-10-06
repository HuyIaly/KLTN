"""Fixed-format fields by regex, kept out of the NER tag set (Ngo, Two-Phase CV).

    find_fields("Email: a@b.vn | 0903 414 848") -> [("EMAIL", 7, 13, "a@b.vn"), ("PHONE", 16, 28, ...)]
"""
from __future__ import annotations

import re

PATTERNS = {
    "EMAIL": r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
    "URL": r"(?:https?://|www\.)\S+|\b(?:linkedin\.com|github\.com|behance\.net)/\S+",
    # VN mobile/landline: +84 / 84 / 0 then 8-10 digits, with optional . - space separators
    "PHONE": r"(?<![\w.])(?:\+?84|0)(?:[ .-]?\d){8,10}(?!\d)",
    # 09/2027, 2020 - 2024, 2030-nay, Tháng 3 năm 2021 – Hiện tại
    "DATE": (r"(?i)(?:(?:tháng\s*)?\d{1,2}\s*(?:/|năm)\s*)?(?:19|20)\d{2}"
             r"(?:\s*[-–—~]\s*(?:(?:tháng\s*)?\d{1,2}\s*(?:/|năm)\s*)?(?:(?:19|20)\d{2}"
             r"|nay|hiện\s*(?:tại|nay)|present|now))?"),
}
_COMPILED = [(label, re.compile(p)) for label, p in PATTERNS.items()]


def find_fields(text: str) -> list[tuple[str, int, int, str]]:
    """Non-overlapping matches; earlier labels in PATTERNS win (EMAIL before URL...)."""
    taken: list[tuple[int, int]] = []
    out = []
    for label, rx in _COMPILED:
        for m in rx.finditer(text):
            s, e = m.span()
            if any(s < te and ts < e for ts, te in taken):
                continue
            taken.append((s, e))
            out.append((label, s, e, m.group().rstrip(".,;)")))
    return sorted(out, key=lambda x: x[1])


def _selfcheck() -> None:
    got = {(lab, txt) for lab, _, _, txt in find_fields(
        "Email: hello@reallygreatsite.com | ĐT: +84 912 345 678 | www.reallygreatsite.com "
        "| 0903414848 | 09/2027 – 05/2028 | 2030-nay | linkedin.com/in/name")}
    assert ("EMAIL", "hello@reallygreatsite.com") in got, got
    assert ("PHONE", "+84 912 345 678") in got, got
    assert ("PHONE", "0903414848") in got, got
    assert ("URL", "www.reallygreatsite.com") in got, got
    assert ("URL", "linkedin.com/in/name") in got, got
    assert ("DATE", "09/2027 – 05/2028") in got, got
    assert ("DATE", "2030-nay") in got, got
    assert not any(lab == "PHONE" for lab, *_ in find_fields("IELTS 7.5, GPA 3.6/4.0")), "false phone"


if __name__ == "__main__":
    _selfcheck()
    print("regex_fields ok")
