"""
Check for core.ingestion.PoliteFetcher's robots.txt rules (RFC 9309).
No network needed: `python test_robots.py`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.ingestion import PoliteFetcher


class _Resp:
    def __init__(self, status: int, text: str = ""):
        self.status_code, self.text = status, text

    def raise_for_status(self):
        pass


class _Session:
    def __init__(self, robots: dict):
        self.robots = robots

    def get(self, url, timeout=None):
        site = url.split("/")[2]
        if url.endswith("/robots.txt"):
            if isinstance(self.robots[site], Exception):
                raise self.robots[site]
            return self.robots[site]
        return _Resp(200, "page")


def test_robots():
    fetcher = PoliteFetcher()
    fetcher.session = _Session({
        "a.it": _Resp(200, "User-agent: *\nDisallow: /private/\n"),
        "b.it": _Resp(200, "User-agent: MorningBrief\nDisallow: /\n"),
        "c.it": _Resp(404),
        "d.it": _Resp(503),
        "e.it": ConnectionError("down"),
    })
    assert fetcher.get("https://a.it/news/1") is not None
    assert fetcher.get("https://a.it/private/1") is None
    assert fetcher.get("https://b.it/news/1") is None  # rule aimed at this reader
    assert fetcher.get("https://c.it/news/1") is not None  # no robots.txt: allowed
    assert fetcher.get("https://d.it/news/1") is None  # server error: assume disallowed
    assert fetcher.get("https://e.it/news/1") is None  # unreachable: assume disallowed


if __name__ == "__main__":
    test_robots()
    print("ok")
