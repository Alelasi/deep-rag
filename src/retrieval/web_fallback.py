"""Web Fallback — 知识库无答案时的外部搜索兜底

免费优先链路（无需 API Key）：
1. DuckDuckGo text（ddgs / duckduckgo_search）
2. DuckDuckGo news（时效问题）
3. Google News RSS（中文/英文，免费公开）
4. 公共新闻 RSS（BBC 中文等）
5. Wikipedia API（概念补充，非实时）

可选付费/Key：Tavily、Serper（配置后优先于 mock）

默认 engine=auto：按上序级联，过滤 mock 与明显 junk。
"""
from __future__ import annotations

import html as html_lib
import logging
import os
import re
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional
from urllib.parse import quote, urlparse

log = logging.getLogger("deeprag.web_fallback")

# 东八区：用户「今天」按国内日历日
CN_TZ = timezone(timedelta(hours=8))

# 延迟导入搜索库（优先新包 ddgs）
try:
    from ddgs import DDGS

    DDGS_AVAILABLE = True
except ImportError:
    try:
        from duckduckgo_search import DDGS

        DDGS_AVAILABLE = True
    except ImportError:
        DDGS_AVAILABLE = False
        log.warning("ddgs/duckduckgo-search not installed. Run: pip install ddgs")

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    log.warning("requests not installed. Run: pip install requests")

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# 明显 junk（DDG 中文查询常被污染）
_JUNK_TITLE_RE = re.compile(
    r"(contact\s*us|microsoft\s*support|cards?\s*chat|merriam[- ]webster|"
    r"dictionary\.com|login|sign\s*in|cookie|privacy\s*policy)",
    re.I,
)
_JUNK_HOST = {
    "support.microsoft.com",
    "login.microsoftonline.com",
    "www.merriam-webster.com",
    "www.cardschat.com",
    "dictionary.cambridge.org",
}

_NEWS_HINT_RE = re.compile(
    r"(新闻|要闻|资讯|头条|联播|热点|breaking|headline|today'?s\s*news|latest\s*news)",
    re.I,
)


# ---------------------------------------------------------------------------
# 环境变量驱动的配置（生产化：避免静默造假）
# ---------------------------------------------------------------------------
# WEB_FALLBACK_MOCK     : "true"/"1"/"yes" -> 强制返回带 is_mock=True 的占位结果
# ENABLE_WEB_FALLBACK   : "false"/"0"/"no" -> 关闭 Web 兜底（返回 []，不造假）
# WEB_FALLBACK_ENGINE   : 选择引擎
#                         (auto|duckduckgo|news|tavily|serper|google_news|rss|today_news)
# 原则：除非显式开启 mock，否则绝不静默返回假数据；真实检索失败时明确报错（返回 []
#       并记录 error 日志）。
def _env_flag(name: str) -> Optional[bool]:
    """读取布尔型环境变量：未设置返回 None，否则按常见真假词解析。"""
    v = os.getenv(name)
    if v is None:
        return None
    s = v.strip().lower()
    if s in ("1", "true", "yes", "on", "y", "启用", "开启"):
        return True
    if s in ("0", "false", "no", "off", "n", "禁用", "关闭"):
        return False
    return None


def _mock_mode_enabled() -> bool:
    return _env_flag("WEB_FALLBACK_MOCK") is True


def _web_fallback_disabled() -> bool:
    return _env_flag("ENABLE_WEB_FALLBACK") is False


class WebFallbackError(RuntimeError):
    """真实检索路径彻底失败时的明确错误信号（不静默造假）。"""


def today_cn(now: Optional[datetime] = None) -> date:
    """返回东八区今天的 date。"""
    n = now or datetime.now(CN_TZ)
    if n.tzinfo is None:
        n = n.replace(tzinfo=CN_TZ)
    return n.astimezone(CN_TZ).date()


def parse_pub_datetime(value: str | None) -> Optional[datetime]:
    """解析 RSS/新闻 pubDate → aware datetime（失败返回 None）。"""
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        dt = parsedate_to_datetime(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
    ):
        try:
            raw = s.replace("Z", "+0000") if fmt.endswith("%z") else s
            dt = datetime.strptime(raw[:26], fmt.replace("%z", "%z") if "%z" in fmt else fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            continue
    # 标题/摘要里夹带 2026-07-18 / 7月18日
    m = re.search(r"(20\d{2})[-/年](\d{1,2})[-/月](\d{1,2})", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=CN_TZ)
        except Exception:
            return None
    return None


def is_same_calendar_day(dt: datetime, day=None, tz=CN_TZ) -> bool:
    """是否同一日历日（默认东八区今天）。"""
    target = day or today_cn()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(tz).date() == target


def filter_today_results(
    items: List[Dict],
    *,
    day=None,
    allow_missing_date: bool = False,
) -> List[Dict]:
    """只保留「今天」（东八区）发布的条目；缺日期默认丢弃。"""
    target = day or today_cn()
    kept: List[Dict] = []
    for it in items or []:
        meta = dict(it.get("metadata") or {})
        raw = meta.get("date") or meta.get("pubDate") or meta.get("published") or ""
        # 正文/标题兜底解析
        blob = f"{raw} {meta.get('title','')} {it.get('content','')}"
        dt = parse_pub_datetime(str(raw)) or parse_pub_datetime(blob[:80])
        if dt is None:
            if allow_missing_date:
                meta["date_status"] = "unknown"
                kept.append({**it, "metadata": meta})
            continue
        if is_same_calendar_day(dt, target):
            meta["date"] = dt.astimezone(CN_TZ).strftime("%Y-%m-%d %H:%M")
            meta["date_status"] = "today"
            meta["pub_ts"] = dt.timestamp()
            kept.append({**it, "metadata": meta})
    # 新→旧
    kept.sort(key=lambda x: float((x.get("metadata") or {}).get("pub_ts") or 0), reverse=True)
    return kept


def web_search_fallback(
    query: str,
    max_results: int = 3,
    engine: str = "auto",
    *,
    today_only: bool = False,
) -> List[Dict]:
    """外部搜索兜底（环境变量驱动）。

    Args:
        query: 搜索查询
        max_results: 最大结果数
        engine: auto | duckduckgo | news | tavily | serper | google_news | rss | today_news
        today_only: True 时硬过滤为东八区「今天」

    行为（避免静默造假）：
        - WEB_FALLBACK_MOCK=true        -> 返回带 is_mock=True 的占位结果
        - ENABLE_WEB_FALLBACK=false     -> 关闭兜底，返回 []
        - 引擎完全未配置（无 env、无显式 engine、且未启用）-> 离线占位（mock，带告警）
        - 其余情况走真实检索；真实检索全失败时返回 [] 并记录 error（不造假）
    """
    q = (query or "").strip()
    if not q:
        return []

    original_engine = (engine or "auto").lower().strip()
    env_engine = os.getenv("WEB_FALLBACK_ENGINE")
    # env 引擎仅在调用方使用默认 auto 时覆盖
    if original_engine == "auto" and env_engine:
        effective_engine = env_engine.lower().strip()
    else:
        effective_engine = original_engine

    # 1) 显式 mock
    if _mock_mode_enabled():
        log.warning("WEB_FALLBACK_MOCK=true -> 返回 MOCK 占位结果（非真实检索）。")
        return _mock_results(q, max_results)

    # 2) 显式关闭
    if _web_fallback_disabled():
        log.info("ENABLE_WEB_FALLBACK=false -> Web 兜底已关闭，返回 []。")
        return []

    # 3) 完全未配置 -> 离线占位（明确告警，可识别）
    caller_explicit = original_engine != "auto"
    if (not caller_explicit) and (not env_engine) and (
        _env_flag("ENABLE_WEB_FALLBACK") is not True
    ):
        log.warning(
            "Web 兜底未配置（WEB_FALLBACK_ENGINE 未设置且未启用）-> 返回 MOCK 占位结果。"
        )
        return _mock_results(q, max_results)

    # 4) 真实检索路径
    engine = effective_engine
    if engine in ("today_news", "today"):
        return web_search_today_news(q, max_results=max_results)

    if engine == "auto":
        results = _search_auto(q, max_results if not today_only else max(max_results * 3, 12))
    elif engine in ("duckduckgo", "ddg"):
        results = _search_duckduckgo(q, max_results)
    elif engine in ("news", "ddg_news"):
        results = _search_duckduckgo_news(q, max_results)
    elif engine in ("google_news", "gn"):
        results = _search_google_news_rss(q, max_results if not today_only else max(max_results * 2, 15))
    elif engine == "rss":
        results = _search_public_rss(q, max_results)
    elif engine == "tavily":
        results = _search_tavily(q, max_results)
    elif engine == "serper":
        results = _search_serper(q, max_results)
    else:
        log.error("Unknown search engine: %s —— 真实检索失败，返回 []（不造假）。", engine)
        return []

    if today_only:
        today_items = filter_today_results(results, allow_missing_date=False)
        if today_items:
            return today_items[:max_results]
        # 新闻意图：再拉今日专用源
        if _NEWS_HINT_RE.search(q):
            return web_search_today_news(q, max_results=max_results)
        return []
    return results[:max_results] if results else results


def web_search_today_news(query: str = "今日新闻", max_results: int = 10) -> List[Dict]:
    """强制「今天」新闻：多路 Google News RSS + 日期硬过滤（免费）。"""
    if _mock_mode_enabled():
        log.warning("WEB_FALLBACK_MOCK=true -> web_search_today_news 返回 MOCK 占位。")
        return _mock_results(query, max_results)
    day = today_cn()
    day_s = day.isoformat()
    # 多路拉取，再统一 filter_today
    feeds_q = [
        # 中文头条
        "https://news.google.com/rss?hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        # 今日关键词
        f"https://news.google.com/rss/search?q={quote('今日 when:1d')}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        f"https://news.google.com/rss/search?q={quote(f'after:{day_s}')}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        f"https://news.google.com/rss/search?q={quote('新闻 when:1d')}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        # 英文当日
        "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
        f"https://news.google.com/rss/search?q={quote('when:1d')}&hl=en-US&gl=US&ceid=US:en",
    ]
    # 用户具体查询（非极短）再加一路
    q = (query or "").strip()
    if q and q not in ("今日新闻", "今天新闻", "新闻", "要闻", "热点", "今日要闻", "最新新闻"):
        feeds_q.insert(
            1,
            f"https://news.google.com/rss/search?q={quote(q + ' when:1d')}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
        )

    raw_items: List[Dict] = []
    seen: set[str] = set()
    for url in feeds_q:
        try:
            items = _parse_rss_items(_http_get(url), limit=40)
        except Exception as e:  # noqa: BLE001
            log.debug("today feed fail %s: %s", url[:50], e)
            continue
        for it in items:
            title = it.get("title") or ""
            link = it.get("link") or ""
            key = link or title
            if not key or key in seen:
                continue
            seen.add(key)
            body = it.get("description") or title
            raw_items.append(
                {
                    "doc_id": f"web_today_{len(raw_items)}",
                    "content": body[:800],
                    "source": link or f"https://news.google.com/search?q={quote(title)}",
                    "page": 0,
                    "metadata": {
                        "is_web": True,
                        "engine": "google_news_today",
                        "title": title,
                        "snippet": body[:200],
                        "date": it.get("pubDate") or "",
                        "is_mock": False,
                    },
                }
            )

    # DDG news 作为补充（可能限流）
    try:
        ddg = _search_duckduckgo_news(q or "今日新闻", max_results=max(max_results, 8))
        for it in ddg:
            key = it.get("source") or (it.get("metadata") or {}).get("title")
            if key and key not in seen:
                seen.add(str(key))
                raw_items.append(it)
    except Exception as e:  # noqa: BLE001
        log.debug("today ddg news skip: %s", e)

    today_items = filter_today_results(raw_items, day=day, allow_missing_date=False)
    log.info(
        "web_search_today_news day=%s raw=%s today=%s query=%s",
        day_s,
        len(raw_items),
        len(today_items),
        (q or "")[:40],
    )
    return today_items[:max_results]


def _search_auto(query: str, max_results: int) -> List[Dict]:
    """免费级联：DDG → DDG News → Google News RSS → 公共 RSS →（可选）Tavily/Serper。"""
    is_news = bool(_NEWS_HINT_RE.search(query))
    chain: list = []
    if is_news:
        chain = [
            ("ddg_news", lambda: _search_duckduckgo_news(query, max_results)),
            ("google_news", lambda: _search_google_news_rss(query, max_results)),
            ("rss", lambda: _search_public_rss(query, max_results)),
            ("ddg", lambda: _search_duckduckgo(query, max_results)),
        ]
    else:
        chain = [
            ("ddg", lambda: _search_duckduckgo(query, max_results)),
            ("google_news", lambda: _search_google_news_rss(query, max_results)),
            ("rss", lambda: _search_public_rss(query, max_results)),
        ]

    # 有 Key 时插入付费引擎（仍免费用户可跳过）
    if os.getenv("TAVILY_API_KEY"):
        chain.insert(0, ("tavily", lambda: _search_tavily(query, max_results)))
    if os.getenv("SERPER_API_KEY"):
        chain.insert(0 if not os.getenv("TAVILY_API_KEY") else 1, ("serper", lambda: _search_serper(query, max_results)))

    seen_urls: set[str] = set()
    merged: List[Dict] = []
    for name, fn in chain:
        try:
            batch = fn() or []
        except Exception as e:  # noqa: BLE001
            log.warning("search engine %s failed: %s", name, e)
            continue
        for item in batch:
            if _is_mock_item(item) or _is_junk_item(item):
                continue
            url = (item.get("source") or "").strip()
            key = url or (item.get("metadata") or {}).get("title", "")
            if key and key in seen_urls:
                continue
            if key:
                seen_urls.add(key)
            merged.append(item)
            if len(merged) >= max_results:
                log.info(
                    "web auto cascade ok: engine_hint=%s n=%s query=%s",
                    name,
                    len(merged),
                    query[:50],
                )
                return merged[:max_results]

    if merged:
        log.info("web auto cascade partial n=%s query=%s", len(merged), query[:50])
        return merged[:max_results]

    log.warning("web auto cascade empty for: %s —— 真实检索无结果，返回 []（不造假）。", query[:50])
    return []


def _is_mock_item(item: Dict) -> bool:
    meta = item.get("metadata") or {}
    if meta.get("is_mock") or meta.get("engine") == "mock":
        return True
    src = str(item.get("source") or "")
    return src.startswith("mock://")


def _is_junk_item(item: Dict) -> bool:
    meta = item.get("metadata") or {}
    title = str(meta.get("title") or "")
    body = str(item.get("content") or "")
    src = str(item.get("source") or "")
    if _JUNK_TITLE_RE.search(title):
        return True
    try:
        host = urlparse(src).netloc.lower().removeprefix("www.")
    except Exception:
        host = ""
    if host in _JUNK_HOST:
        return True
    # 空正文且标题极短
    if len(body.strip()) < 8 and len(title.strip()) < 4:
        return True
    return False


def _strip_html(text: str) -> str:
    if not text:
        return ""
    t = re.sub(r"<[^>]+>", " ", text)
    t = html_lib.unescape(t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _with_cleared_proxy(fn):
    """临时清除代理，避免 DDG 被错误代理劫持。"""
    keys = ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]
    old = {k: os.environ.pop(k, None) for k in keys}
    try:
        return fn()
    finally:
        for k, v in old.items():
            if v is not None:
                os.environ[k] = v


def _search_duckduckgo(query: str, max_results: int) -> List[Dict]:
    """DuckDuckGo 文本搜索（免费）。"""
    if not DDGS_AVAILABLE:
        log.warning("DuckDuckGo search not available")
        return []

    def _run():
        ddgs = DDGS()
        results: List[Dict] = []
        # 中文优先 wt-wt；失败再试 cn-zh
        variants = [
            {"region": "wt-wt"},
            {"region": "cn-zh"},
            {},
        ]
        seen = set()
        for kw in variants:
            try:
                for i, result in enumerate(ddgs.text(query, max_results=max_results + 2, **kw)):
                    href = result.get("href") or result.get("url") or ""
                    if href in seen:
                        continue
                    seen.add(href)
                    item = {
                        "doc_id": f"web_ddg_{len(results)}",
                        "content": result.get("body", "") or "",
                        "source": href,
                        "page": 0,
                        "metadata": {
                            "is_web": True,
                            "is_mock": False,
                            "engine": "duckduckgo",
                            "title": result.get("title", ""),
                            "snippet": (result.get("body") or "")[:200],
                        },
                    }
                    if not _is_junk_item(item):
                        results.append(item)
                    if len(results) >= max_results:
                        return results
            except Exception as e:  # noqa: BLE001
                log.debug("ddg variant %s failed: %s", kw, e)
                continue
        return results

    try:
        results = _with_cleared_proxy(_run)
        log.info("DuckDuckGo text returned %s for: %s", len(results), query[:40])
        return results
    except Exception as e:  # noqa: BLE001
        log.error("DuckDuckGo search failed: %s", e)
        return []


def _search_duckduckgo_news(query: str, max_results: int) -> List[Dict]:
    """DuckDuckGo 新闻垂类（免费，易限流）。"""
    if not DDGS_AVAILABLE:
        return []

    def _run():
        ddgs = DDGS()
        results: List[Dict] = []
        for region in ("cn-zh", "wt-wt"):
            try:
                for result in ddgs.news(query, max_results=max_results + 2, region=region):
                    href = result.get("url") or result.get("href") or ""
                    body = result.get("body") or result.get("excerpt") or ""
                    item = {
                        "doc_id": f"web_ddgnews_{len(results)}",
                        "content": body,
                        "source": href,
                        "page": 0,
                        "metadata": {
                            "is_web": True,
                            "is_mock": False,
                            "engine": "duckduckgo_news",
                            "title": result.get("title", ""),
                            "snippet": body[:200],
                            "date": result.get("date") or result.get("published") or "",
                            "source_name": result.get("source") or "",
                        },
                    }
                    if not _is_junk_item(item):
                        results.append(item)
                    if len(results) >= max_results:
                        return results
            except Exception as e:  # noqa: BLE001
                log.debug("ddg news region=%s failed: %s", region, e)
                continue
        return results

    try:
        results = _with_cleared_proxy(_run)
        log.info("DuckDuckGo news returned %s for: %s", len(results), query[:40])
        return results
    except Exception as e:  # noqa: BLE001
        log.error("DuckDuckGo news failed: %s", e)
        return []


def _http_get(url: str, timeout: float = 12.0) -> bytes:
    if REQUESTS_AVAILABLE:
        r = requests.get(
            url,
            headers={"User-Agent": _UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.content
    import urllib.request

    req = urllib.request.Request(
        url,
        headers={"User-Agent": _UA, "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _parse_rss_items(raw: bytes, limit: int = 20) -> List[Dict]:
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        log.warning("RSS parse error: %s", e)
        return []
    out: List[Dict] = []
    for it in root.findall(".//item"):
        title = _strip_html(it.findtext("title") or "")
        link = (it.findtext("link") or "").strip()
        desc = _strip_html(it.findtext("description") or "")
        pub = (it.findtext("pubDate") or it.findtext("published") or "").strip()
        if not title and not desc:
            continue
        out.append(
            {
                "title": title,
                "link": link,
                "description": desc,
                "pubDate": pub,
            }
        )
        if len(out) >= limit:
            break
    return out


def _search_google_news_rss(query: str, max_results: int) -> List[Dict]:
    """Google News 公开 RSS（免费、无需 Key，适合今日新闻）。"""
    q = query.strip()
    # 对极短「今日新闻」用头条 feed；否则用搜索 RSS
    if q in ("今日新闻", "今天新闻", "新闻", "要闻", "热点", "今日要闻", "最新新闻") or len(q) <= 6:
        urls = [
            "https://news.google.com/rss?hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
            "https://news.google.com/rss?hl=en-US&gl=US&ceid=US:en",
        ]
    else:
        q_enc = quote(q)
        urls = [
            f"https://news.google.com/rss/search?q={q_enc}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
            f"https://news.google.com/rss/search?q={q_enc}&hl=en-US&gl=US&ceid=US:en",
        ]

    results: List[Dict] = []
    for url in urls:
        try:
            raw = _http_get(url)
            items = _parse_rss_items(raw, limit=max_results + 5)
            for it in items:
                title = it["title"]
                link = it["link"] or f"https://news.google.com/search?q={quote(title)}"
                body = it["description"] or title
                # Google RSS description 常含「标题 来源」
                results.append(
                    {
                        "doc_id": f"web_gn_{len(results)}",
                        "content": body[:800],
                        "source": link,
                        "page": 0,
                        "metadata": {
                            "is_web": True,
                            "engine": "google_news_rss",
                            "title": title,
                            "snippet": body[:200],
                            "date": it.get("pubDate") or "",
                            "is_mock": False,
                        },
                    }
                )
                if len(results) >= max_results:
                    log.info("Google News RSS returned %s for: %s", len(results), q[:40])
                    return results
        except Exception as e:  # noqa: BLE001
            log.warning("Google News RSS failed %s: %s", url[:60], e)
            continue

    log.info("Google News RSS returned %s for: %s", len(results), q[:40])
    return results


def _search_public_rss(query: str, max_results: int) -> List[Dict]:
    """公共媒体 RSS 兜底（BBC 中文等，免费）。"""
    feeds = [
        ("bbc_zh", "https://feeds.bbci.co.uk/zhongwen/simp/rss.xml"),
        ("bbc_zh_alt", "http://www.bbc.com/zhongwen/simp/index.xml"),
        ("nyt_world", "https://rss.nytimes.com/services/xml/rss/nyt/World.xml"),
    ]
    # 关键词粗过滤：对「今日新闻」不过滤；对其它 query 做简单包含
    q_tokens = [t for t in re.split(r"\s+", query) if len(t) >= 2]
    loose = bool(_NEWS_HINT_RE.search(query)) or len(query) <= 8

    results: List[Dict] = []
    for name, url in feeds:
        try:
            raw = _http_get(url)
            items = _parse_rss_items(raw, limit=30)
            for it in items:
                title = it["title"]
                body = it["description"] or title
                blob = title + " " + body
                if not loose and q_tokens:
                    if not any(t in blob for t in q_tokens[:4]):
                        continue
                results.append(
                    {
                        "doc_id": f"web_rss_{name}_{len(results)}",
                        "content": body[:800],
                        "source": it["link"] or url,
                        "page": 0,
                    "metadata": {
                        "is_web": True,
                        "is_mock": False,
                        "engine": f"rss:{name}",
                        "title": title,
                        "snippet": body[:200],
                        "date": it.get("pubDate") or "",
                    },
                    }
                )
                if len(results) >= max_results:
                    log.info("Public RSS returned %s via %s", len(results), name)
                    return results
        except Exception as e:  # noqa: BLE001
            log.debug("RSS %s failed: %s", name, e)
            continue
    return results


def _search_tavily(query: str, max_results: int) -> List[Dict]:
    """Tavily API（需 Key）。"""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []
    if not REQUESTS_AVAILABLE:
        return []
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": False,
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for i, item in enumerate(data.get("results", [])):
            results.append(
                {
                    "doc_id": f"web_tavily_{i}",
                    "content": item.get("content", ""),
                    "source": item.get("url", ""),
                    "page": 0,
                    "metadata": {
                        "is_web": True,
                        "is_mock": False,
                        "engine": "tavily",
                        "title": item.get("title", ""),
                        "snippet": item.get("content", "")[:200],
                        "score": item.get("score", 0.0),
                    },
                }
            )
        log.info("Tavily search returned %s for: %s", len(results), query[:40])
        return results
    except Exception as e:  # noqa: BLE001
        log.error("Tavily search failed: %s", e)
        return []


def _search_serper(query: str, max_results: int) -> List[Dict]:
    """Serper API（需 Key）。"""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return []
    if not REQUESTS_AVAILABLE:
        return []
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": max_results},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        results = []
        for i, item in enumerate(data.get("organic", [])):
            results.append(
                {
                    "doc_id": f"web_serper_{i}",
                    "content": item.get("snippet", ""),
                    "source": item.get("link", ""),
                    "page": 0,
                    "metadata": {
                        "is_web": True,
                        "is_mock": False,
                        "engine": "serper",
                        "title": item.get("title", ""),
                        "snippet": item.get("snippet", ""),
                        "position": item.get("position", 0),
                    },
                }
            )
        log.info("Serper search returned %s for: %s", len(results), query[:40])
        return results
    except Exception as e:  # noqa: BLE001
        log.error("Serper search failed: %s", e)
        return []


def _mock_results(query: str, max_results: int) -> List[Dict]:
    """Mock 占位（不可作证据）。"""
    return [
        {
            "doc_id": f"web_mock_{i}",
            "content": f"[Web搜索结果占位] 查询: {query}",
            "source": "mock://web_search",
            "page": 0,
            "metadata": {
                "is_web": True,
                "engine": "mock",
                "is_mock": True,
            },
        }
        for i in range(max_results)
    ]


class WebSearchFallback:
    """面向上层（enhanced_knowledge_retrieval 等）的面向对象封装。

    与函数式 API 行为一致，同样受环境变量约束：
    - WEB_FALLBACK_MOCK=true   -> search() 返回带 is_mock=True 的占位结果
    - ENABLE_WEB_FALLBACK=false -> search() 返回 []
    - 未配置                   -> 离线占位（mock，带告警）
    """

    def __init__(self, engine: Optional[str] = None, max_results: int = 3):
        env_engine = os.getenv("WEB_FALLBACK_ENGINE")
        self.engine = (engine or env_engine or "auto").lower().strip()
        self.max_results = max_results

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        n = top_k if top_k is not None else self.max_results
        return web_search_fallback(query, max_results=n, engine=self.engine)

    def is_mock_result(self, item: Dict) -> bool:
        """上层据此区分真实结果与占位结果。"""
        return bool((item.get("metadata") or {}).get("is_mock"))


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    test_query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "今日新闻"
    print("=== Web Fallback 测试 ===")
    print("查询:", test_query)
    results = web_search_fallback(test_query, max_results=5, engine="auto")
    for i, result in enumerate(results, 1):
        meta = result.get("metadata") or {}
        print(f"[{i}] [{meta.get('engine')}] {meta.get('title', '')[:60]}")
        print(f"    {result.get('source', '')[:90]}")
        print(f"    {(result.get('content') or '')[:120]}")
