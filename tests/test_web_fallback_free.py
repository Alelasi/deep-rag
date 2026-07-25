"""免费 Web 搜索链路（Google News RSS 等，无 Key）+ 今日硬过滤"""


def test_google_news_rss_today():
    from src.retrieval.web_fallback import web_search_fallback
    from src.pipeline_routing import filter_real_web_results

    r = web_search_fallback("今日新闻", max_results=5, engine="google_news")
    real = filter_real_web_results(r)
    assert len(real) >= 1
    assert all(not (x.get("metadata") or {}).get("is_mock") for x in real)
    assert any((x.get("metadata") or {}).get("title") for x in real)


def test_auto_cascade_not_mock_for_news():
    from src.retrieval.web_fallback import web_search_fallback
    from src.pipeline_routing import filter_real_web_results

    r = web_search_fallback("今日新闻", max_results=5, engine="auto")
    real = filter_real_web_results(r)
    assert len(real) >= 1
    engines = {(x.get("metadata") or {}).get("engine") for x in real}
    assert "mock" not in engines


def test_today_news_all_same_cn_day():
    from src.retrieval.web_fallback import web_search_today_news, today_cn

    day = today_cn()
    r = web_search_today_news("今日新闻", max_results=8)
    assert len(r) >= 1
    for x in r:
        d = (x.get("metadata") or {}).get("date") or ""
        assert d.startswith(day.isoformat()), f"not today: {d} expected {day}"
