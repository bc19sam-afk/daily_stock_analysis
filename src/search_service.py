# -*- coding: utf-8 -*-
"""
===================================
A股/澳股自选股智能分析系统 - 搜索服务模块
===================================

职责：
1. 提供统一的新闻搜索接口
2. 支持 Tavily 和 SerpAPI 两种搜索引擎
3. 多 Key 负载均衡和故障转移
4. 搜索结果缓存和格式化
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from itertools import cycle
import requests
from newspaper import Article, Config

logger = logging.getLogger(__name__)


def fetch_url_content(url: str, timeout: int = 5) -> str:
    """
    获取 URL 网页正文内容 (使用 newspaper3k)
    """
    try:
        # 配置 newspaper3k
        config = Config()
        config.browser_user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        config.request_timeout = timeout
        config.fetch_images = False  # 不下载图片
        config.memoize_articles = False # 不缓存

        article = Article(url, config=config, language='zh') # 默认中文，但也支持其他
        article.download()
        article.parse()

        # 获取正文
        text = article.text.strip()

        # 简单的后处理，去除空行
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        return text[:1500]  # 限制返回长度
    except Exception as e:
        logger.debug(f"Fetch content failed for {url}: {e}")

    return ""


@dataclass
class SearchResult:
    """搜索结果数据类"""
    title: str
    snippet: str  # 摘要
    url: str
    source: str  # 来源网站
    published_date: Optional[str] = None
    
    def to_text(self) -> str:
        """转换为文本格式"""
        date_str = f" ({self.published_date})" if self.published_date else ""
        return f"【{self.source}】{self.title}{date_str}\n{self.snippet}"


@dataclass 
class SearchResponse:
    """搜索响应"""
    query: str
    results: List[SearchResult]
    provider: str  # 使用的搜索引擎
    success: bool = True
    error_message: Optional[str] = None
    search_time: float = 0.0  # 搜索耗时（秒）
    
    def to_context(self, max_results: int = 5) -> str:
        """将搜索结果转换为可用于 AI 分析的上下文"""
        if not self.success or not self.results:
            return f"搜索 '{self.query}' 未找到相关结果。"
        
        lines = [f"【{self.query} 搜索结果】（来源：{self.provider}）"]
        for i, result in enumerate(self.results[:max_results], 1):
            lines.append(f"\n{i}. {result.to_text()}")
        
        return "\n".join(lines)


class BaseSearchProvider(ABC):
    """搜索引擎基类"""
    
    def __init__(self, api_keys: List[str], name: str):
        self._api_keys = api_keys
        self._name = name
        self._key_cycle = cycle(api_keys) if api_keys else None
        self._key_usage: Dict[str, int] = {key: 0 for key in api_keys}
        self._key_errors: Dict[str, int] = {key: 0 for key in api_keys}
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def is_available(self) -> bool:
        """检查是否有可用的 API Key"""
        return bool(self._api_keys)
    
    def _get_next_key(self) -> Optional[str]:
        if not self._key_cycle:
            return None
        
        # 最多尝试所有 key
        for _ in range(len(self._api_keys)):
            key = next(self._key_cycle)
            # 跳过错误次数过多的 key（超过 3 次）
            if self._key_errors.get(key, 0) < 3:
                return key
        
        # 所有 key 都有问题，重置错误计数并返回第一个
        logger.warning(f"[{self._name}] 所有 API Key 都有错误记录，重置错误计数")
        self._key_errors = {key: 0 for key in self._api_keys}
        return self._api_keys[0] if self._api_keys else None
    
    def _record_success(self, key: str) -> None:
        self._key_usage[key] = self._key_usage.get(key, 0) + 1
        if key in self._key_errors and self._key_errors[key] > 0:
            self._key_errors[key] -= 1
    
    def _record_error(self, key: str) -> None:
        self._key_errors[key] = self._key_errors.get(key, 0) + 1
        logger.warning(f"[{self._name}] API Key {key[:8]}... 错误计数: {self._key_errors[key]}")
    
    @abstractmethod
    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        pass
    
    def search(self, query: str, max_results: int = 5, days: int = 7) -> SearchResponse:
        api_key = self._get_next_key()
        if not api_key:
            return SearchResponse(
                query=query, results=[], provider=self._name, success=False,
                error_message=f"{self._name} 未配置 API Key"
            )
        
        start_time = time.time()
        try:
            response = self._do_search(query, api_key, max_results, days=days)
            response.search_time = time.time() - start_time
            
            if response.success:
                self._record_success(api_key)
                logger.info(f"[{self._name}] 搜索 '{query}' 成功，返回 {len(response.results)} 条结果")
            else:
                self._record_error(api_key)
            
            return response
            
        except Exception as e:
            self._record_error(api_key)
            elapsed = time.time() - start_time
            logger.error(f"[{self._name}] 搜索 '{query}' 失败: {e}")
            return SearchResponse(
                query=query, results=[], provider=self._name, success=False,
                error_message=str(e), search_time=elapsed
            )


class TavilySearchProvider(BaseSearchProvider):
    """Tavily 搜索引擎 (ASX 优化)"""
    
    def __init__(self, api_keys: List[str]):
        super().__init__(api_keys, "Tavily")
    
    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        try:
            from tavily import TavilyClient
        except ImportError:
            return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message="tavily-python 未安装")
        
        try:
            client = TavilyClient(api_key=api_key)
            response = client.search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=False,
                include_raw_content=False,
                days=days,
            )
            
            results = []
            for item in response.get('results', []):
                results.append(SearchResult(
                    title=item.get('title', ''),
                    snippet=item.get('content', '')[:500],
                    url=item.get('url', ''),
                    source=self._extract_domain(item.get('url', '')),
                    published_date=item.get('published_date'),
                ))
            
            return SearchResponse(query=query, results=results, provider=self.name, success=True)
            
        except Exception as e:
            return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message=str(e))
    
    @staticmethod
    def _extract_domain(url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.replace('www.', '') or '未知来源'
        except:
            return '未知来源'


class SerpAPISearchProvider(BaseSearchProvider):
    """SerpAPI 搜索引擎"""
    
    def __init__(self, api_keys: List[str]):
        super().__init__(api_keys, "SerpAPI")
    
    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        try:
            from serpapi import GoogleSearch
        except ImportError:
            return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message="google-search-results 未安装")
        
        try:
            tbs = "qdr:w"
            if days <= 1: tbs = "qdr:d"
            elif days <= 30: tbs = "qdr:m"
            
            params = {
                "engine": "google", "q": query, "api_key": api_key,
                "google_domain": "google.com.hk", "hl": "zh-cn", "gl": "cn",
                "tbs": tbs, "num": max_results
            }
            
            search = GoogleSearch(params)
            response = search.get_dict()
            results = []
            
            # 解析自然搜索结果
            for item in response.get('organic_results', [])[:max_results]:
                results.append(SearchResult(
                    title=item.get('title', ''),
                    snippet=item.get('snippet', '')[:1000],
                    url=item.get('link', ''),
                    source=item.get('source', 'Google'),
                    published_date=item.get('date'),
                ))
            
            return SearchResponse(query=query, results=results, provider=self.name, success=True)
            
        except Exception as e:
            return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message=str(e))


class BochaSearchProvider(BaseSearchProvider):
    """博查搜索引擎"""
    
    def __init__(self, api_keys: List[str]):
        super().__init__(api_keys, "Bocha")
    
    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        try:
            url = "https://api.bocha.cn/v1/web-search"
            headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
            freshness = "oneWeek"
            if days <= 1: freshness = "oneDay"
            elif days > 30: freshness = "oneYear"

            payload = {
                "query": query, "freshness": freshness,
                "summary": True, "count": min(max_results, 50)
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            if response.status_code != 200:
                return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message=f"HTTP {response.status_code}: {response.text}")
            
            data = response.json()
            if data.get('code') != 200:
                return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message=data.get('msg'))
                
            results = []
            for item in data.get('data', {}).get('webPages', {}).get('value', [])[:max_results]:
                results.append(SearchResult(
                    title=item.get('name', ''),
                    snippet=(item.get('summary') or item.get('snippet', ''))[:500],
                    url=item.get('url', ''),
                    source=item.get('siteName', ''),
                    published_date=item.get('datePublished'),
                ))
                
            return SearchResponse(query=query, results=results, provider=self.name, success=True)
            
        except Exception as e:
            return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message=str(e))


class BraveSearchProvider(BaseSearchProvider):
    """Brave Search 搜索引擎"""

    def __init__(self, api_keys: List[str]):
        super().__init__(api_keys, "Brave")

    def _do_search(self, query: str, api_key: str, max_results: int, days: int = 7) -> SearchResponse:
        try:
            headers = {'X-Subscription-Token': api_key, 'Accept': 'application/json'}
            freshness = "pw" # 默认一周
            if days <= 1: freshness = "pd"
            elif days > 30: freshness = "py"
            
            params = {
                "q": query, "count": min(max_results, 20),
                "freshness": freshness, "search_lang": "en", "country": "US"
            }
            
            response = requests.get("https://api.search.brave.com/res/v1/web/search", headers=headers, params=params, timeout=10)
            if response.status_code != 200:
                return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message=f"HTTP {response.status_code}")

            data = response.json()
            results = []
            for item in data.get('web', {}).get('results', [])[:max_results]:
                results.append(SearchResult(
                    title=item.get('title', ''),
                    snippet=item.get('description', '')[:500],
                    url=item.get('url', ''),
                    source="Brave",
                    published_date=item.get('age')
                ))
            
            return SearchResponse(query=query, results=results, provider=self.name, success=True)
            
        except Exception as e:
            return SearchResponse(query=query, results=[], provider=self.name, success=False, error_message=str(e))


class SearchService:
    """
    搜索服务 (澳洲优化版)
    """
    
    # 增强搜索关键词模板（港股/美股 英文）
    ENHANCED_SEARCH_KEYWORDS_EN = [
        "{name} stock price today",
        "{name} {code} latest quote trend",
        "{name} stock analysis chart",
        "{name} technical analysis",
        "{name} {code} performance volume",
    ]
    
    # 增强搜索关键词模板（A股 中文）
    ENHANCED_SEARCH_KEYWORDS = [
        "{name} 股票 今日 股价",
        "{name} {code} 最新 行情 走势",
        "{name} 股票 分析 走势图",
    ]
    
    def __init__(
        self,
        bocha_keys: Optional[List[str]] = None,
        tavily_keys: Optional[List[str]] = None,
        brave_keys: Optional[List[str]] = None,
        serpapi_keys: Optional[List[str]] = None,
    ):
        """初始化搜索服务（已针对澳洲股票优化：Tavily 优先）"""
        self._providers: List[BaseSearchProvider] = []

        # 1. Tavily 优先（针对 ASX 澳洲股票搜索能力强）
        if tavily_keys:
            self._providers.append(TavilySearchProvider(tavily_keys))
            logger.info(f"已配置 Tavily 搜索，共 {len(tavily_keys)} 个 API Key")

        # 2. SerpAPI 第二（Google 原生搜索）
        if serpapi_keys:
            self._providers.append(SerpAPISearchProvider(serpapi_keys))
            logger.info(f"已配置 SerpAPI 搜索，共 {len(serpapi_keys)} 个 API Key")

        # 3. Brave Search 第三
        if brave_keys:
            self._providers.append(BraveSearchProvider(brave_keys))
            logger.info(f"已配置 Brave 搜索，共 {len(brave_keys)} 个 API Key")

        # 4. Bocha 降至最后（目前欠费，仅作最后备份）
        if bocha_keys:
            self._providers.append(BochaSearchProvider(bocha_keys))
            logger.info(f"已配置 Bocha 搜索，共 {len(bocha_keys)} 个 API Key")
        
        if not self._providers:
            logger.warning("未配置任何搜索引擎 API Key，新闻搜索功能将不可用")

        self._cache: Dict[str, Tuple[float, 'SearchResponse']] = {}
        self._cache_ttl: int = 600

    @staticmethod
    def _is_foreign_stock(stock_code: str) -> bool:
        """判断是否为港股或美股"""
        import re
        code = stock_code.strip()
        # 美股/澳股：1-5个大写字母，可能包含点（如 BRK.B, CBA.AX）
        if re.match(r'^[A-Za-z]{1,5}(\.[A-Za-z]+)?$', code):
            return True
        # 港股
        if code.lower().startswith('hk') or (code.isdigit() and len(code) == 5):
            return True
        return False

    def _cache_key(self, query: str, max_results: int, days: int) -> str:
        return f"{query}|{max_results}|{days}"

    def _get_cached(self, key: str) -> Optional['SearchResponse']:
        entry = self._cache.get(key)
        if entry is None: return None
        ts, response = entry
        if time.time() - ts > self._cache_ttl:
            del self._cache[key]
            return None
        return response

    def _put_cache(self, key: str, response: 'SearchResponse') -> None:
        self._cache[key] = (time.time(), response)
    
    def search_stock_news(self, stock_code: str, stock_name: str, max_results: int = 5, focus_keywords: Optional[List[str]] = None) -> SearchResponse:
        today_weekday = datetime.now().weekday()
        search_days = 3 if today_weekday == 0 else (2 if today_weekday >= 5 else 1)

        is_foreign = self._is_foreign_stock(stock_code)
        if focus_keywords:
            query = " ".join(focus_keywords)
        elif is_foreign:
            query = f"{stock_name} {stock_code} stock latest news"
        else:
            query = f"{stock_name} {stock_code} 股票 最新消息"

        logger.info(f"搜索股票新闻: {stock_name}({stock_code}), query='{query}'")
        
        cache_key = self._cache_key(query, max_results, search_days)
        cached = self._get_cached(cache_key)
        if cached: return cached

        for provider in self._providers:
            if not provider.is_available: continue
            response = provider.search(query, max_results, days=search_days)
            if response.success and response.results:
                self._put_cache(cache_key, response)
                return response
        
        return SearchResponse(query=query, results=[], provider="None", success=False, error_message="All providers failed")

    def search_stock_events(self, stock_code: str, stock_name: str, event_types: Optional[List[str]] = None) -> SearchResponse:
        if event_types is None:
            if self._is_foreign_stock(stock_code):
                event_types = ["earnings report", "insider selling", "quarterly results"]
            else:
                event_types = ["年报预告", "减持公告", "业绩快报"]
        
        query = f"{stock_name} ({' OR '.join(event_types)})"
        for provider in self._providers:
            if not provider.is_available: continue
            response = provider.search(query, max_results=5)
            if response.success: return response
            
        return SearchResponse(query=query, results=[], provider="None", success=False, error_message="Events search failed")

    def search_comprehensive_intel(self, stock_code: str, stock_name: str, max_searches: int = 3) -> Dict[str, SearchResponse]:
        results = {}
        is_foreign = self._is_foreign_stock(stock_code)
        
        if is_foreign:
            dims = [
                {'name': 'latest_news', 'query': f"{stock_name} {stock_code} latest news events", 'desc': '最新消息'},
                {'name': 'market_analysis', 'query': f"{stock_name} analyst rating target price report", 'desc': '机构分析'},
                {'name': 'risk_check', 'query': f"{stock_name} risk insider selling lawsuit litigation", 'desc': '风险排查'},
                {'name': 'earnings', 'query': f"{stock_name} earnings revenue profit growth forecast", 'desc': '业绩预期'},
                {'name': 'industry', 'query': f"{stock_name} industry competitors market share outlook", 'desc': '行业分析'},
            ]
        else:
            dims = [
                {'name': 'latest_news', 'query': f"{stock_name} 最新新闻", 'desc': '最新消息'},
                {'name': 'market_analysis', 'query': f"{stock_name} 研报 评级", 'desc': '机构分析'},
                {'name': 'risk_check', 'query': f"{stock_name} 利空 风险", 'desc': '风险排查'},
                {'name': 'earnings', 'query': f"{stock_name} 业绩预告", 'desc': '业绩预期'},
                {'name': 'industry', 'query': f"{stock_name} 行业分析", 'desc': '行业分析'},
            ]

        for dim in dims:
            # 始终优先使用 Tavily (如果配置了且排在第一位)
            for provider in self._providers:
                if not provider.is_available: continue
                resp = provider.search(dim['query'], max_results=3)
                results[dim['name']] = resp
                if resp.success: break # 只要有一个成功就跳出，进行下一个维度
                time.sleep(0.5)
                
        return results

    def format_intel_report(self, intel_results: Dict[str, SearchResponse], stock_name: str) -> str:
        lines = [f"【{stock_name} 情报搜索结果】"]
        order = ['latest_news', 'market_analysis', 'risk_check', 'earnings', 'industry']
        
        for dim_name in order:
            if dim_name not in intel_results: continue
            resp = intel_results[dim_name]
            dim_desc = {'latest_news': '📰 最新消息', 'market_analysis': '📈 机构分析', 'risk_check': '⚠️ 风险排查', 'earnings': '📊 业绩预期', 'industry': '🏭 行业分析'}.get(dim_name, dim_name)
            
            lines.append(f"\n{dim_desc} (来源: {resp.provider}):")
            if resp.success and resp.results:
                for i, r in enumerate(resp.results[:3], 1):
                    date_str = f" [{r.published_date}]" if r.published_date else ""
                    lines.append(f"  {i}. {r.title}{date_str}")
            else:
                lines.append("  未找到相关信息")
        
        return "\n".join(lines)


# === 便捷函数 ===
_search_service: Optional[SearchService] = None

def get_search_service() -> SearchService:
    global _search_service
    if _search_service is None:
        from src.config import get_config
        config = get_config()
        _search_service = SearchService(
            bocha_keys=config.bocha_api_keys,
            tavily_keys=config.tavily_api_keys,
            brave_keys=config.brave_api_keys,
            serpapi_keys=config.serpapi_keys,
        )
    return _search_service

def reset_search_service() -> None:
    global _search_service
    _search_service = None

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    service = get_search_service()
    if service.is_available:
        print("=== 测试股票新闻搜索 ===")
        # 测试澳股代码
        response = service.search_stock_news("CBA.AX", "CommBank")
        print(f"搜索状态: {'成功' if response.success else '失败'}")
        print(f"搜索引擎: {response.provider}")
        print("\n" + response.to_context())
