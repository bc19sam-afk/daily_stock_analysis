# -*- coding: utf-8 -*-
"""
===================================
数据源基类与管理器
===================================

设计模式：策略模式 (Strategy Pattern)
- BaseFetcher: 抽象基类，定义统一接口
- DataFetcherManager: 策略管理器，实现自动切换

防封禁策略：
1. 每个 Fetcher 内置流控逻辑
2. 失败自动切换到下一个数据源
3. 指数退避重试机制
"""

import logging
import random
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any

import pandas as pd
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# 配置日志
logger = logging.getLogger(__name__)


# === 标准化列名定义 ===
STANDARD_COLUMNS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']


class DataFetchError(Exception):
    """数据获取异常基类"""
    pass


class RateLimitError(DataFetchError):
    """API 速率限制异常"""
    pass


class DataSourceUnavailableError(DataFetchError):
    """数据源不可用异常"""
    pass


class BaseFetcher(ABC):
    """
    数据源抽象基类
    
    职责：
    1. 定义统一的数据获取接口
    2. 提供数据标准化方法
    3. 实现通用的技术指标计算
    
    子类实现：
    - _fetch_raw_data(): 从具体数据源获取原始数据
    - _normalize_data(): 将原始数据转换为标准格式
    """
    
    name: str = "BaseFetcher"
    priority: int = 99  # 优先级数字越小越优先
    
    @abstractmethod
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从数据源获取原始数据（子类必须实现）
        
        Args:
            stock_code: 股票代码，如 '600519', '000001'
            start_date: 开始日期，格式 'YYYY-MM-DD'
            end_date: 结束日期，格式 'YYYY-MM-DD'
            
        Returns:
            原始数据 DataFrame（列名因数据源而异）
        """
        pass
    
    @abstractmethod
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化数据列名（子类必须实现）

        将不同数据源的列名统一为：
        ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        """
        pass

    def get_main_indices(self) -> Optional[List[Dict[str, Any]]]:
        """
        获取主要指数实时行情

        Returns:
            List[Dict]: 指数列表，每个元素为字典，包含:
                - code: 指数代码
                - name: 指数名称
                - current: 当前点位
                - change: 涨跌点数
                - change_pct: 涨跌幅(%)
                - volume: 成交量
                - amount: 成交额
        """
        return None

    def get_market_stats(self) -> Optional[Dict[str, Any]]:
        """
        获取市场涨跌统计

        Returns:
            Dict: 包含:
                - up_count: 上涨家数
                - down_count: 下跌家数
                - flat_count: 平盘家数
                - limit_up_count: 涨停家数
                - limit_down_count: 跌停家数
                - total_amount: 两市成交额
        """
        return None

    def get_sector_rankings(self, n: int = 5) -> Optional[Tuple[List[Dict], List[Dict]]]:
        """
        获取板块涨跌榜

        Args:
            n: 返回前n个

        Returns:
            Tuple: (领涨板块列表, 领跌板块列表)
        """
        return None

    def get_daily_data(
        self,
        stock_code: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> pd.DataFrame:
        """
        获取日线数据（统一入口）
        
        流程：
        1. 计算日期范围
        2. 调用子类获取原始数据
        3. 标准化列名
        4. 计算技术指标
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期（可选）
            end_date: 结束日期（可选，默认今天）
            days: 获取天数（当 start_date 未指定时使用）
            
        Returns:
            标准化的 DataFrame，包含技术指标
        """
        # 计算日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if start_date is None:
            # 默认获取最近 30 个交易日（按日历日估算，多取一些）
            from datetime import timedelta
            start_dt = datetime.strptime(end_date, '%Y-%m-%d') - timedelta(days=days * 2)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        logger.info(f"[{self.name}] 获取 {stock_code} 数据: {start_date} ~ {end_date}")
        
        try:
            # Step 1: 获取原始数据
            raw_df = self._fetch_raw_data(stock_code, start_date, end_date)
            
            if raw_df is None or raw_df.empty:
                raise DataFetchError(f"[{self.name}] 未获取到 {stock_code} 的数据")
            
            # Step 2: 标准化列名
            df = self._normalize_data(raw_df, stock_code)
            
            # Step 3: 数据清洗
            df = self._clean_data(df)
            
            # Step 4: 计算技术指标
            df = self._calculate_indicators(df)
            
            logger.info(f"[{self.name}] {stock_code} 获取成功，共 {len(df)} 条数据")
            return df
            
        except Exception as e:
            logger.error(f"[{self.name}] 获取 {stock_code} 失败: {str(e)}")
            raise DataFetchError(f"[{self.name}] {stock_code}: {str(e)}") from e
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据清洗
        
        处理：
        1. 确保日期列格式正确
        2. 数值类型转换
        3. 去除空值行
        4. 按日期排序
        """
        df = df.copy()
        
        # 确保日期列为 datetime 类型
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 数值列类型转换
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pct_chg']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 去除关键列为空的行
        df = df.dropna(subset=['close', 'volume'])
        
        # 按日期升序排序
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        计算指标：
        - MA5, MA10, MA20: 移动平均线
        - Volume_Ratio: 量比（今日成交量 / 5日平均成交量）
        """
        df = df.copy()
        
        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5, min_periods=1).mean()
        df['ma10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['ma20'] = df['close'].rolling(window=20, min_periods=1).mean()
        
        # 量比：当日成交量 / 5日平均成交量
        avg_volume_5 = df['volume'].rolling(window=5, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / avg_volume_5.shift(1)
        df['volume_ratio'] = df['volume_ratio'].fillna(1.0)
        
        # 保留2位小数
        for col in ['ma5', 'ma10', 'ma20', 'volume_ratio']:
            if col in df.columns:
                df[col] = df[col].round(2)
        
        return df
    
    @staticmethod
    def random_sleep(min_seconds: float = 1.0, max_seconds: float = 3.0) -> None:
        """
        智能随机休眠（Jitter）
        
        防封禁策略：模拟人类行为的随机延迟
        在请求之间加入不规则的等待时间
        """
        sleep_time = random.uniform(min_seconds, max_seconds)
        logger.debug(f"随机休眠 {sleep_time:.2f} 秒...")
        time.sleep(sleep_time)


class DataFetcherManager:
    """
    数据源策略管理器
    
    职责：
    1. 管理多个数据源（按优先级排序）
    2. 自动故障切换（Failover）
    3. 提供统一的数据获取接口
    
    切换策略：
    - 优先使用高优先级数据源
    - 失败后自动切换到下一个
    - 所有数据源都失败时抛出异常
    """
    
    def __init__(self, fetchers: Optional[List[BaseFetcher]] = None):
        """
        初始化管理器
        
        Args:
            fetchers: 数据源列表（可选，默认按优先级自动创建）
        """
        self._fetchers: List[BaseFetcher] = []
        
        if fetchers:
            # 按优先级排序
            self._fetchers = sorted(fetchers, key=lambda f: f.priority)
        else:
            # 默认数据源将在首次使用时延迟加载
            self._init_default_fetchers()
    
    def _init_default_fetchers(self) -> None:
        """
        初始化默认数据源列表

        优先级动态调整逻辑：
        - 如果配置了 TUSHARE_TOKEN：Tushare 优先级提升为 0（最高）
        - 否则按默认优先级：
          0. EfinanceFetcher (Priority 0) - 最高优先级
          1. AkshareFetcher (Priority 1)
          2. PytdxFetcher (Priority 2) - 通达信
          2. TushareFetcher (Priority 2)
          3. BaostockFetcher (Priority 3)
          4. YfinanceFetcher (Priority 4)
        """
        from .efinance_fetcher import EfinanceFetcher
        from .akshare_fetcher import AkshareFetcher
        from .tushare_fetcher import TushareFetcher
        from .pytdx_fetcher import PytdxFetcher
        from .baostock_fetcher import BaostockFetcher
        from .yfinance_fetcher import YfinanceFetcher
        from src.config import get_config

        config = get_config()

        # 创建所有数据源实例（优先级在各 Fetcher 的 __init__ 中确定）
        efinance = EfinanceFetcher()
        akshare = AkshareFetcher()
        tushare = TushareFetcher()  # 会根据 Token 配置自动调整优先级
        pytdx = PytdxFetcher()      # 通达信数据源
        baostock = BaostockFetcher()
        yfinance = YfinanceFetcher()

        # 初始化数据源列表
        self._fetchers = [
            efinance,
            akshare,
            tushare,
            pytdx,
            baostock,
            yfinance,
        ]

        # 按优先级排序（Tushare 如果配置了 Token 且初始化成功，优先级为 0）
        self._fetchers.sort(key=lambda f: f.priority)

        # 构建优先级说明
        priority_info = ", ".join([f"{f.name}(P{f.priority})" for f in self._fetchers])
        logger.info(f"已初始化 {len(self._fetchers)} 个数据源（按优先级）: {priority_info}")
    
    def add_fetcher(self, fetcher: BaseFetcher) -> None:
        """添加数据源并重新排序"""
        self._fetchers.append(fetcher)
        self._fetchers.sort(key=lambda f: f.priority)
    
    def get_daily_data(
        self, 
        stock_code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        days: int = 30
    ) -> Tuple[pd.DataFrame, str]:
        """
        获取日线数据（自动切换数据源）
        
        故障切换策略：
        1. 从最高优先级数据源开始尝试
        2. 捕获异常后自动切换到下一个
        3. 记录每个数据源的失败原因
        4. 所有数据源失败后抛出详细异常
        
        Args:
            stock_code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
            days: 获取天数
            
        Returns:
            Tuple[DataFrame, str]: (数据, 成功的数据源名称)
            
        Raises:
            DataFetchError: 所有数据源都失败时抛出
        """
        errors = []
        
        for fetcher in self._fetchers:
            # === 核心修改：智能分流 (Start) ===
            # 如果是澳洲股票(.AX) 或 美股(纯字母)，且当前工具不是雅虎，直接跳过
            is_au_us = ('.AX' in stock_code) or (stock_code.isalpha()) or ('.US' in stock_code)
            
            if is_au_us and 'YfinanceFetcher' not in fetcher.name:
                continue
            # === 核心修改：智能分流 (End) ===

            try:
                logger.info(f"尝试使用 [{fetcher.name}] 获取 {stock_code}...")
                
                # 👇👇👇 这里是你刚才漏掉的关键代码，必须加回来！ 👇👇👇
                df = fetcher.get_daily_data(
                    stock_code=stock_code,
                    start_date=start_date,
                    end_date=end_date,
                    days=days
                )
                # 👆👆👆 没有这一段，程序就废了 👆👆👆

                if df is not None and not df.empty:
                    logger.info(f"[{fetcher.name}] 成功获取 {stock_code}")
                    return df, fetcher.name
                    
            except Exception as e:
                error_msg = f"[{fetcher.name}] 失败: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                # 继续尝试下一个数据源
                continue
        
        # 所有数据源都失败
        error_summary = f"所有数据源获取 {stock_code} 失败:\n" + "\n".join(errors)
        logger.error(error_summary)
        raise DataFetchError(error_summary)
    
    @property
    def available_fetchers(self) -> List[str]:
        """返回可用数据源名称列表"""
        return [f.name for f in self._fetchers]
    
    def prefetch_realtime_quotes(self, stock_codes: List[str]) -> int:
        """
        批量预取实时行情数据（在分析开始前调用）
        
        策略：
        1. 检查优先级中是否包含全量拉取数据源（efinance/akshare_em）
        2. 如果不包含，跳过预取（新浪/腾讯是单股票查询，无需预取）
        3. 如果自选股数量 >= 5 且使用全量数据源，则预取填充缓存
        
        这样做的好处：
        - 使用新浪/腾讯时：每只股票独立查询，无全量拉取问题
        - 使用 efinance/东财时：预取一次，后续缓存命中
        
        Args:
            stock_codes: 待分析的股票代码列表
            
        Returns:
            预取的股票数量（0 表示跳过预取）
        """
        from src.config import get_config
        
        config = get_config()
        
        # 如果实时行情被禁用，跳过预取
        if not config.enable_realtime_quote:
            logger.debug("[预取] 实时行情功能已禁用，跳过预取")
            return 0
        
        # 检查优先级中是否包含全量拉取数据源
        # 注意：新增全量接口（如 tushare_realtime）时需同步更新此列表
        # 全量接口特征：一次 API 调用拉取全市场 5000+ 股票数据
        priority = config.realtime_source_priority.lower()
        bulk_sources = ['efinance', 'akshare_em', 'tushare']  # 全量接口列表
        
        # 如果优先级中前两个都不是全量数据源，跳过预取
        # 因为新浪/腾讯是单股票查询，不需要预取
        priority_list = [s.strip() for s in priority.split(',')]
        first_bulk_source_index = None
        for i, source in enumerate(priority_list):
            if source in bulk_sources:
                first_bulk_source_index = i
                break
        
        # 如果没有全量数据源，或者全量数据源排在第 3 位之后，跳过预取
        if first_bulk_source_index is None or first_bulk_source_index >= 2:
            logger.info(f"[预取] 当前优先级使用轻量级数据源(sina/tencent)，无需预取")
            return 0
        
        # 如果股票数量少于 5 个，不进行批量预取（逐个查询更高效）
        if len(stock_codes) < 5:
            logger.info(f"[预取] 股票数量 {len(stock_codes)} < 5，跳过批量预取")
            return 0
        
        logger.info(f"[预取] 开始批量预取实时行情，共 {len(stock_codes)} 只股票...")
        
        # 尝试通过 efinance 或 akshare 预取
        # 只需要调用一次 get_realtime_quote，缓存机制会自动拉取全市场数据
        try:
            # 用第一只股票触发全量拉取
            first_code = stock_codes[0]
            quote = self.get_realtime_quote(first_code)
            
            if quote:
                logger.info(f"[预取] 批量预取完成，缓存已填充")
                return len(stock_codes)
            else:
                logger.warning(f"[预取] 批量预取失败，将使用逐个查询模式")
                return 0
                
        except Exception as e:
            logger.error(f"[预取] 批量预取异常: {e}")
            return 0
    
    def get_realtime_quote(self, stock_code: str):
        """
        获取实时行情数据（自动故障切换）
        
        故障切换策略（按配置的优先级）：
        1. 美股：使用 YfinanceFetcher.get_realtime_quote()
        2. EfinanceFetcher.get_realtime_quote()
        3. AkshareFetcher.get_realtime_quote(source="em")  - 东财
        4. AkshareFetcher.get_realtime_quote(source="sina") - 新浪
        5. AkshareFetcher.get_realtime_quote(source="tencent") - 腾讯
        6. 返回 None（降级兜底）
        
        Args:
            stock_code: 股票代码
            
        Returns:
            UnifiedRealtimeQuote 对象，所有数据源都失败则返回 None
        """
        from .realtime_types import get_realtime_circuit_breaker
        from .akshare_fetcher import _is_us_code
        from src.config import get_config
        
        config = get_config()
        
        # 如果实时行情功能被禁用，直接返回 None
        if not config.enable_realtime_quote:
            logger.debug(f"[实时行情] 功能已禁用，跳过 {stock_code}")
            return None
        
        # 美股单独处理，使用 YfinanceFetcher
        if _is_us_code(stock_code):
            for fetcher in self._fetchers:
                if fetcher.name == "YfinanceFetcher":
                    if hasattr(fetcher, 'get_realtime_quote'):
                        try:
                            quote = fetcher.get_realtime_quote(stock_code)
                            if quote is not None:
                                logger.info(f"[实时行情] 美股 {stock_code} 成功获取 (来源: yfinance)")
                                return quote
                        except Exception as e:
                            logger.warning(f"[实时行情] 美股 {stock_code} 获取失败: {e}")
                    break
            logger.warning(f"[实时行情] 美股 {stock_code} 无可用数据源")
            return None
        
        # 获取配置的数据源优先级
        source_priority = config.realtime_source_priority.split(',')
        
        errors = []
        # primary_quote holds the first successful result; we may supplement
        # missing fields (volume_ratio, turnover_rate, etc.) from later sources.
        primary_quote = None
        
        for source in source_priority:
            source = source.strip().lower()
            
            try:
                quote = None
                
                if source == "efinance":
                    # 尝试 EfinanceFetcher
                    for fetcher in self._fetchers:
                        if fetcher.name == "EfinanceFetcher":
                            if hasattr(fetcher, 'get_realtime_quote'):
                                quote = fetcher.get_realtime_quote(stock_code)
                            break
                
                elif source == "akshare_em":
                    # 尝试 AkshareFetcher 东财数据源
                    for fetcher in self._fetchers:
                        if fetcher.name == "AkshareFetcher":
                            if hasattr(fetcher, 'get_realtime_quote'):
                                quote = fetcher.get_realtime_quote(stock_code, source="em")
                            break
                
                elif source == "akshare_sina":
                    # 尝试 AkshareFetcher 新浪数据源
                    for fetcher in self._fetchers:
                        if fetcher.name == "AkshareFetcher":
                            if hasattr(fetcher, 'get_realtime_quote'):
                                quote = fetcher.get_realtime_quote(stock_code, source="sina")
                            break
                
                elif source in ("tencent", "akshare_qq"):
                    # 尝试 AkshareFetcher 腾讯数据源
                    for fetcher in self._fetchers:
                        if fetcher.name == "AkshareFetcher":
                            if hasattr(fetcher, 'get_realtime_quote'):
                                quote = fetcher.get_realtime_quote(stock_code, source="tencent")
                            break
                
                elif source == "tushare":
                    # 尝试 TushareFetcher（需要 Tushare Pro 积分）
                    for fetcher in self._fetchers:
                        if fetcher.name == "TushareFetcher":
                            if hasattr(fetcher, 'get_realtime_quote'):
                                quote = fetcher.get_realtime_quote(stock_code)
                            break
                
                if quote is not None and quote.has_basic_data():
                    if primary_quote is None:
                        # First successful source becomes primary
                        primary_quote = quote
                        logger.info(f"[实时行情] {stock_code} 成功获取 (来源: {source})")
                        # If all key supplementary fields are present, return early
                        if not self._quote_needs_supplement(primary_quote):
                            return primary_quote
                        # Otherwise, continue to try later sources for missing fields
                        logger.debug(f"[实时行情] {stock_code} 部分字段缺失，尝试从后续数据源补充")
                        supplement_attempts = 0
                    else:
                        # Supplement missing fields from this source (limit attempts)
                        supplement_attempts += 1
                        if supplement_attempts > 1:
                            logger.debug(f"[实时行情] {stock_code} 补充尝试已达上限，停止继续")
                            break
                        merged = self._merge_quote_fields(primary_quote, quote)
                        if merged:
                            logger.info(f"[实时行情] {stock_code} 从 {source} 补充了缺失字段: {merged}")
                        # Stop supplementing once all key fields are filled
                        if not self._quote_needs_supplement(primary_quote):
                            break
                    
            except Exception as e:
                error_msg = f"[{source}] 失败: {str(e)}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue
        
        # Return primary even if some fields are still missing
        if primary_quote is not None:
            return primary_quote

        # 所有数据源都失败，返回 None（降级兜底）
        if errors:
            logger.warning(f"[实时行情] {stock_code} 所有数据源均失败，降级处理: {'; '.join(errors)}")
        else:
            logger.warning(f"[实时行情] {stock_code} 无可用数据源")
        
        return None

    # Fields worth supplementing from secondary sources when the primary
    # source returns None for them. Ordered by importance.
    _SUPPLEMENT_FIELDS = [
        'volume_ratio', 'turnover_rate',
        'pe_ratio', 'pb_ratio', 'total_mv', 'circ_mv',
        'amplitude',
    ]

    @classmethod
    def _quote_needs_supplement(cls, quote) -> bool:
        """Check if any key supplementary field is still None."""
        for f in cls._SUPPLEMENT_FIELDS:
            if getattr(quote, f, None) is None:
                return True
        return False

    @classmethod
    def _merge_quote_fields(cls, primary, secondary) -> list:
        """
        Copy non-None fields from *secondary* into *primary* where
        *primary* has None. Returns list of field names that were filled.
        """
        filled = []
        for f in cls._SUPPLEMENT_FIELDS:
            if getattr(primary, f, None) is None:
                val = getattr(secondary, f, None)
                if val is not None:
                    setattr(primary, f, val)
                    filled.append(f)
        return filled

    def get_chip_distribution(self, stock_code: str):
        """
        获取筹码分布数据（带熔断和多数据源降级）

        策略：
        1. 检查配置开关
        2. 检查熔断器状态
        3. 依次尝试多个数据源：AkshareFetcher -> TushareFetcher -> EfinanceFetcher
        4. 所有数据源失败则返回 None（降级兜底）

        Args:
            stock_code: 股票代码

        Returns:
            ChipDistribution 对象，失败则返回 None
        """
        from .realtime_types import get_chip_circuit_breaker
        from src.config import get_config

        config = get_config()

        # 如果筹码分布功能被禁用，直接返回 None
        if not config.enable_chip_distribution:
            logger.debug(f"[筹码分布] 功能已禁用，跳过 {stock_code}")
            return None

        circuit_breaker = get_chip_circuit_breaker()

        # 定义筹码数据源优先级列表
        chip_sources = [
            ("AkshareFetcher", "akshare_chip"),
            ("TushareFetcher", "tushare_chip"),
            ("EfinanceFetcher", "efinance_chip"),
        ]

        for fetcher_name, source_key in chip_sources:
            # 检查熔断器状态
            if not circuit_breaker.is_available(source_key):
                logger.debug(f"[熔断] {fetcher_name} 筹码接口处于熔断状态，尝试下一个")
                continue

            try:
                for fetcher in self._fetchers:
                    if fetcher.name == fetcher_name:
                        if hasattr(fetcher, 'get_chip_distribution'):
                            chip = fetcher.get_chip_distribution(stock_code)
                            if chip is not None:
                                circuit_breaker.record_success(source_key)
                                logger.info(f"[筹码分布] {stock_code} 成功获取 (来源: {fetcher_name})")
                                return chip
                        break
            except Exception as e:
                logger.warning(f"[筹码分布] {fetcher_name} 获取 {stock_code} 失败: {e}")
                circuit_breaker.record_failure(source_key, str(e))
                continue

        logger.warning(f"[筹码分布] {stock_code} 所有数据源均失败")
        return None

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        """
        获取股票中文名称（自动切换数据源）
        
        尝试从多个数据源获取股票名称：
        1. 先从实时行情缓存中获取（如果有）
        2. 依次尝试各个数据源的 get_stock_name 方法
        3. 最后尝试让大模型通过搜索获取（需要外部调用）
        
        Args:
            stock_code: 股票代码
            
        Returns:
            股票中文名称，所有数据源都失败则返回 None
        """
        # 1. 先检查缓存
        if hasattr(self, '_stock_name_cache') and stock_code in self._stock_name_cache:
            return self._stock_name_cache[stock_code]
        
        # 初始化缓存
        if not hasattr(self, '_stock_name_cache'):
            self._stock_name_cache = {}
        
        # 2. 尝试从实时行情中获取（最快）
        quote = self.get_realtime_quote(stock_code)
        if quote and hasattr(quote, 'name') and quote.name:
            name = quote.name
            self._stock_name_cache[stock_code] = name
            logger.info(f"[股票名称] 从实时行情获取: {stock_code} -> {name}")
            return name
        
        # 3. 依次尝试各个数据源
        for fetcher in self._fetchers:
            if hasattr(fetcher, 'get_stock_name'):
                try:
                    name = fetcher.get_stock_name(stock_code)
                    if name:
                        self._stock_name_cache[stock_code] = name
                        logger.info(f"[股票名称] 从 {fetcher.name} 获取: {stock_code} -> {name}")
                        return name
                except Exception as e:
                    logger.debug(f"[股票名称] {fetcher.name} 获取失败: {e}")
                    continue
        
        # 4. 所有数据源都失败
        logger.warning(f"[股票名称] 所有数据源都无法获取 {stock_code} 的名称")
        return None

    def batch_get_stock_names(self, stock_codes: List[str]) -> Dict[str, str]:
        """
        批量获取股票中文名称
        
        先尝试从支持批量查询的数据源获取股票列表，
        然后再逐个查询缺失的股票名称。
        
        Args:
            stock_codes: 股票代码列表
            
        Returns:
            {股票代码: 股票名称} 字典
        """
        result = {}
        missing_codes = set(stock_codes)
        
        # 1. 先检查缓存
        if not hasattr(self, '_stock_name_cache'):
            self._stock_name_cache = {}
        
        for code in stock_codes:
            if code in self._stock_name_cache:
                result[code] = self._stock_name_cache[code]
                missing_codes.discard(code)
        
        if not missing_codes:
            return result
        
        # 2. 尝试批量获取股票列表
        for fetcher in self._fetchers:
            if hasattr(fetcher, 'get_stock_list') and missing_codes:
                try:
                    stock_list = fetcher.get_stock_list()
                    if stock_list is not None and not stock_list.empty:
                        for _, row in stock_list.iterrows():
                            code = row.get('code')
                            name = row.get('name')
                            if code and name:
                                self._stock_name_cache[code] = name
                                if code in missing_codes:
                                    result[code] = name
                                    missing_codes.discard(code)
                        
                        if not missing_codes:
                            break
                        
                        logger.info(f"[股票名称] 从 {fetcher.name} 批量获取完成，剩余 {len(missing_codes)} 个待查")
                except Exception as e:
                    logger.debug(f"[股票名称] {fetcher.name} 批量获取失败: {e}")
                    continue
        
        # 3. 逐个获取剩余的
        for code in list(missing_codes):
            name = self.get_stock_name(code)
            if name:
                result[code] = name
                missing_codes.discard(code)
        
        logger.info(f"[股票名称] 批量获取完成，成功 {len(result)}/{len(stock_codes)}")
        return result

    def get_main_indices(self) -> List[Dict[str, Any]]:
        """获取主要指数实时行情（自动切换数据源）"""
        for fetcher in self._fetchers:
            try:
                data = fetcher.get_main_indices()
                if data:
                    logger.info(f"[{fetcher.name}] 获取指数行情成功")
                    return data
            except Exception as e:
                logger.warning(f"[{fetcher.name}] 获取指数行情失败: {e}")
                continue
        return []

    def get_market_stats(self) -> Dict[str, Any]:
        """获取市场涨跌统计（自动切换数据源）"""
        for fetcher in self._fetchers:
            try:
                data = fetcher.get_market_stats()
                if data:
                    logger.info(f"[{fetcher.name}] 获取市场统计成功")
                    return data
            except Exception as e:
                logger.warning(f"[{fetcher.name}] 获取市场统计失败: {e}")
                continue
        return {}

    def get_sector_rankings(self, n: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """获取板块涨跌榜（自动切换数据源）"""
        for fetcher in self._fetchers:
            try:
                data = fetcher.get_sector_rankings(n)
                if data:
                    logger.info(f"[{fetcher.name}] 获取板块排行成功")
                    return data
            except Exception as e:
                logger.warning(f"[{fetcher.name}] 获取板块排行失败: {e}")
                continue
        return [], []
