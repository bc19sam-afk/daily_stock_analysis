# -*- coding: utf-8 -*-
"""
===================================
YfinanceFetcher - 完整全量修复版
===================================

数据来源：Yahoo Finance（通过 yfinance 库）
特点：国际数据源、可能有延迟或缺失
定位：当所有国内数据源都失败时的最后保障

关键修复 (2026-02-11):
1. 修正正则表达式，支持 .AX (澳洲)、.TW (台湾) 等多位后缀。
2. 修改 _fetch_raw_data 策略，强制获取近 3 个月数据，解决时区导致的 N/A 问题。
3. 保留了完整的实时行情 (fast_info + history 兜底) 逻辑。
"""

import logging
import re
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
import yfinance as yf

# 引入重试机制
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from .base import BaseFetcher, DataFetchError, STANDARD_COLUMNS
from .realtime_types import UnifiedRealtimeQuote, RealtimeSource

logger = logging.getLogger(__name__)


class YfinanceFetcher(BaseFetcher):
    """
    Yahoo Finance 数据源实现 (完整版)
    
    优先级：4（最低，作为兜底）
    数据来源：Yahoo Finance
    """
    
    name = "YfinanceFetcher"
    priority = int(os.getenv("YFINANCE_PRIORITY", "4"))
    
    def __init__(self):
        """初始化 YfinanceFetcher"""
        pass
    
    def _convert_stock_code(self, stock_code: str) -> str:
        """
        转换股票代码为 Yahoo Finance 格式
        
        Yahoo Finance 代码格式：
        - A股沪市：600519.SS
        - A股深市：000001.SZ
        - 港股：0700.HK
        - 澳股：BHP.AX (关键修复)
        - 美股：AAPL
        """
        code = stock_code.strip().upper()

        # === 修复点 1：修正正则，允许 .AX 等多字母后缀 ===
        # 原来的 (\.[A-Z])? 只能匹配一位后缀，现在改为 (\.[A-Z]+)?
        # 这样就能识别 .AX, .TW, .L 等后缀了
        if re.match(r'^[A-Z]{1,5}(\.[A-Z]+)?$', code):
            # logger.debug(f"识别为标准/国际代码: {code}")
            return code

        # 港股：hk前缀 -> .HK后缀
        if code.startswith('HK'):
            hk_code = code[2:].lstrip('0') or '0'  # 去除前导0
            hk_code = hk_code.zfill(4)  # 补齐到4位
            return f"{hk_code}.HK"

        # 已经包含后缀的情况 (SS, SZ, HK, AX, TW)
        if any(suffix in code for suffix in ['.SS', '.SZ', '.HK', '.AX', '.TW']):
            return code

        # 去除可能的 .SH 后缀 (兼容习惯)
        code = code.replace('.SH', '')

        # A股：根据代码前缀智能判断市场
        if code.startswith(('600', '601', '603', '688')):
            return f"{code}.SS"
        elif code.startswith(('000', '002', '300')):
            return f"{code}.SZ"
        
        # 默认返回原始代码，不做猜测，防止错误
        return code
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ConnectionError, TimeoutError, Exception)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _fetch_raw_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        从 Yahoo Finance 获取原始数据
        
        关键修改：
        不再使用 download(start=..., end=...)，因为时区问题容易返回空。
        改用 history(period='3mo')，强制拉取最近数据，确保有值。
        """
        # 转换代码格式
        yf_code = self._convert_stock_code(stock_code)
        
        logger.info(f"[Yfinance] 连接雅虎获取: {yf_code} (策略: 动态近期数据)")
        
        try:
            ticker = yf.Ticker(yf_code)
            
            # === 修复点 2：使用 history(period='3mo') ===
            # 这样可以避开 "今天澳洲是12号，美国服务器还是11号" 导致的 "No data found"
            df = ticker.history(period="3mo", auto_adjust=False)
            
            # 如果 3 个月没数据（可能是停牌或长假），尝试 6 个月
            if df.empty:
                logger.warning(f"[Yfinance] {yf_code} 近3月数据为空，尝试拉取6个月...")
                df = ticker.history(period="6mo", auto_adjust=False)
            
            if df.empty:
                raise DataFetchError(f"Yahoo Finance 未查询到 {stock_code} 的数据")
            
            return df
            
        except Exception as e:
            if isinstance(e, DataFetchError):
                raise
            raise DataFetchError(f"Yahoo Finance 获取数据失败: {e}") from e
    
    def _normalize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化 Yahoo Finance 数据
        处理列名、时区、缺失字段计算
        """
        df = df.copy()
        
        # 1. 处理 MultiIndex 列名（新版 yfinance 特性）
        # 例如: ('Close', 'AAPL') -> 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            # logger.debug(f"检测到 MultiIndex 列名，进行扁平化处理")
            df.columns = df.columns.get_level_values(0)
        
        # 2. 重置索引，将日期从索引变为列
        df = df.reset_index()
        
        # 3. 列名映射（yfinance 使用首字母大写 -> 标准小写）
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'close', # 如果存在Adj Close，优先映射为close
            'Volume': 'volume',
        }
        
        # 4. 执行重命名
        df = df.rename(columns=column_mapping)
        # 将剩余未映射的列名统一转小写
        df.columns = [c.lower() for c in df.columns]

        # 5. 特殊处理：如果有 adj close 且没有 close (或者 adj close 更好)，优先用 adj close
        if 'adj close' in df.columns and 'close' not in df.columns:
            df.rename(columns={'adj close': 'close'}, inplace=True)
        
        # 6. 处理时区问题 (这是导致 N/A 的隐形杀手)
        if 'date' in df.columns:
            # 强制去除时区信息，转为本地时间
            df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
            # 格式化为字符串 YYYY-MM-DD
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        
        # 7. 计算涨跌幅（yfinance 不直接提供）
        if 'pct_chg' not in df.columns and 'close' in df.columns:
            df['pct_chg'] = df['close'].pct_change() * 100
            df['pct_chg'] = df['pct_chg'].fillna(0).round(2)
        
        # 8. 计算成交额（yfinance 不提供，使用估算值）
        # 成交额 ≈ 成交量 * 收盘价
        if 'amount' not in df.columns and 'volume' in df.columns and 'close' in df.columns:
            df['amount'] = df['volume'] * df['close']
        else:
            df['amount'] = 0
        
        # 9. 添加股票代码列
        df['code'] = stock_code
        
        # 10. 只保留标准列，过滤杂质
        keep_cols = ['code'] + STANDARD_COLUMNS
        existing_cols = [col for col in keep_cols if col in df.columns]
        df = df[existing_cols]
        
        return df

    def get_main_indices(self) -> Optional[List[Dict[str, Any]]]:
        """
        获取主要指数行情 (Yahoo Finance)
        """
        # 映射关系：akshare代码 -> (yfinance代码, 名称)
        yf_mapping = {
            'sh000001': ('000001.SS', '上证指数'),
            'sz399001': ('399001.SZ', '深证成指'),
            'sz399006': ('399006.SZ', '创业板指'),
            'sh000688': ('000688.SS', '科创50'),
            'sh000016': ('000016.SS', '上证50'),
            'sh000300': ('000300.SS', '沪深300'),
            # 可以根据需要添加澳洲指数，如 ^AXJO (ASX 200)
            'au000001': ('^AXJO', '澳洲标普200'),
        }

        results = []
        try:
            for ak_code, (yf_code, name) in yf_mapping.items():
                try:
                    ticker = yf.Ticker(yf_code)
                    # 获取最近5天数据以计算涨跌，防止假期无数据
                    hist = ticker.history(period='5d')
                    if hist.empty:
                        continue

                    today = hist.iloc[-1]
                    prev = hist.iloc[-2] if len(hist) > 1 else today

                    price = float(today['Close'])
                    prev_close = float(prev['Close'])
                    change = price - prev_close
                    change_pct = (change / prev_close) * 100 if prev_close else 0

                    # 振幅
                    high = float(today['High'])
                    low = float(today['Low'])
                    amplitude = ((high - low) / prev_close * 100) if prev_close else 0

                    results.append({
                        'code': ak_code,
                        'name': name,
                        'current': price,
                        'change': change,
                        'change_pct': change_pct,
                        'open': float(today['Open']),
                        'high': high,
                        'low': low,
                        'prev_close': prev_close,
                        'volume': float(today['Volume']),
                        'amount': 0.0, # Yahoo Finance 可能不提供准确的指数成交额
                        'amplitude': amplitude
                    })
                    # logger.debug(f"[Yfinance] 获取指数 {name} 成功")

                except Exception as e:
                    logger.warning(f"[Yfinance] 获取指数 {name} 失败: {e}")
                    continue

            if results:
                logger.info(f"[Yfinance] 成功获取 {len(results)} 个指数行情")
                return results

        except Exception as e:
            logger.error(f"[Yfinance] 获取指数行情失败: {e}")

        return None

    def _is_us_stock(self, stock_code: str) -> bool:
        """
        判断代码是否为美股/国际股票
        
        修正后规则：
        - 1-5个大写字母，如 'AAPL'
        - 可能包含 '.' 及后缀，如 'BRK.B', 'BHP.AX'
        """
        code = stock_code.strip().upper()
        # 修复：允许 .AX, .US 等后缀
        return bool(re.match(r'^[A-Z]{1,5}(\.[A-Z]+)?$', code))

    def get_realtime_quote(self, stock_code: str) -> Optional[UnifiedRealtimeQuote]:
        """
        获取实时行情数据
        
        数据来源：yfinance Ticker.info 或 fast_info
        策略：
        1. 优先尝试 fast_info (速度快)
        2. 失败回退到 history (数据全)
        """
        
        # 即使不是美股，只要符合 yfinance 格式，都尝试查一下作为兜底
        yf_code = self._convert_stock_code(stock_code)
        
        # 如果是 A 股纯数字代码且没加后缀，可能不是我们想查的，跳过
        if yf_code.isdigit(): 
             return None

        try:
            # logger.debug(f"[Yfinance] 获取实时行情: {yf_code}")
            
            ticker = yf.Ticker(yf_code)
            
            # === 策略 1: 尝试获取 fast_info (更快，但字段较少) ===
            try:
                info = ticker.fast_info
                
                # 检查是否有价格，如果没有则抛出异常进入 fallback
                if info is None or info.last_price is None:
                    raise ValueError("fast_info returned None")
                
                price = info.last_price
                prev_close = info.previous_close
                open_price = info.open
                high = info.day_high
                low = info.day_low
                volume = info.last_volume
                
                # 市值
                market_cap = getattr(info, 'market_cap', None)
                
            except Exception:
                # === 策略 2: 回退到 history 方法获取最新数据 ===
                # logger.debug(f"[Yfinance] fast_info 失败，尝试 history 方法")
                hist = ticker.history(period='5d') # 获取5天，确保有数据
                if hist.empty:
                    # logger.warning(f"[Yfinance] 无法获取 {yf_code} 的数据")
                    return None
                
                today = hist.iloc[-1]
                prev = hist.iloc[-2] if len(hist) > 1 else today
                
                price = float(today['Close'])
                prev_close = float(prev['Close'])
                open_price = float(today['Open'])
                high = float(today['High'])
                low = float(today['Low'])
                volume = float(today['Volume'])
                market_cap = None
            
            # 计算涨跌幅
            change_amount = None
            change_pct = None
            if price is not None and prev_close is not None and prev_close > 0:
                change_amount = price - prev_close
                change_pct = (change_amount / prev_close) * 100
            
            # 计算振幅
            amplitude = None
            if high is not None and low is not None and prev_close is not None and prev_close > 0:
                amplitude = ((high - low) / prev_close) * 100
            
            # 获取股票名称 (尝试从 info 中获取)
            name = yf_code
            try:
                # 获取 info 字典比较慢，只在必要时调用
                full_info = ticker.info
                name = full_info.get('shortName', '') or full_info.get('longName', '') or yf_code
            except Exception:
                pass
            
            # 构造统一返回对象
            quote = UnifiedRealtimeQuote(
                code=stock_code,
                name=name,
                source=RealtimeSource.FALLBACK,
                price=price,
                change_pct=round(change_pct, 2) if change_pct is not None else None,
                change_amount=round(change_amount, 4) if change_amount is not None else None,
                volume=volume,
                amount=None,  # yfinance 不直接提供成交额
                volume_ratio=None,
                turnover_rate=None,
                amplitude=round(amplitude, 2) if amplitude is not None else None,
                open_price=open_price,
                high=high,
                low=low,
                pre_close=prev_close,
                pe_ratio=None,
                pb_ratio=None,
                total_mv=market_cap,
                circ_mv=None,
            )
            
            logger.info(f"[Yfinance] 获取实时行情成功: {yf_code} = {price}")
            return quote
            
        except Exception as e:
            # logger.warning(f"[Yfinance] 获取实时行情 {stock_code} 失败: {e}")
            return None


if __name__ == "__main__":
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    fetcher = YfinanceFetcher()
    
    test_codes = ['CBA.AX', 'BHP.AX', '600519', 'AAPL']
    
    print("="*50)
    print("开始测试 YfinanceFetcher (完整修复版)")
    print("="*50)

    for code in test_codes:
        print(f"\n正在测试: {code}")
        try:
            # 1. 测试日线数据
            df = fetcher.get_daily_data(code)
            print(f"✅ 日线数据获取成功: {len(df)} 条")
            if not df.empty:
                print(f"   最新日期: {df.iloc[-1]['date']}")
                print(f"   最新收盘: {df.iloc[-1]['close']}")
            
            # 2. 测试实时行情
            quote = fetcher.get_realtime_quote(code)
            if quote:
                print(f"✅ 实时行情获取成功: 价格 {quote.price}, 涨跌 {quote.change_pct}%")
            else:
                print(f"⚠️ 实时行情未获取到 (可能是收盘时间)")
                
        except Exception as e:
            print(f"❌ 测试失败: {e}")
