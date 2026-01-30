"""
API工具模块
提供统一的HTTP请求重试机制、超时处理和错误处理
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union

import requests
from requests.exceptions import (
    ConnectionError,
    HTTPError,
    RequestException,
    Timeout,
)

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_TIMEOUT = 30  # 秒
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY = 2  # 秒
DEFAULT_RETRY_BACKOFF = 2  # 指数退避因子


class APIRequestError(Exception):
    """API请求基础异常"""
    def __init__(self, message: str, status_code: Optional[int] = None, 
                 response: Optional[requests.Response] = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class RateLimitError(APIRequestError):
    """API速率限制异常"""
    def __init__(self, message: str, retry_after: Optional[int] = None,
                 status_code: int = 429, response: Optional[requests.Response] = None):
        super().__init__(message, status_code, response)
        self.retry_after = retry_after


class NetworkError(APIRequestError):
    """网络连接异常"""
    pass


class TimeoutError(APIRequestError):
    """请求超时异常"""
    pass


def calculate_backoff_delay(attempt: int, base_delay: float = DEFAULT_RETRY_DELAY,
                           backoff_factor: float = DEFAULT_RETRY_BACKOFF,
                           max_delay: float = 60.0) -> float:
    """
    计算指数退避延迟
    
    Args:
        attempt: 当前尝试次数（从0开始）
        base_delay: 基础延迟时间
        backoff_factor: 退避因子
        max_delay: 最大延迟时间
        
    Returns:
        延迟时间（秒）
    """
    delay = base_delay * (backoff_factor ** attempt)
    return min(delay, max_delay)


def api_request_with_retry(
    url: str,
    method: str = 'GET',
    max_retries: int = DEFAULT_MAX_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    retry_on_status: tuple = (429, 500, 502, 503, 504),
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, Any]] = None,
    json_data: Optional[Dict[str, Any]] = None,
    data: Optional[Any] = None,
    raise_for_status: bool = True,
    return_json: bool = True,
    session: Optional[requests.Session] = None,
) -> Union[Dict[str, Any], requests.Response]:
    """
    带重试机制的API请求函数
    
    Args:
        url: 请求URL
        method: HTTP方法 (GET, POST, PUT, DELETE等)
        max_retries: 最大重试次数
        timeout: 请求超时时间（秒）
        retry_delay: 基础重试延迟（秒）
        retry_backoff: 重试退避因子
        retry_on_status: 需要重试的HTTP状态码
        headers: 请求头
        params: URL参数
        json_data: JSON请求体
        data: 表单数据
        raise_for_status: 是否在HTTP错误时抛出异常
        return_json: 是否返回JSON解析后的数据
        session: 可选的requests.Session对象
        
    Returns:
        响应数据（JSON字典或Response对象）
        
    Raises:
        RateLimitError: 速率限制错误
        NetworkError: 网络连接错误
        TimeoutError: 请求超时
        APIRequestError: 其他API错误
    """
    request_func = session.request if session else requests.request
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            response = request_func(
                method=method.upper(),
                url=url,
                headers=headers,
                params=params,
                json=json_data,
                data=data,
                timeout=timeout,
            )
            
            # 检查是否需要重试（基于状态码）
            if response.status_code in retry_on_status:
                if attempt < max_retries:
                    # 处理429速率限制
                    if response.status_code == 429:
                        retry_after = response.headers.get('Retry-After')
                        if retry_after:
                            try:
                                delay = int(retry_after)
                            except ValueError:
                                delay = calculate_backoff_delay(attempt, retry_delay, retry_backoff)
                        else:
                            delay = calculate_backoff_delay(attempt, retry_delay, retry_backoff)
                        logger.warning(f"Rate limited (429), retrying after {delay}s (attempt {attempt + 1}/{max_retries})")
                    else:
                        delay = calculate_backoff_delay(attempt, retry_delay, retry_backoff)
                        logger.warning(f"Server error ({response.status_code}), retrying after {delay}s (attempt {attempt + 1}/{max_retries})")
                    
                    time.sleep(delay)
                    continue
                else:
                    # 重试次数用尽
                    if response.status_code == 429:
                        raise RateLimitError(
                            f"Rate limit exceeded after {max_retries} retries",
                            status_code=429,
                            response=response
                        )
                    else:
                        raise APIRequestError(
                            f"Server error ({response.status_code}) after {max_retries} retries",
                            status_code=response.status_code,
                            response=response
                        )
            
            # 检查HTTP错误
            if raise_for_status:
                response.raise_for_status()
            
            # 返回结果
            if return_json:
                try:
                    return response.json()
                except ValueError as e:
                    logger.warning(f"Failed to parse JSON response: {e}")
                    return response
            else:
                return response
                
        except Timeout as e:
            last_exception = e
            if attempt < max_retries:
                delay = calculate_backoff_delay(attempt, retry_delay, retry_backoff)
                logger.warning(f"Request timeout, retrying after {delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise TimeoutError(f"Request timed out after {max_retries} retries: {url}") from e
                
        except ConnectionError as e:
            last_exception = e
            if attempt < max_retries:
                delay = calculate_backoff_delay(attempt, retry_delay, retry_backoff)
                logger.warning(f"Connection error, retrying after {delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise NetworkError(f"Connection failed after {max_retries} retries: {url}") from e
                
        except HTTPError as e:
            # HTTP错误通常不重试（除非状态码在retry_on_status中）
            raise APIRequestError(
                f"HTTP error: {e}",
                status_code=e.response.status_code if e.response else None,
                response=e.response
            ) from e
            
        except RequestException as e:
            last_exception = e
            if attempt < max_retries:
                delay = calculate_backoff_delay(attempt, retry_delay, retry_backoff)
                logger.warning(f"Request error ({type(e).__name__}), retrying after {delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise APIRequestError(f"Request failed after {max_retries} retries: {e}") from e
    
    # 不应该到达这里，但以防万一
    raise APIRequestError(f"Request failed: {last_exception}")


def retry_decorator(
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    exceptions: tuple = (RequestException, ConnectionError, Timeout),
):
    """
    重试装饰器
    
    可用于装饰任意需要重试的函数
    
    Args:
        max_retries: 最大重试次数
        retry_delay: 基础重试延迟
        retry_backoff: 退避因子
        exceptions: 需要重试的异常类型
        
    Returns:
        装饰后的函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = calculate_backoff_delay(attempt, retry_delay, retry_backoff)
                        logger.warning(
                            f"{func.__name__} failed ({type(e).__name__}), "
                            f"retrying after {delay}s (attempt {attempt + 1}/{max_retries})"
                        )
                        time.sleep(delay)
                    else:
                        raise
            
            # 不应该到达这里
            raise last_exception
        
        return wrapper
    return decorator


def safe_get_nested(data: Dict, *keys, default: Any = None) -> Any:
    """
    安全地获取嵌套字典的值
    
    Args:
        data: 字典对象
        *keys: 键路径
        default: 默认值
        
    Returns:
        获取的值或默认值
        
    Example:
        >>> safe_get_nested({'a': {'b': {'c': 1}}}, 'a', 'b', 'c')
        1
        >>> safe_get_nested({'a': {}}, 'a', 'b', 'c', default=0)
        0
    """
    result = data
    for key in keys:
        if isinstance(result, dict):
            result = result.get(key)
            if result is None:
                return default
        elif isinstance(result, (list, tuple)) and isinstance(key, int):
            try:
                result = result[key]
            except (IndexError, TypeError):
                return default
        else:
            return default
    return result if result is not None else default


def safe_list_get(lst: list, index: int, default: Any = None) -> Any:
    """
    安全地获取列表元素
    
    Args:
        lst: 列表
        index: 索引
        default: 默认值
        
    Returns:
        元素值或默认值
    """
    try:
        return lst[index] if lst and len(lst) > index else default
    except (TypeError, IndexError):
        return default


# 便捷的GET和POST函数
def get_with_retry(url: str, **kwargs) -> Union[Dict[str, Any], requests.Response]:
    """带重试的GET请求"""
    return api_request_with_retry(url, method='GET', **kwargs)


def post_with_retry(url: str, **kwargs) -> Union[Dict[str, Any], requests.Response]:
    """带重试的POST请求"""
    return api_request_with_retry(url, method='POST', **kwargs)


if __name__ == '__main__':
    # 测试代码
    logging.basicConfig(level=logging.INFO)
    
    # 测试正常请求
    try:
        result = get_with_retry(
            'https://httpbin.org/get',
            params={'test': 'value'},
            max_retries=2
        )
        print(f"Success: {type(result)}")
    except APIRequestError as e:
        print(f"Error: {e}")
    
    # 测试安全获取
    data = {'a': {'b': [1, 2, 3]}}
    print(f"Nested value: {safe_get_nested(data, 'a', 'b', 0)}")
    print(f"Missing value: {safe_get_nested(data, 'a', 'c', default='not found')}")
