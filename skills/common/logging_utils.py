"""
日志工具模块
提供统一的日志配置和上下文日志功能
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import json


# 日志级别标准
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,      # 详细调试信息
    'INFO': logging.INFO,        # 正常操作信息
    'WARNING': logging.WARNING,  # 警告（可恢复的问题）
    'ERROR': logging.ERROR,      # 错误（操作失败）
    'CRITICAL': logging.CRITICAL # 严重错误（系统级问题）
}

# 建议的日志级别使用场景
LOG_LEVEL_GUIDELINES = {
    'DEBUG': [
        '技能执行开始/结束',
        '中间计算结果',
        '详细的状态变化'
    ],
    'INFO': [
        '阶段开始/完成',
        '重要里程碑',
        '配置加载成功'
    ],
    'WARNING': [
        '使用降级方案',
        '性能下降',
        '非必需配置缺失'
    ],
    'ERROR': [
        '质量门禁失败',
        'API调用失败',
        '文件操作失败'
    ],
    'CRITICAL': [
        '系统无法继续',
        '关键依赖缺失',
        '数据损坏'
    ]
}


class ContextLogger:
    """带上下文信息的日志器"""
    
    def __init__(self, name: str, context: Optional[Dict[str, Any]] = None):
        """
        初始化上下文日志器
        
        Args:
            name: 日志器名称
            context: 默认上下文信息
        """
        self.logger = logging.getLogger(name)
        self.context = context or {}
        self._execution_id: Optional[str] = None
        self._phase: Optional[int] = None
        self._skill: Optional[str] = None
    
    def set_execution_context(
        self,
        execution_id: Optional[str] = None,
        phase: Optional[int] = None,
        skill: Optional[str] = None
    ) -> None:
        """
        设置执行上下文
        
        Args:
            execution_id: 执行ID
            phase: 当前阶段
            skill: 当前技能
        """
        if execution_id is not None:
            self._execution_id = execution_id
        if phase is not None:
            self._phase = phase
        if skill is not None:
            self._skill = skill
    
    def _format_message(self, message: str, extra: Optional[Dict] = None) -> str:
        """格式化消息，添加上下文"""
        context_parts = []
        
        if self._execution_id:
            context_parts.append(f"exec={self._execution_id[:8]}")
        if self._phase is not None:
            context_parts.append(f"phase={self._phase}")
        if self._skill:
            context_parts.append(f"skill={self._skill}")
        
        if extra:
            for k, v in extra.items():
                if v is not None:
                    context_parts.append(f"{k}={v}")
        
        context_str = " ".join(context_parts)
        if context_str:
            return f"[{context_str}] {message}"
        return message
    
    def debug(self, message: str, **extra) -> None:
        """记录DEBUG级别日志"""
        self.logger.debug(self._format_message(message, extra))
    
    def info(self, message: str, **extra) -> None:
        """记录INFO级别日志"""
        self.logger.info(self._format_message(message, extra))
    
    def warning(self, message: str, **extra) -> None:
        """记录WARNING级别日志"""
        self.logger.warning(self._format_message(message, extra))
    
    def error(self, message: str, **extra) -> None:
        """记录ERROR级别日志"""
        self.logger.error(self._format_message(message, extra))
    
    def critical(self, message: str, **extra) -> None:
        """记录CRITICAL级别日志"""
        self.logger.critical(self._format_message(message, extra))
    
    def exception(self, message: str, **extra) -> None:
        """记录异常（包含堆栈跟踪）"""
        self.logger.exception(self._format_message(message, extra))


def setup_logging(
    level: str = 'INFO',
    log_file: Optional[str] = None,
    format_style: str = 'standard',
    colorize: bool = True
) -> None:
    """
    配置全局日志
    
    Args:
        level: 日志级别
        log_file: 日志文件路径（可选）
        format_style: 格式风格 ('standard', 'detailed', 'json')
        colorize: 是否使用彩色输出
    """
    # 格式定义
    formats = {
        'standard': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'detailed': '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        'json': None  # JSON格式特殊处理
    }
    
    log_format = formats.get(format_style, formats['standard'])
    
    # 配置根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
    
    # 清除现有处理器
    root_logger.handlers = []
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
    
    if format_style == 'json':
        console_handler.setFormatter(JsonFormatter())
    else:
        if colorize:
            try:
                from colorlog import ColoredFormatter
                colored_format = '%(log_color)s' + log_format
                formatter = ColoredFormatter(
                    colored_format,
                    log_colors={
                        'DEBUG': 'cyan',
                        'INFO': 'green',
                        'WARNING': 'yellow',
                        'ERROR': 'red',
                        'CRITICAL': 'red,bg_white',
                    }
                )
                console_handler.setFormatter(formatter)
            except ImportError:
                console_handler.setFormatter(logging.Formatter(log_format))
        else:
            console_handler.setFormatter(logging.Formatter(log_format))
    
    root_logger.addHandler(console_handler)
    
    # 文件处理器（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(LOG_LEVELS.get(level.upper(), logging.INFO))
        file_handler.setFormatter(logging.Formatter(formats['detailed']))
        root_logger.addHandler(file_handler)


class JsonFormatter(logging.Formatter):
    """JSON格式的日志格式化器"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


def get_logger(name: str, context: Optional[Dict[str, Any]] = None) -> ContextLogger:
    """
    获取带上下文的日志器
    
    Args:
        name: 日志器名称（通常使用__name__）
        context: 默认上下文
        
    Returns:
        ContextLogger实例
    """
    return ContextLogger(name, context)


# 便捷函数
def log_phase_start(logger: ContextLogger, phase: int, description: str = '') -> None:
    """记录阶段开始"""
    logger.set_execution_context(phase=phase)
    logger.info(f"Phase {phase} started: {description}")


def log_phase_end(logger: ContextLogger, phase: int, success: bool = True) -> None:
    """记录阶段结束"""
    status = "completed" if success else "failed"
    logger.info(f"Phase {phase} {status}")


def log_skill_execution(
    logger: ContextLogger,
    skill: str,
    action: str = 'start',
    duration_ms: Optional[int] = None
) -> None:
    """记录技能执行"""
    logger.set_execution_context(skill=skill)
    if action == 'start':
        logger.debug(f"Skill '{skill}' execution started")
    elif action == 'end':
        msg = f"Skill '{skill}' execution completed"
        if duration_ms is not None:
            msg += f" ({duration_ms}ms)"
        logger.debug(msg)


if __name__ == '__main__':
    # 测试代码
    setup_logging(level='DEBUG', colorize=True)
    
    logger = get_logger(__name__)
    logger.set_execution_context(execution_id='test-123', phase=1)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    log_phase_start(logger, 2, "Data collection")
    log_skill_execution(logger, 'data-collector', 'start')
    log_skill_execution(logger, 'data-collector', 'end', duration_ms=1234)
    log_phase_end(logger, 2, success=True)
