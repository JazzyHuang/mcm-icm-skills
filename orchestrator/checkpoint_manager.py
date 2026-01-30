"""
检查点管理器
管理MCM/ICM流水线的检查点保存和恢复
"""

import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        初始化检查点管理器
        
        Args:
            checkpoint_dir: 检查点保存目录
        """
        self.checkpoint_dir = Path(checkpoint_dir or 'output/checkpoints')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _sanitize_checkpoint_name(self, name: str) -> str:
        """
        清理检查点名称，防止路径注入攻击
        
        Args:
            name: 原始检查点名称
            
        Returns:
            清理后的安全名称
            
        Raises:
            ValueError: 如果名称包含非法字符
        """
        # 移除路径遍历字符和特殊字符，只保留字母、数字、下划线和连字符
        sanitized = re.sub(r'[^\w\-]', '_', name)
        
        # 检查是否包含路径遍历尝试
        if '..' in sanitized or sanitized.startswith('/') or sanitized.startswith('\\'):
            raise ValueError(f"Invalid checkpoint name: {name}")
        
        # 检查名称长度
        if len(sanitized) > 200:
            raise ValueError(f"Checkpoint name too long: {len(name)} characters (max 200)")
        
        # 确保名称不为空
        if not sanitized:
            raise ValueError("Checkpoint name cannot be empty")
        
        return sanitized
        
    def save(self, state: Dict[str, Any], checkpoint_name: str) -> Path:
        """
        保存检查点（使用原子操作确保数据完整性）
        
        Args:
            state: 状态字典
            checkpoint_name: 检查点名称
            
        Returns:
            检查点文件路径
            
        Raises:
            ValueError: 如果检查点名称无效
        """
        # 清理检查点名称，防止路径注入
        safe_name = self._sanitize_checkpoint_name(checkpoint_name)
        
        # 创建检查点数据
        checkpoint_data = {
            'name': safe_name,
            'timestamp': datetime.now().isoformat(),
            'state': state
        }
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{safe_name}_{timestamp}.json"
        filepath = self.checkpoint_dir / filename
        
        # 使用原子操作保存到文件（临时文件 + 重命名）
        try:
            # 创建临时文件在同一目录下（确保在同一文件系统，重命名才是原子的）
            fd, tmp_path = tempfile.mkstemp(suffix='.tmp', dir=self.checkpoint_dir)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)
                # 原子重命名
                shutil.move(tmp_path, filepath)
            except Exception:
                # 清理临时文件
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise
        except (OSError, IOError) as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
            
        # 更新最新检查点链接（也使用原子操作）
        latest_link = self.checkpoint_dir / f"{safe_name}_latest.json"
        try:
            fd, tmp_latest = tempfile.mkstemp(suffix='.tmp', dir=self.checkpoint_dir)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)
                shutil.move(tmp_latest, latest_link)
            except Exception:
                if os.path.exists(tmp_latest):
                    os.unlink(tmp_latest)
                raise
        except (OSError, IOError) as e:
            logger.warning(f"Failed to update latest checkpoint link: {e}")
            # 不抛出异常，主检查点已保存成功
        
        logger.info(f"Checkpoint saved: {filepath}")
        return filepath
        
    def load(self, checkpoint_name: str, specific_timestamp: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        加载检查点
        
        Args:
            checkpoint_name: 检查点名称
            specific_timestamp: 指定时间戳，默认加载最新
            
        Returns:
            状态字典，如果不存在返回None
            
        Raises:
            ValueError: 如果检查点名称无效
        """
        # 清理检查点名称，防止路径注入
        safe_name = self._sanitize_checkpoint_name(checkpoint_name)
        
        if specific_timestamp:
            # 也清理时间戳参数
            safe_timestamp = re.sub(r'[^\w\-]', '_', specific_timestamp)
            filename = f"{safe_name}_{safe_timestamp}.json"
            filepath = self.checkpoint_dir / filename
        else:
            filepath = self.checkpoint_dir / f"{safe_name}_latest.json"
        
        # 确保路径在检查点目录内（额外安全检查）
        try:
            filepath = filepath.resolve()
            checkpoint_dir_resolved = self.checkpoint_dir.resolve()
            if not str(filepath).startswith(str(checkpoint_dir_resolved)):
                logger.error(f"Path traversal attempt detected: {filepath}")
                return None
        except (OSError, ValueError) as e:
            logger.error(f"Path resolution error: {e}")
            return None
            
        if not filepath.exists():
            logger.warning(f"Checkpoint not found: {filepath}")
            return None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load checkpoint {filepath}: {e}")
            return None
            
        logger.info(f"Checkpoint loaded: {filepath}")
        return checkpoint_data.get('state')
        
    def list_checkpoints(self, checkpoint_name: Optional[str] = None) -> List[Dict]:
        """
        列出所有检查点
        
        Args:
            checkpoint_name: 筛选特定名称的检查点
            
        Returns:
            检查点信息列表
        """
        checkpoints = []
        
        for filepath in self.checkpoint_dir.glob('*.json'):
            if filepath.name.endswith('_latest.json'):
                continue
                
            if checkpoint_name and not filepath.name.startswith(checkpoint_name):
                continue
                
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    checkpoints.append({
                        'name': data.get('name'),
                        'timestamp': data.get('timestamp'),
                        'filepath': str(filepath)
                    })
            except Exception as e:
                logger.warning(f"Failed to read checkpoint {filepath}: {e}")
                
        # 按时间戳排序
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        return checkpoints
        
    def delete_checkpoint(self, checkpoint_name: str, specific_timestamp: Optional[str] = None) -> bool:
        """
        删除检查点
        
        Args:
            checkpoint_name: 检查点名称
            specific_timestamp: 指定时间戳
            
        Returns:
            是否删除成功
            
        Raises:
            ValueError: 如果检查点名称无效
        """
        # 清理检查点名称，防止路径注入
        safe_name = self._sanitize_checkpoint_name(checkpoint_name)
        
        if specific_timestamp:
            # 也清理时间戳参数
            safe_timestamp = re.sub(r'[^\w\-]', '_', specific_timestamp)
            filename = f"{safe_name}_{safe_timestamp}.json"
            filepath = self.checkpoint_dir / filename
            
            # 确保路径在检查点目录内
            try:
                filepath = filepath.resolve()
                if not str(filepath).startswith(str(self.checkpoint_dir.resolve())):
                    logger.error(f"Path traversal attempt detected")
                    return False
            except (OSError, ValueError):
                return False
            
            if filepath.exists():
                try:
                    filepath.unlink()
                    logger.info(f"Checkpoint deleted: {filepath}")
                    return True
                except OSError as e:
                    logger.error(f"Failed to delete checkpoint: {e}")
                    return False
        else:
            # 删除所有同名检查点
            deleted = False
            # 使用安全的glob模式
            pattern = f"{safe_name}*.json"
            for filepath in self.checkpoint_dir.glob(pattern):
                try:
                    # 确保路径在检查点目录内
                    filepath = filepath.resolve()
                    if str(filepath).startswith(str(self.checkpoint_dir.resolve())):
                        filepath.unlink()
                        deleted = True
                except (OSError, ValueError) as e:
                    logger.warning(f"Failed to delete {filepath}: {e}")
                
            if deleted:
                logger.info(f"All checkpoints for '{safe_name}' deleted")
            return deleted
            
        return False
        
    def cleanup_old_checkpoints(self, keep_latest: int = 5) -> int:
        """
        清理旧检查点
        
        Args:
            keep_latest: 每个名称保留的最新检查点数量
            
        Returns:
            删除的检查点数量
        """
        # 按名称分组
        checkpoints_by_name = {}
        
        for filepath in self.checkpoint_dir.glob('*.json'):
            if filepath.name.endswith('_latest.json'):
                continue
                
            # 提取检查点名称
            parts = filepath.stem.rsplit('_', 2)
            if len(parts) >= 3:
                name = parts[0]
            else:
                name = filepath.stem
                
            if name not in checkpoints_by_name:
                checkpoints_by_name[name] = []
            checkpoints_by_name[name].append(filepath)
            
        # 删除旧检查点
        deleted_count = 0
        for name, filepaths in checkpoints_by_name.items():
            # 按修改时间排序
            filepaths.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # 删除超出保留数量的检查点
            for filepath in filepaths[keep_latest:]:
                filepath.unlink()
                deleted_count += 1
                logger.debug(f"Deleted old checkpoint: {filepath}")
                
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old checkpoints")
            
        return deleted_count
        
    def get_latest_checkpoint_name(self) -> Optional[str]:
        """
        获取最新检查点的名称
        
        Returns:
            最新检查点名称
        """
        latest_checkpoints = list(self.checkpoint_dir.glob('*_latest.json'))
        
        if not latest_checkpoints:
            return None
            
        # 找到最新修改的
        latest = max(latest_checkpoints, key=lambda x: x.stat().st_mtime)
        return latest.stem.replace('_latest', '')
        
    def checkpoint_exists(self, checkpoint_name: str) -> bool:
        """
        检查检查点是否存在
        
        Args:
            checkpoint_name: 检查点名称
            
        Returns:
            是否存在
            
        Raises:
            ValueError: 如果检查点名称无效
        """
        # 清理检查点名称，防止路径注入
        safe_name = self._sanitize_checkpoint_name(checkpoint_name)
        latest = self.checkpoint_dir / f"{safe_name}_latest.json"
        return latest.exists()
