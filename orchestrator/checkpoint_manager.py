"""
检查点管理器
管理MCM/ICM流水线的检查点保存和恢复
"""

import json
import logging
import shutil
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
        
    def save(self, state: Dict[str, Any], checkpoint_name: str) -> Path:
        """
        保存检查点
        
        Args:
            state: 状态字典
            checkpoint_name: 检查点名称
            
        Returns:
            检查点文件路径
        """
        # 创建检查点数据
        checkpoint_data = {
            'name': checkpoint_name,
            'timestamp': datetime.now().isoformat(),
            'state': state
        }
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{checkpoint_name}_{timestamp}.json"
        filepath = self.checkpoint_dir / filename
        
        # 保存到文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)
            
        # 更新最新检查点链接
        latest_link = self.checkpoint_dir / f"{checkpoint_name}_latest.json"
        if latest_link.exists():
            latest_link.unlink()
        shutil.copy(filepath, latest_link)
        
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
        """
        if specific_timestamp:
            filename = f"{checkpoint_name}_{specific_timestamp}.json"
            filepath = self.checkpoint_dir / filename
        else:
            filepath = self.checkpoint_dir / f"{checkpoint_name}_latest.json"
            
        if not filepath.exists():
            logger.warning(f"Checkpoint not found: {filepath}")
            return None
            
        with open(filepath, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)
            
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
        """
        if specific_timestamp:
            filename = f"{checkpoint_name}_{specific_timestamp}.json"
            filepath = self.checkpoint_dir / filename
            
            if filepath.exists():
                filepath.unlink()
                logger.info(f"Checkpoint deleted: {filepath}")
                return True
        else:
            # 删除所有同名检查点
            deleted = False
            for filepath in self.checkpoint_dir.glob(f"{checkpoint_name}*.json"):
                filepath.unlink()
                deleted = True
                
            if deleted:
                logger.info(f"All checkpoints for '{checkpoint_name}' deleted")
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
        """
        latest = self.checkpoint_dir / f"{checkpoint_name}_latest.json"
        return latest.exists()
