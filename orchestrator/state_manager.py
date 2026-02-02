"""
状态管理器
管理MCM/ICM流水线的全局状态
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# 章节最小字数要求（O奖标准）
SECTION_MIN_WORDS = {
    'introduction': 800,
    'problem_analysis': 600,
    'assumptions': 400,
    'model_design': 1500,
    'model_implementation': 1000,
    'results_analysis': 1200,
    'sensitivity_analysis': 800,
    'strengths_weaknesses': 600,
    'conclusions': 400,
    'executive_summary': 300,
    'memo': 500,
}

# 章节目标字数（推荐）
SECTION_TARGET_WORDS = {
    'introduction': 1000,
    'problem_analysis': 800,
    'assumptions': 500,
    'model_design': 2000,
    'model_implementation': 1200,
    'results_analysis': 1500,
    'sensitivity_analysis': 1000,
    'strengths_weaknesses': 700,
    'conclusions': 500,
    'executive_summary': 400,
    'memo': 600,
}


class StateManager:
    """全局状态管理器"""
    
    def __init__(self):
        """初始化状态管理器"""
        self.state = {}
        self.history = []
        
    def initialize(self, problem_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        初始化状态
        
        Args:
            problem_input: 问题输入
            
        Returns:
            初始化后的状态
        """
        self.state = {
            # 基本信息
            'problem_input': problem_input,
            'problem_text': problem_input.get('problem_text', ''),
            'problem_type': problem_input.get('problem_type', 'A'),
            'team_control_number': problem_input.get('team_control_number', 'XXXXX'),
            'data_files': problem_input.get('data_files', []),
            
            # 执行状态
            'current_phase': 0,
            'completed_phases': [],
            'started_at': datetime.now().isoformat(),
            
            # 中间结果
            'parsed_problem': None,
            'collected_data': None,
            'literature': None,
            'assumptions': None,
            'variables': None,
            'constraints': None,
            'selected_model': None,
            'model_code': None,
            'solution': None,
            'sensitivity_results': None,
            'validation_results': None,
            'sections': {},
            'abstract': None,
            'figures': [],
            'tables': [],
            'citations': [],
            'latex_document': None,
            'final_pdf': None,
            
            # 质量指标
            'quality_scores': {},
            'grammar_score': None,
            'hallucination_check': None,
            
            # 错误和警告
            'errors': [],
            'warnings': [],
            'fallback_used': [],
            'skipped_skills': [],
        }
        
        logger.info(f"State initialized for problem type: {self.state['problem_type']}")
        return self.state
        
    def update(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        更新状态
        
        Args:
            updates: 要更新的字段
            
        Returns:
            更新后的状态
        """
        # 记录历史
        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'updates': list(updates.keys())
        })
        
        # 深度合并更新
        for key, value in updates.items():
            if key in self.state and isinstance(self.state[key], dict) and isinstance(value, dict):
                self.state[key].update(value)
            elif key in self.state and isinstance(self.state[key], list) and isinstance(value, list):
                self.state[key].extend(value)
            else:
                self.state[key] = value
                
        self.state['last_updated'] = datetime.now().isoformat()
        
        return self.state
        
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取状态值
        
        Args:
            key: 键名
            default: 默认值
            
        Returns:
            状态值
        """
        return self.state.get(key, default)
        
    def set(self, key: str, value: Any) -> None:
        """
        设置状态值
        
        Args:
            key: 键名
            value: 值
        """
        self.update({key: value})
        
    def get_state(self) -> Dict[str, Any]:
        """获取完整状态"""
        return self.state.copy()
        
    def get_final_output(self, state: Optional[Dict] = None) -> Dict[str, Any]:
        """
        获取最终输出
        
        Args:
            state: 状态字典，默认使用内部状态
            
        Returns:
            最终输出信息
        """
        s = state or self.state
        
        return {
            'paper': {
                'latex_file': s.get('latex_document'),
                'pdf_file': s.get('final_pdf'),
                'team_control_number': s.get('team_control_number'),
                'problem_type': s.get('problem_type'),
            },
            'sections': s.get('sections', {}),
            'abstract': s.get('abstract'),
            'figures': s.get('figures', []),
            'tables': s.get('tables', []),
            'citations': s.get('citations', []),
            'quality': {
                'scores': s.get('quality_scores', {}),
                'grammar_score': s.get('grammar_score'),
                'hallucination_check': s.get('hallucination_check'),
            },
            'metadata': {
                'started_at': s.get('started_at'),
                'completed_at': datetime.now().isoformat(),
                'completed_phases': s.get('completed_phases', []),
                'warnings': s.get('warnings', []),
                'fallback_used': s.get('fallback_used', []),
            }
        }
        
    def add_error(self, error: str, phase: Optional[int] = None) -> None:
        """添加错误记录"""
        self.state['errors'].append({
            'message': error,
            'phase': phase or self.state.get('current_phase'),
            'timestamp': datetime.now().isoformat()
        })
        
    def add_warning(self, warning: str, phase: Optional[int] = None) -> None:
        """添加警告记录"""
        self.state['warnings'].append({
            'message': warning,
            'phase': phase or self.state.get('current_phase'),
            'timestamp': datetime.now().isoformat()
        })
        
    def add_figure(self, figure_path: str, caption: str, label: str) -> None:
        """添加图表记录"""
        self.state['figures'].append({
            'path': figure_path,
            'caption': caption,
            'label': label,
            'index': len(self.state['figures']) + 1
        })
        
    def add_table(self, table_data: Dict, caption: str, label: str) -> None:
        """添加表格记录"""
        self.state['tables'].append({
            'data': table_data,
            'caption': caption,
            'label': label,
            'index': len(self.state['tables']) + 1
        })
        
    def add_citation(self, citation: Dict) -> None:
        """添加引用记录"""
        self.state['citations'].append(citation)
        
    def set_section(self, section_name: str, content: str) -> Dict[str, Any]:
        """
        设置章节内容并验证长度
        
        Args:
            section_name: 章节名称
            content: 章节内容
            
        Returns:
            包含验证结果的字典
        """
        word_count = len(content.split())
        min_required = SECTION_MIN_WORDS.get(section_name, 300)
        target_words = SECTION_TARGET_WORDS.get(section_name, 500)
        
        validation = {
            'word_count': word_count,
            'min_required': min_required,
            'target_words': target_words,
            'meets_minimum': word_count >= min_required,
            'meets_target': word_count >= target_words,
            'deficit': max(0, min_required - word_count),
            'target_deficit': max(0, target_words - word_count)
        }
        
        self.state['sections'][section_name] = {
            'content': content,
            'validation': validation,
            'updated_at': datetime.now().isoformat()
        }
        
        # 如果不满足最小要求，添加到待扩展列表
        if not validation['meets_minimum']:
            if 'sections_needing_expansion' not in self.state:
                self.state['sections_needing_expansion'] = []
            if section_name not in self.state['sections_needing_expansion']:
                self.state['sections_needing_expansion'].append(section_name)
                logger.warning(
                    f"Section '{section_name}' needs expansion: "
                    f"{word_count}/{min_required} words (deficit: {validation['deficit']})"
                )
        else:
            # 如果满足要求，从待扩展列表中移除
            if 'sections_needing_expansion' in self.state:
                if section_name in self.state['sections_needing_expansion']:
                    self.state['sections_needing_expansion'].remove(section_name)
        
        return validation
    
    def get_sections_needing_expansion(self) -> List[Dict[str, Any]]:
        """
        获取需要扩展的章节列表
        
        Returns:
            需要扩展的章节信息列表
        """
        result = []
        sections_to_expand = self.state.get('sections_needing_expansion', [])
        
        for section_name in sections_to_expand:
            section_data = self.state['sections'].get(section_name, {})
            validation = section_data.get('validation', {})
            result.append({
                'section_name': section_name,
                'current_word_count': validation.get('word_count', 0),
                'min_required': validation.get('min_required', 0),
                'target_words': validation.get('target_words', 0),
                'deficit': validation.get('deficit', 0)
            })
        
        return result
    
    def validate_all_sections(self) -> Dict[str, Any]:
        """
        验证所有章节的内容长度
        
        Returns:
            验证报告
        """
        report = {
            'total_sections': len(self.state.get('sections', {})),
            'sections_meeting_minimum': 0,
            'sections_meeting_target': 0,
            'total_word_count': 0,
            'sections_needing_expansion': [],
            'section_details': {}
        }
        
        for section_name, section_data in self.state.get('sections', {}).items():
            content = section_data.get('content', '')
            word_count = len(content.split())
            min_required = SECTION_MIN_WORDS.get(section_name, 300)
            target_words = SECTION_TARGET_WORDS.get(section_name, 500)
            
            meets_minimum = word_count >= min_required
            meets_target = word_count >= target_words
            
            report['total_word_count'] += word_count
            if meets_minimum:
                report['sections_meeting_minimum'] += 1
            else:
                report['sections_needing_expansion'].append(section_name)
            if meets_target:
                report['sections_meeting_target'] += 1
            
            report['section_details'][section_name] = {
                'word_count': word_count,
                'min_required': min_required,
                'target_words': target_words,
                'meets_minimum': meets_minimum,
                'meets_target': meets_target,
                'deficit': max(0, min_required - word_count)
            }
        
        # 计算覆盖率
        if report['total_sections'] > 0:
            report['minimum_coverage'] = report['sections_meeting_minimum'] / report['total_sections']
            report['target_coverage'] = report['sections_meeting_target'] / report['total_sections']
        else:
            report['minimum_coverage'] = 0
            report['target_coverage'] = 0
        
        # 更新state中的扩展列表
        self.state['sections_needing_expansion'] = report['sections_needing_expansion']
        
        return report
    
    def mark_section_expanded(self, section_name: str) -> None:
        """标记章节已扩展"""
        if section_name in self.state.get('sections', {}):
            self.state['sections'][section_name]['expanded'] = True
            self.state['sections'][section_name]['expanded_at'] = datetime.now().isoformat()
        
    def set_quality_score(self, dimension: str, score: float) -> None:
        """设置质量分数"""
        self.state['quality_scores'][dimension] = {
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        
    def to_json(self) -> str:
        """导出为JSON字符串"""
        return json.dumps(self.state, indent=2, ensure_ascii=False, default=str)
        
    def from_json(self, json_str: str) -> Dict[str, Any]:
        """从JSON字符串导入"""
        self.state = json.loads(json_str)
        return self.state
        
    def save_to_file(self, filepath: Path) -> None:
        """保存状态到文件"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
            
        logger.info(f"State saved to {filepath}")
        
    def load_from_file(self, filepath: Path) -> Dict[str, Any]:
        """从文件加载状态"""
        filepath = Path(filepath)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            self.state = json.load(f)
            
        logger.info(f"State loaded from {filepath}")
        return self.state
        
    def get_history(self) -> List[Dict]:
        """获取更新历史"""
        return self.history.copy()
        
    def clear(self) -> None:
        """清空状态"""
        self.state = {}
        self.history = []
