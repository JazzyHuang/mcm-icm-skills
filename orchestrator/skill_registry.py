"""
技能注册表模块
负责扫描、注册、加载和管理所有技能
"""

import logging
import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .base_skill import (
    BaseSkill,
    SkillMetadata,
    ExecutionMode,
    create_skill
)
from .llm_adapter import LLMAdapter

logger = logging.getLogger(__name__)

# 阶段名称映射
PHASE_NAMES = {
    1: "phase1-input",
    2: "phase2-analysis",
    3: "phase3-modeling",
    4: "phase4-validation",
    5: "phase5-writing",
    6: "phase6-visualization",
    7: "phase7-integration",
    8: "phase8-quality",
    9: "phase9-optimization",
    10: "phase10-submission"
}

# 阶段技能映射 (与编排器保持一致 - 增强版 v2)
PHASE_SKILLS = {
    1: ['problem-parser', 'problem-type-classifier', 'problem-reference-extractor',
        'data-collector', 'deep-reference-searcher', 'literature-searcher',
        'ai-deep-search-guide',  # 新增: 确保深度搜索和引用多样性
        'citation-validator', 'citation-diversity-validator'],
    2: ['problem-decomposer', 'sub-problem-analyzer', 'assumption-generator',
        'variable-definer', 'constraint-identifier'],
    3: [
        # 基础建模流程
        'model-selector', 'model-justification-generator', 'hybrid-model-designer',
        'model-builder', 'model-solver', 'code-verifier',
        # 高级算法 (新增: 前沿创新方法)
        'physics-informed-nn',      # PINN物理信息神经网络
        'neural-operators',         # FNO/DeepONet神经算子
        'transformer-forecasting',  # Transformer时间序列预测
        'reinforcement-learning',   # 强化学习
        'kan-networks',             # KAN网络 (2025 ICLR前沿方法)
        'causal-inference'          # 因果推断
    ],
    4: ['sensitivity-analyzer', 'uncertainty-quantifier', 'model-validator',
        'error-analyzer', 'limitation-analyzer', 'strengths-weaknesses', 
        'ethical-analyzer', 'model-explainer'],
    5: ['section-writer', 'section-iterative-optimizer',  # 新增: 章节迭代优化
        'fact-checker', 'abstract-first-impression',
        'abstract-generator', 'abstract-iterative-optimizer', 'memo-letter-writer'],
    6: ['chart-generator', 'figure-narrative-generator', 'publication-scaler',
        'table-formatter', 'figure-validator',
        'infographic-generator'],  # 新增: 信息图生成器
    7: ['latex-compiler', 'compilation-error-handler', 'citation-manager',
        'format-checker', 'anonymization-checker'],
    8: ['quality-reviewer', 'hallucination-detector', 'grammar-checker',
        'chinglish-detector', 'consistency-checker', 'global-consistency-checker'],
    9: ['final-polisher', 'academic-english-optimizer', 'submission-preparer'],
    10: ['pre-submission-validator', 'submission-checklist']
}


class SkillRegistry:
    """
    技能注册表

    负责:
    - 扫描技能目录
    - 加载技能元数据
    - 懒加载技能实例
    - 管理技能依赖关系
    """

    def __init__(
        self,
        skills_dir: Path,
        llm_adapter: Optional[LLMAdapter] = None
    ):
        """
        初始化技能注册表

        Args:
            skills_dir: 技能根目录
            llm_adapter: LLM适配器 (可选)
        """
        self.skills_dir = Path(skills_dir)
        self.llm = llm_adapter

        # 存储结构
        self._skills: Dict[str, SkillMetadata] = {}       # name -> metadata
        self._instances: Dict[str, BaseSkill] = {}       # name -> instance
        self._by_phase: Dict[int, List[str]] = {}        # phase -> [skill_names]

        # 扫描并注册所有技能
        self._scan_and_register()

        logger.info(f"SkillRegistry initialized with {len(self._skills)} skills")

    def _scan_and_register(self):
        """扫描所有技能目录并注册"""
        registered_count = 0
        failed_count = 0

        # 遍历每个阶段目录
        for phase_dir in sorted(self.skills_dir.iterdir()):
            if not phase_dir.is_dir() or not phase_dir.name.startswith("phase"):
                continue

            # 提取阶段号
            try:
                phase_num = int(phase_dir.name.split("-")[0][5:])
            except (ValueError, IndexError):
                continue

            # 遍历技能子目录
            for skill_dir in phase_dir.iterdir():
                if not skill_dir.is_dir():
                    continue

                # 尝试加载 skill.yaml
                skill_yaml = skill_dir / "skill.yaml"
                if skill_yaml.exists():
                    try:
                        metadata = self._load_metadata(skill_yaml, phase_num)
                        self._register_metadata(metadata)
                        registered_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load {skill_yaml}: {e}")
                        failed_count += 1
                else:
                    # 没有 skill.yaml，尝试从 SKILL.md 推断
                    skill_md = skill_dir / "SKILL.md"
                    if skill_md.exists():
                        try:
                            metadata = self._infer_metadata(skill_dir, phase_num)
                            self._register_metadata(metadata)
                            registered_count += 1
                        except Exception as e:
                            logger.debug(f"Could not infer metadata for {skill_dir}: {e}")
                            failed_count += 1

        logger.info(f"Registered {registered_count} skills, {failed_count} failed")

    def _load_metadata(self, yaml_path: Path, phase: int) -> SkillMetadata:
        """从 YAML 文件加载元数据"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # 提取技能名 (从目录名)
        skill_name = yaml_path.parent.name

        # 补充缺失字段
        data.setdefault('name', skill_name)
        data.setdefault('phase', phase)
        data.setdefault('execution_mode', 'llm')
        data.setdefault('depends_on', [])
        data.setdefault('outputs', [])
        data.setdefault('timeout', 300)
        data.setdefault('description', '')

        return SkillMetadata(**data)

    def _infer_metadata(self, skill_dir: Path, phase: int) -> SkillMetadata:
        """从 SKILL.md 推断元数据"""
        skill_name = skill_dir.name

        # 默认元数据
        return SkillMetadata(
            name=skill_name,
            phase=phase,
            execution_mode="llm",
            depends_on=[],
            outputs=[],
            description=f"Inferred from {skill_dir.name}"
        )

    def _register_metadata(self, metadata: SkillMetadata):
        """注册技能元数据"""
        self._skills[metadata.name] = metadata

        # 按阶段索引
        if metadata.phase not in self._by_phase:
            self._by_phase[metadata.phase] = []
        if metadata.name not in self._by_phase[metadata.phase]:
            self._by_phase[metadata.phase].append(metadata.name)

        logger.debug(f"Registered skill: {metadata.name} (Phase {metadata.phase})")

    def get_skill(self, name: str) -> BaseSkill:
        """
        获取技能实例 (懒加载)

        Args:
            name: 技能名称

        Returns:
            技能实例

        Raises:
            ValueError: 技能不存在
        """
        if name not in self._instances:
            metadata = self._skills.get(name)
            if metadata is None:
                raise ValueError(f"Skill not found: {name}")

            self._instances[name] = self._instantiate(metadata)

        return self._instances[name]

    def _instantiate(self, metadata: SkillMetadata) -> BaseSkill:
        """
        根据元数据实例化技能

        Args:
            metadata: 技能元数据

        Returns:
            技能实例
        """
        mode = metadata.execution_mode
        if isinstance(mode, str):
            mode = ExecutionMode(mode)

        # 构建技能目录路径
        phase_name = PHASE_NAMES.get(metadata.phase, f"phase{metadata.phase}")
        skill_dir = self.skills_dir / phase_name / metadata.name

        # LLM 技能
        if mode == ExecutionMode.LLM:
            # 读取 prompt.md
            prompt_path = skill_dir / "prompt.md"
            prompt_template = self._load_prompt(prompt_path)

            return create_skill(
                metadata,
                llm_adapter=self.llm,
                prompt_template=prompt_template
            )

        # Script 技能
        elif mode == ExecutionMode.SCRIPT:
            return create_skill(
                metadata,
                skills_dir=self.skills_dir
            )

        # API 技能
        elif mode == ExecutionMode.API:
            # 读取 api_config.yaml
            config_path = skill_dir / "api_config.yaml"
            api_config = self._load_api_config(config_path)

            return create_skill(
                metadata,
                api_config=api_config
            )

        # Hybrid 技能
        elif mode == ExecutionMode.HYBRID:
            prompt_path = skill_dir / "prompt.md"
            prompt_template = self._load_prompt(prompt_path) if prompt_path.exists() else None

            return create_skill(
                metadata,
                llm_adapter=self.llm,
                skills_dir=self.skills_dir,
                prompt_template=prompt_template
            )

        else:
            raise ValueError(f"Unknown execution mode: {mode}")

    def _load_prompt(self, prompt_path: Path) -> str:
        """加载提示词模板"""
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""

    def _load_api_config(self, config_path: Path) -> Dict[str, Any]:
        """加载API配置"""
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def get_metadata(self, name: str) -> Optional[SkillMetadata]:
        """获取技能元数据 (不实例化)"""
        return self._skills.get(name)

    def get_dependencies(self, name: str) -> List[str]:
        """获取技能依赖列表"""
        metadata = self._skills.get(name)
        return metadata.depends_on if metadata else []

    def get_outputs(self, name: str) -> List[str]:
        """获取技能输出列表"""
        metadata = self._skills.get(name)
        return metadata.outputs if metadata else []

    def get_phase_skills(self, phase: int) -> List[str]:
        """获取指定阶段的所有技能名称"""
        return self._by_phase.get(phase, [])

    def list_all_skills(self) -> List[str]:
        """列出所有已注册的技能名称"""
        return list(self._skills.keys())

    def list_skills_by_phase(self) -> Dict[int, List[str]]:
        """按阶段列出所有技能"""
        return self._by_phase.copy()

    def get_skill_count(self) -> int:
        """获取已注册技能总数"""
        return len(self._skills)

    def get_phase_count(self) -> int:
        """获取有技能的阶段数"""
        return len(self._by_phase)

    def has_skill(self, name: str) -> bool:
        """检查技能是否存在"""
        return name in self._skills

    def reload_skill(self, name: str) -> BaseSkill:
        """重新加载技能 (清除缓存)"""
        if name in self._instances:
            del self._instances[name]
        return self.get_skill(name)

    def get_parallel_group(self, phase: int) -> List[List[str]]:
        """
        获取可并行执行的技能组

        基于依赖关系分析，返回可以并行执行的技能组
        每组内的技能可以并行，组间必须顺序执行

        Args:
            phase: 阶段编号

        Returns:
            技能组列表 [[skill1, skill2], [skill3], ...]
        """
        phase_skills = self.get_phase_skills(phase)
        if not phase_skills:
            return []

        # 构建依赖图
        graph: Dict[str, Set[str]] = {s: set() for s in phase_skills}
        in_degree: Dict[str, int] = {s: 0 for s in phase_skills}

        for skill in phase_skills:
            deps = self.get_dependencies(skill)
            for dep in deps:
                # 只考虑阶段内依赖
                if dep in graph:
                    graph[dep].add(skill)
                    in_degree[skill] += 1

        # 拓扑排序 (Kahn算法)
        groups = []
        queue = [s for s in phase_skills if in_degree[s] == 0]

        while queue:
            current_group = sorted(queue)  # 排序保证稳定性
            groups.append(current_group)
            queue = []

            for skill in current_group:
                for dependent in graph[skill]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        return groups

    def validate_dependencies(self) -> List[str]:
        """
        验证所有技能的依赖关系

        Returns:
            错误信息列表 (空列表表示无错误)
        """
        errors = []

        for name, metadata in self._skills.items():
            for dep in metadata.depends_on:
                if dep not in self._skills:
                    errors.append(f"{name}: dependency '{dep}' not found")

        return errors

    def get_statistics(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        mode_counts = {}
        for metadata in self._skills.values():
            mode = metadata.execution_mode
            if isinstance(mode, str):
                mode = ExecutionMode(mode)
            mode_counts[mode.value] = mode_counts.get(mode.value, 0) + 1

        return {
            "total_skills": len(self._skills),
            "total_phases": len(self._by_phase),
            "execution_modes": mode_counts,
            "instantiated": len(self._instances),
            "skills_by_phase": {
                phase: len(skills)
                for phase, skills in self._by_phase.items()
            }
        }

    def __len__(self) -> int:
        """返回已注册技能数"""
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        """检查技能是否已注册"""
        return name in self._skills

    def __repr__(self) -> str:
        return f"SkillRegistry(skills={len(self._skills)}, phases={len(self._by_phase)})"


def create_skill_registry(
    skills_dir: Path,
    llm_adapter: Optional[LLMAdapter] = None
) -> SkillRegistry:
    """
    创建技能注册表实例

    Args:
        skills_dir: 技能根目录
        llm_adapter: LLM适配器 (可选)

    Returns:
        技能注册表实例
    """
    return SkillRegistry(skills_dir, llm_adapter)
