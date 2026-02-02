"""
依赖解析器模块
解析技能依赖关系，构建执行图，实现并行调度
"""

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from .skill_registry import SkillRegistry, PHASE_SKILLS

logger = logging.getLogger(__name__)


class DependencyCycleError(Exception):
    """循环依赖错误"""
    def __init__(self, cycle: List[str]):
        self.cycle = cycle
        super().__init__(f"Circular dependency detected: {' -> '.join(cycle)}")


class DependencyResolver:
    """
    依赖关系解析器

    负责:
    - 构建技能执行DAG
    - 检测循环依赖
    - 计算并行执行组
    - 优化执行顺序
    """

    # 预定义的并行执行配置 (基于对各阶段技能的了解 - 增强版v2)
    PREDEFINED_PARALLEL_GROUPS = {
        # Phase 1: 输入处理阶段
        1: [
            # 第一组: 题目解析相关 (无依赖，可并行)
            ["problem-parser", "problem-type-classifier", "problem-reference-extractor"],
            # 第二组: 数据和文献收集 (依赖第一组的结果)
            ["data-collector", "literature-searcher", "deep-reference-searcher", "ai-deep-search-guide"],
            # 第三组: 引用验证 (依赖第二组的结果)
            ["citation-validator", "citation-diversity-validator"],
        ],
        
        # Phase 2: 问题分析阶段
        2: [
            # 第一组: 问题分解 (必须先执行)
            ["problem-decomposer"],
            # 第二组: 子问题分析和假设生成 (可并行)
            ["sub-problem-analyzer", "assumption-generator"],
            # 第三组: 变量和约束定义 (依赖前两组)
            ["variable-definer", "constraint-identifier"],
        ],
        
        # Phase 3: 建模阶段 (高级算法通过动态选择添加)
        3: [
            # 第一组: 模型选择 (必须先执行，用于确定后续高级算法)
            ["model-selector"],
            # 第二组: 模型设计相关 (可并行)
            ["model-justification-generator", "hybrid-model-designer"],
            # 第三组: 模型构建
            ["model-builder"],
            # 第四组: 模型求解和验证 (可并行)
            ["model-solver", "code-verifier"],
            # 第五组: 高级算法 (动态添加，可并行执行)
            # 注意: 这一组会在运行时动态扩展
        ],
        
        # Phase 4: 验证阶段
        4: [
            # 第一组: 敏感性和不确定性 (可并行)
            ["sensitivity-analyzer", "uncertainty-quantifier"],
            # 第二组: 误差和局限性分析 (依赖第一组)
            ["error-analyzer", "limitation-analyzer"],
            # 第三组: 模型验证、优缺点和伦理分析 (可并行)
            ["model-validator", "strengths-weaknesses", "ethical-analyzer"],
            # 第四组: 模型解释 (依赖前三组)
            ["model-explainer"],
        ],
        
        # Phase 5: 写作阶段
        5: [
            # 第一组: 章节写作 (核心，必须先完成)
            ["section-writer"],
            # 第二组: 章节迭代优化 (Self-Refine模式，依赖section-writer)
            ["section-iterative-optimizer"],
            # 第三组: 事实核查和摘要初稿 (可并行)
            ["fact-checker", "abstract-first-impression"],
            # 第四组: 摘要生成
            ["abstract-generator"],
            # 第五组: 摘要优化
            ["abstract-iterative-optimizer"],
            # 第六组: 备忘录
            ["memo-letter-writer"],
        ],
        
        # Phase 6: 可视化阶段
        6: [
            # 第一组: 图表生成 (可并行)
            ["chart-generator", "table-formatter"],
            # 第二组: 图表叙述和信息图 (可并行，依赖第一组)
            ["figure-narrative-generator", "infographic-generator"],
            # 第三组: 缩放和验证 (可并行)
            ["publication-scaler", "figure-validator"],
        ],
        
        # Phase 7: 集成阶段
        7: [
            # 第一组: LaTeX编译
            ["latex-compiler"],
            # 第二组: 错误处理和引用管理 (可并行)
            ["compilation-error-handler", "citation-manager"],
            # 第三组: 格式和匿名检查 (可并行)
            ["format-checker", "anonymization-checker"],
        ],
        
        # Phase 8: 质量检查阶段
        8: [
            # 第一组: 质量审查和幻觉检测 (可并行)
            ["quality-reviewer", "hallucination-detector"],
            # 第二组: 语法和中式英语检查 (可并行)
            ["grammar-checker", "chinglish-detector"],
            # 第三组: 一致性检查 (依赖前两组)
            ["consistency-checker", "global-consistency-checker"],
        ],
        
        # Phase 9: 优化阶段
        9: [
            # 第一组: 润色和英语优化 (可并行)
            ["final-polisher", "academic-english-optimizer"],
            # 第二组: 提交准备
            ["submission-preparer"],
        ],
        
        # Phase 10: 提交阶段
        10: [
            # 第一组: 提交前验证
            ["pre-submission-validator"],
            # 第二组: 提交清单
            ["submission-checklist"],
        ],
    }

    def __init__(self, registry: SkillRegistry):
        """
        初始化依赖解析器

        Args:
            registry: 技能注册表
        """
        self.registry = registry

        # 缓存
        self._graph_cache: Dict[int, Dict[str, Set[str]]] = {}
        self._parallel_cache: Dict[int, List[List[str]]] = {}

    def build_execution_graph(
        self,
        phase: int,
        skills: Optional[List[str]] = None
    ) -> Dict[str, Set[str]]:
        """
        构建阶段内执行图 (DAG)

        Args:
            phase: 阶段编号
            skills: 技能列表 (None则使用预定义)

        Returns:
            邻接表 {skill: {dependents, ...}}
        """
        # 检查缓存
        if phase in self._graph_cache:
            return self._graph_cache[phase]

        skills = skills or self.registry.get_phase_skills(phase)
        if not skills:
            self._graph_cache[phase] = {}
            return {}

        # 构建图
        graph: Dict[str, Set[str]] = {skill: set() for skill in skills}

        for skill in skills:
            deps = self.registry.get_dependencies(skill)
            for dep in deps:
                # 只考虑阶段内依赖
                if dep in graph:
                    graph[dep].add(skill)

        # 缓存并返回
        self._graph_cache[phase] = graph
        return graph

    def get_parallel_groups(
        self,
        phase: int,
        skills: Optional[List[str]] = None,
        use_predefined: bool = True
    ) -> List[List[str]]:
        """
        获取可并行执行的技能组

        使用拓扑排序将技能分为多层，同层内可并行执行

        Args:
            phase: 阶段编号
            skills: 技能列表
            use_predefined: 是否优先使用预定义的并行组

        Returns:
            技能组列表 [[skill1, skill2], [skill3], ...]
        """
        # 优先使用预定义配置
        if use_predefined and phase in self.PREDEFINED_PARALLEL_GROUPS:
            # 验证预定义的技能是否都存在
            predefined = self.PREDEFINED_PARALLEL_GROUPS[phase]
            all_exist = all(
                self.registry.has_skill(s)
                for group in predefined
                for s in group
            )
            if all_exist:
                self._parallel_cache[phase] = predefined
                return predefined

        # 检查缓存
        if phase in self._parallel_cache:
            return self._parallel_cache[phase]

        skills = skills or self.registry.get_phase_skills(phase)
        if not skills:
            return []

        # 计算并行组
        graph, in_degree = self._compute_in_degrees(phase, skills)
        groups = self._topological_grouping(graph, in_degree)

        # 缓存
        self._parallel_cache[phase] = groups
        return groups

    def _compute_in_degrees(
        self,
        phase: int,
        skills: List[str]
    ) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
        """计算每个技能的入度"""
        graph = self.build_execution_graph(phase, skills)
        in_degree = {skill: 0 for skill in skills}

        for skill in skills:
            deps = self.registry.get_dependencies(skill)
            for dep in deps:
                if dep in in_degree:
                    in_degree[skill] += 1

        return graph, in_degree

    def _topological_grouping(
        self,
        graph: Dict[str, Set[str]],
        in_degree: Dict[str, int]
    ) -> List[List[str]]:
        """
        Kahn算法拓扑分组

        同一层的节点可以并行执行
        """
        groups = []
        remaining = set(in_degree.keys())

        while remaining:
            # 找出入度为0的节点
            current_group = sorted([s for s in remaining if in_degree[s] == 0])

            if not current_group:
                # 没有无依赖的节点，可能存在循环
                logger.warning("No dependency-free nodes found, possible cycle")
                # 强制选择一个
                current_group = [sorted(remaining)[0]]

            groups.append(current_group)

            # 更新入度
            for skill in current_group:
                remaining.remove(skill)
                for dependent in graph.get(skill, set()):
                    if dependent in in_degree:
                        in_degree[dependent] -= 1

        return groups

    def validate_no_cycles(self, phase: int) -> bool:
        """
        检查阶段内是否存在循环依赖

        Args:
            phase: 阶段编号

        Returns:
            True表示无循环，False表示有循环
        """
        skills = self.registry.get_phase_skills(phase)
        if not skills:
            return True

        graph = self.build_execution_graph(phase)

        # DFS检测环
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    if not dfs(neighbor):
                        return False
                elif neighbor in rec_stack:
                    # 找到环
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    logger.error(f"Cycle detected: {' -> '.join(cycle)}")
                    return False

            path.pop()
            rec_stack.remove(node)
            return True

        for skill in skills:
            if skill not in visited:
                if not dfs(skill):
                    return False

        return True

    def detect_cycles(self, phase: int) -> List[List[str]]:
        """
        检测并返回所有循环依赖

        Args:
            phase: 阶段编号

        Returns:
            循环列表
        """
        cycles = []
        skills = self.registry.get_phase_skills(phase)
        graph = self.build_execution_graph(phase)

        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)

            path.pop()
            rec_stack.remove(node)
            return True

        for skill in skills:
            if skill not in visited:
                dfs(skill)

        return cycles

    def get_execution_order(
        self,
        phase: int,
        skills: Optional[List[str]] = None
    ) -> List[str]:
        """
        获取顺序执行列表 (拓扑排序)

        Args:
            phase: 阶段编号
            skills: 技能列表

        Returns:
            拓扑排序后的技能列表
        """
        groups = self.get_parallel_groups(phase, skills)
        # 展平为顺序列表
        return [skill for group in groups for skill in group]

    def get_cross_phase_dependencies(
        self,
        from_phase: int,
        to_phase: int
    ) -> List[Tuple[str, str]]:
        """
        获取跨阶段依赖关系

        Args:
            from_phase: 源阶段
            to_phase: 目标阶段

        Returns:
            [(skill_in_from, skill_in_to), ...]
        """
        dependencies = []
        to_skills = self.registry.get_phase_skills(to_phase)

        for skill in to_skills:
            deps = self.registry.get_dependencies(skill)
            for dep in deps:
                dep_metadata = self.registry.get_metadata(dep)
                if dep_metadata and dep_metadata.phase == from_phase:
                    dependencies.append((dep, skill))

        return dependencies

    def get_critical_path(self, phase: int) -> List[str]:
        """
        获取关键路径 (最长依赖链)

        Args:
            phase: 阶段编号

        Returns:
            关键路径上的技能列表
        """
        skills = self.registry.get_phase_skills(phase)
        graph = self.build_execution_graph(phase)

        # 计算最长路径
        memo = {}

        def dfs(node: str) -> int:
            if node in memo:
                return memo[node]

            max_depth = 0
            for dependent in graph.get(node, set()):
                depth = dfs(dependent)
                max_depth = max(max_depth, depth + 1)

            memo[node] = max_depth
            return max_depth

        # 找到深度最大的起点
        max_depth = 0
        start_node = None

        for skill in skills:
            # 只考虑无入度或入度最小的节点作为起点
            deps = self.registry.get_dependencies(skill)
            cross_phase_deps = [d for d in deps if d in skills]
            if not cross_phase_deps:
                depth = dfs(skill)
                if depth > max_depth:
                    max_depth = depth
                    start_node = skill

        # 重建路径
        if start_node is None:
            return []

        path = []
        current = start_node
        visited = set()

        while current and current not in visited:
            path.append(current)
            visited.add(current)

            # 找下一个深度最大的节点
            next_node = None
            next_depth = -1

            for dependent in graph.get(current, set()):
                if dependent not in visited:
                    depth = memo.get(dependent, 0)
                    if depth > next_depth:
                        next_depth = depth
                        next_node = dependent

            current = next_node

        return path

    def analyze_complexity(self, phase: int) -> Dict[str, Any]:
        """
        分析阶段执行复杂度

        Args:
            phase: 阶段编号

        Returns:
            复杂度分析结果
        """
        skills = self.registry.get_phase_skills(phase)
        groups = self.get_parallel_groups(phase)
        graph = self.build_execution_graph(phase)

        # 计算边数
        total_edges = sum(len(neighbors) for neighbors in graph.values())

        # 计算依赖深度
        max_depth = len(groups)

        # 计算最大并行度
        max_parallel = max(len(g) for g in groups) if groups else 0

        # 串行执行时间 (假设每个技能1单位时间)
        serial_time = len(skills)

        # 理想并行执行时间
        parallel_time = len(groups)

        # 加速比
        speedup = serial_time / parallel_time if parallel_time > 0 else 1.0

        return {
            "phase": phase,
            "total_skills": len(skills),
            "total_dependencies": total_edges,
            "depth": max_depth,
            "max_parallel": max_parallel,
            "serial_time": serial_time,
            "parallel_time": parallel_time,
            "speedup": speedup,
            "parallelism_ratio": max_parallel / len(skills) if skills else 0
        }

    def clear_cache(self):
        """清除缓存"""
        self._graph_cache.clear()
        self._parallel_cache.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """获取解析器统计信息"""
        stats = {
            "phases_analyzed": len(self._graph_cache),
            "phases_cached": len(self._parallel_cache),
            "complexity_by_phase": {}
        }

        for phase in range(1, 11):
            if self.registry.get_phase_skills(phase):
                stats["complexity_by_phase"][f"phase{phase}"] = self.analyze_complexity(phase)

        return stats

    def visualize_graph(self, phase: int) -> str:
        """
        生成可视化图的DOT格式

        Args:
            phase: 阶段编号

        Returns:
            DOT格式的图描述
        """
        skills = self.registry.get_phase_skills(phase)
        graph = self.build_execution_graph(phase)

        lines = ["digraph PhaseDependencies {"]
        lines.append(f"  label=\"Phase {phase} Dependencies\";")
        lines.append("  rankdir=LR;")
        lines.append("  node [shape=box];")

        # 节点
        for skill in skills:
            metadata = self.registry.get_metadata(skill)
            mode = metadata.execution_mode.value if metadata else "unknown"
            lines.append(f'  "{skill}" [label="{skill}\\n({mode})"];')

        # 边
        for skill, dependents in graph.items():
            for dep in dependents:
                lines.append(f'  "{skill}" -> "{dep}";')

        lines.append("}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"DependencyResolver(phases={len(self._graph_cache)})"


def create_dependency_resolver(registry: SkillRegistry) -> DependencyResolver:
    """
    创建依赖解析器实例

    Args:
        registry: 技能注册表

    Returns:
        依赖解析器实例
    """
    return DependencyResolver(registry)
