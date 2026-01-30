"""
Figure Validator - Publication Quality Assurance
图表验证器 - 出版质量保证

检查WCAG无障碍合规、色盲友好性、Nature期刊规范等
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class FigureValidator:
    """图表质量验证器"""
    
    def __init__(self):
        self.validation_results = {}
        
    def validate_all(self, fig_path: str = None, fig: 'plt.Figure' = None) -> Dict:
        """
        执行全面验证
        
        Args:
            fig_path: 图像文件路径
            fig: Matplotlib Figure对象
            
        Returns:
            验证结果字典
        """
        results = {
            'overall_pass': True,
            'wcag_compliance': {},
            'colorblind_safety': {},
            'nature_guidelines': {},
            'recommendations': []
        }
        
        # WCAG合规检查
        results['wcag_compliance'] = self.check_wcag_compliance(fig)
        
        # 色盲安全检查
        results['colorblind_safety'] = self.check_colorblind_safety(fig)
        
        # Nature规范检查
        results['nature_guidelines'] = self.check_nature_guidelines(fig, fig_path)
        
        # 汇总
        results['overall_pass'] = (
            results['wcag_compliance'].get('pass', False) and
            results['colorblind_safety'].get('safe', False) and
            results['nature_guidelines'].get('compliant', False)
        )
        
        # 生成建议
        results['recommendations'] = self._generate_recommendations(results)
        
        self.validation_results = results
        return results
        
    def check_wcag_compliance(self, fig: 'plt.Figure' = None) -> Dict:
        """
        检查WCAG 2.1 AA无障碍合规性
        
        主要检查：
        - 颜色对比度（>=4.5:1 for text, >=3:1 for graphics）
        - 不仅依赖颜色传达信息
        """
        results = {
            'pass': True,
            'contrast_issues': [],
            'color_only_issues': [],
            'details': {}
        }
        
        # 默认检查配置
        min_text_contrast = 4.5
        min_graphics_contrast = 3.0
        
        # 如果有图形对象，提取颜色进行分析
        if fig:
            # 提取图形中使用的颜色
            colors_used = self._extract_colors(fig)
            
            # 检查与背景的对比度
            background = '#FFFFFF'  # 假设白色背景
            
            for color in colors_used:
                contrast = self._calculate_contrast(color, background)
                results['details'][color] = {
                    'contrast_ratio': contrast,
                    'text_pass': contrast >= min_text_contrast,
                    'graphics_pass': contrast >= min_graphics_contrast
                }
                
                if contrast < min_graphics_contrast:
                    results['contrast_issues'].append({
                        'color': color,
                        'contrast': contrast,
                        'required': min_graphics_contrast
                    })
                    results['pass'] = False
                    
        return results
        
    def check_colorblind_safety(self, fig: 'plt.Figure' = None) -> Dict:
        """
        检查色盲友好性
        
        检查三种主要色盲类型：
        - Deuteranopia (红绿色盲-绿)
        - Protanopia (红绿色盲-红)
        - Tritanopia (蓝黄色盲)
        """
        results = {
            'safe': True,
            'simulations': {},
            'problematic_pairs': [],
            'recommendations': []
        }
        
        # 色盲安全调色板
        safe_palettes = {
            'wong': ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7'],
            'tol_bright': ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB']
        }
        
        if fig:
            colors_used = self._extract_colors(fig)
            
            # 模拟各种色盲看到的颜色
            for cb_type in ['deuteranopia', 'protanopia', 'tritanopia']:
                simulated = [self._simulate_colorblind(c, cb_type) for c in colors_used]
                results['simulations'][cb_type] = simulated
                
                # 检查模拟后颜色是否太相似
                for i, c1 in enumerate(simulated):
                    for j, c2 in enumerate(simulated[i+1:], i+1):
                        if self._colors_too_similar(c1, c2):
                            results['problematic_pairs'].append({
                                'type': cb_type,
                                'color1': colors_used[i],
                                'color2': colors_used[j]
                            })
                            results['safe'] = False
                            
            # 检查是否使用了安全调色板
            using_safe = any(
                all(any(self._color_match(c, sc) for sc in palette) for c in colors_used)
                for palette in safe_palettes.values()
            )
            
            if not using_safe and len(colors_used) > 2:
                results['recommendations'].append(
                    "Consider using a colorblind-safe palette (Wong or Tol)"
                )
                
        return results
        
    def check_nature_guidelines(self, fig: 'plt.Figure' = None, 
                                fig_path: str = None) -> Dict:
        """
        检查Nature期刊图表规范
        
        Nature 2025 Figure Guidelines:
        - Font size: 5-7pt
        - Line width: 0.25-1pt
        - Figure width: single column (89mm) or double column (183mm)
        - Resolution: 300 DPI for bitmap
        - Format: PDF/EPS preferred
        """
        results = {
            'compliant': True,
            'issues': [],
            'details': {}
        }
        
        # 尺寸规范
        nature_specs = {
            'single_column_mm': 89,
            'double_column_mm': 183,
            'min_font_pt': 5,
            'max_font_pt': 7,
            'min_line_pt': 0.25,
            'max_line_pt': 1.0,
            'min_dpi': 300
        }
        
        if fig:
            # 检查尺寸
            fig_width_in = fig.get_figwidth()
            fig_width_mm = fig_width_in * 25.4
            
            results['details']['width_mm'] = fig_width_mm
            
            if fig_width_mm > nature_specs['double_column_mm'] * 1.1:
                results['issues'].append(f"Figure width ({fig_width_mm:.0f}mm) exceeds double column ({nature_specs['double_column_mm']}mm)")
                results['compliant'] = False
                
            # 检查DPI
            dpi = fig.dpi
            results['details']['dpi'] = dpi
            
            if dpi < nature_specs['min_dpi']:
                results['issues'].append(f"DPI ({dpi}) below minimum ({nature_specs['min_dpi']})")
                results['compliant'] = False
                
        # 检查文件格式
        if fig_path:
            path = Path(fig_path)
            suffix = path.suffix.lower()
            
            results['details']['format'] = suffix
            
            preferred_formats = ['.pdf', '.eps', '.svg']
            if suffix not in preferred_formats:
                results['issues'].append(f"Format {suffix} not preferred. Use PDF/EPS/SVG.")
                
        return results
        
    def _extract_colors(self, fig: 'plt.Figure') -> List[str]:
        """从图形中提取使用的颜色"""
        colors = set()
        
        try:
            for ax in fig.axes:
                # 从线条提取
                for line in ax.lines:
                    c = line.get_color()
                    if c:
                        colors.add(mcolors.to_hex(c))
                        
                # 从patches（柱状图等）提取
                for patch in ax.patches:
                    c = patch.get_facecolor()
                    if c is not None:
                        colors.add(mcolors.to_hex(c[:3]))
                        
                # 从collections（散点图等）提取
                for coll in ax.collections:
                    fc = coll.get_facecolors()
                    if len(fc) > 0:
                        for c in fc[:5]:  # 限制数量
                            colors.add(mcolors.to_hex(c[:3]))
        except:
            pass
            
        return list(colors) if colors else ['#0077BB']  # 默认
        
    def _calculate_contrast(self, color1: str, color2: str) -> float:
        """计算两个颜色的对比度（WCAG算法）"""
        def relative_luminance(hex_color):
            hex_color = hex_color.lstrip('#')
            r, g, b = [int(hex_color[i:i+2], 16)/255 for i in (0, 2, 4)]
            
            def adjust(c):
                return c/12.92 if c <= 0.03928 else ((c+0.055)/1.055)**2.4
                
            return 0.2126*adjust(r) + 0.7152*adjust(g) + 0.0722*adjust(b)
            
        l1 = relative_luminance(color1)
        l2 = relative_luminance(color2)
        
        lighter = max(l1, l2)
        darker = min(l1, l2)
        
        return (lighter + 0.05) / (darker + 0.05)
        
    def _simulate_colorblind(self, hex_color: str, cb_type: str) -> str:
        """模拟色盲看到的颜色"""
        hex_color = hex_color.lstrip('#')
        r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
        
        # 简化的色盲模拟矩阵
        matrices = {
            'deuteranopia': [[0.625, 0.375, 0], [0.7, 0.3, 0], [0, 0, 1]],
            'protanopia': [[0.567, 0.433, 0], [0.558, 0.442, 0], [0, 0, 1]],
            'tritanopia': [[1, 0, 0], [0, 0.95, 0.05], [0, 0.433, 0.567]]
        }
        
        m = matrices.get(cb_type, matrices['deuteranopia'])
        
        r_new = int(min(255, max(0, m[0][0]*r + m[0][1]*g + m[0][2]*b)))
        g_new = int(min(255, max(0, m[1][0]*r + m[1][1]*g + m[1][2]*b)))
        b_new = int(min(255, max(0, m[2][0]*r + m[2][1]*g + m[2][2]*b)))
        
        return f'#{r_new:02x}{g_new:02x}{b_new:02x}'
        
    def _colors_too_similar(self, c1: str, c2: str, threshold: float = 30) -> bool:
        """检查两个颜色是否太相似"""
        c1 = c1.lstrip('#')
        c2 = c2.lstrip('#')
        
        r1, g1, b1 = [int(c1[i:i+2], 16) for i in (0, 2, 4)]
        r2, g2, b2 = [int(c2[i:i+2], 16) for i in (0, 2, 4)]
        
        distance = np.sqrt((r1-r2)**2 + (g1-g2)**2 + (b1-b2)**2)
        return distance < threshold
        
    def _color_match(self, c1: str, c2: str, tolerance: float = 10) -> bool:
        """检查两个颜色是否匹配"""
        return self._colors_too_similar(c1, c2, tolerance)
        
    def _generate_recommendations(self, results: Dict) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        # WCAG建议
        if not results['wcag_compliance'].get('pass', True):
            recommendations.append(
                "增加颜色对比度：使用更深的颜色或增加线条粗细"
            )
            
        # 色盲建议
        if not results['colorblind_safety'].get('safe', True):
            recommendations.append(
                "使用色盲友好调色板（如Wong或Tol调色板）"
            )
            recommendations.append(
                "除颜色外，使用形状、线型或标签区分数据"
            )
            
        # Nature规范建议
        if not results['nature_guidelines'].get('compliant', True):
            for issue in results['nature_guidelines'].get('issues', []):
                if 'width' in issue.lower():
                    recommendations.append(
                        "调整图形尺寸：单栏89mm或双栏183mm"
                    )
                if 'dpi' in issue.lower():
                    recommendations.append(
                        "导出时设置DPI>=300"
                    )
                if 'format' in issue.lower():
                    recommendations.append(
                        "使用矢量格式（PDF/EPS/SVG）导出"
                    )
                    
        return recommendations
        
    def generate_report(self, output_path: str = None) -> str:
        """生成验证报告"""
        report = []
        report.append("=" * 60)
        report.append("Figure Validation Report")
        report.append("=" * 60)
        report.append("")
        
        results = self.validation_results
        
        # 总体结果
        status = "✅ PASS" if results.get('overall_pass', False) else "❌ FAIL"
        report.append(f"Overall Status: {status}")
        report.append("")
        
        # WCAG合规
        report.append("WCAG 2.1 AA Compliance:")
        wcag = results.get('wcag_compliance', {})
        report.append(f"  Status: {'Pass' if wcag.get('pass', False) else 'Fail'}")
        for issue in wcag.get('contrast_issues', []):
            report.append(f"  - Low contrast: {issue['color']} ({issue['contrast']:.2f})")
        report.append("")
        
        # 色盲安全
        report.append("Colorblind Safety:")
        cb = results.get('colorblind_safety', {})
        report.append(f"  Status: {'Safe' if cb.get('safe', False) else 'Unsafe'}")
        for pair in cb.get('problematic_pairs', [])[:3]:
            report.append(f"  - {pair['type']}: {pair['color1']} ~ {pair['color2']}")
        report.append("")
        
        # Nature规范
        report.append("Nature Guidelines:")
        nature = results.get('nature_guidelines', {})
        report.append(f"  Status: {'Compliant' if nature.get('compliant', False) else 'Non-compliant'}")
        for issue in nature.get('issues', []):
            report.append(f"  - {issue}")
        report.append("")
        
        # 建议
        report.append("Recommendations:")
        for rec in results.get('recommendations', []):
            report.append(f"  • {rec}")
            
        report.append("")
        report.append("=" * 60)
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
                
        return report_text


def validate_figure(fig_path: str = None, fig: 'plt.Figure' = None,
                   generate_report: bool = True) -> Dict:
    """
    便捷函数：验证图表质量
    
    Args:
        fig_path: 图像文件路径
        fig: Matplotlib Figure对象
        generate_report: 是否生成报告
        
    Returns:
        验证结果
    """
    validator = FigureValidator()
    results = validator.validate_all(fig_path, fig)
    
    if generate_report:
        print(validator.generate_report())
        
    return results


if __name__ == '__main__':
    print("Testing Figure Validator...")
    
    # 创建测试图
    fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)
    ax.plot([1, 2, 3], [1, 4, 9], 'o-', color='#0077BB', label='Data')
    ax.plot([1, 2, 3], [1, 2, 3], 's--', color='#EE7733', label='Linear')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    
    # 验证
    validator = FigureValidator()
    results = validator.validate_all(fig=fig)
    
    # 打印报告
    print(validator.generate_report())
    
    print("\nFigure Validator test completed!")
    plt.close()
