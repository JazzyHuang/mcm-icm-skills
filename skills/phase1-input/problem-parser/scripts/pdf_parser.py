"""
PDF解析器
将美赛题目PDF转换为结构化文本
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_problem_pdf(pdf_path: str) -> Dict:
    """
    解析美赛题目PDF文件
    
    Args:
        pdf_path: PDF文件路径
        
    Returns:
        包含解析结果的字典
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber not installed. Run: pip install pdfplumber")
        raise
        
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
    result = {
        'text': '',
        'pages': [],
        'images': [],
        'tables': [],
        'metadata': {}
    }
    
    with pdfplumber.open(pdf_path) as pdf:
        # 提取元数据
        result['metadata'] = {
            'page_count': len(pdf.pages),
            'filename': pdf_path.name
        }
        
        for i, page in enumerate(pdf.pages):
            page_data = {
                'page_number': i + 1,
                'text': '',
                'tables': [],
                'images': []
            }
            
            # 提取文本
            text = page.extract_text()
            if text:
                page_data['text'] = text
                result['text'] += text + '\n\n'
                
            # 提取表格
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    page_data['tables'].append(table)
                    result['tables'].append({
                        'page': i + 1,
                        'data': table
                    })
                    
            # 记录图片位置
            if page.images:
                for img in page.images:
                    img_info = {
                        'page': i + 1,
                        'x0': img.get('x0'),
                        'y0': img.get('y0'),
                        'x1': img.get('x1'),
                        'y1': img.get('y1')
                    }
                    page_data['images'].append(img_info)
                    result['images'].append(img_info)
                    
            result['pages'].append(page_data)
            
    logger.info(f"Parsed PDF: {len(result['pages'])} pages, "
                f"{len(result['tables'])} tables, {len(result['images'])} images")
    
    return result


def parse_problem_text(text: str) -> Dict:
    """
    解析美赛题目文本
    
    Args:
        text: 题目文本
        
    Returns:
        结构化的问题描述
    """
    result = {
        'problem_type': None,
        'problem_title': None,
        'year': None,
        'background': '',
        'main_questions': [],
        'provided_data': None,
        'constraints': [],
        'keywords': [],
        'raw_text': text
    }
    
    # 清理文本
    text = clean_text(text)
    
    # 提取题型
    result['problem_type'] = extract_problem_type(text)
    
    # 提取年份
    result['year'] = extract_year(text)
    
    # 提取标题
    result['problem_title'] = extract_title(text)
    
    # 提取背景
    result['background'] = extract_background(text)
    
    # 提取子问题
    result['main_questions'] = extract_questions(text)
    
    # 提取数据要求
    result['provided_data'] = extract_data_requirements(text)
    
    # 提取约束
    result['constraints'] = extract_constraints(text)
    
    # 提取关键词
    result['keywords'] = extract_keywords(text)
    
    return result


def clean_text(text: str) -> str:
    """清理文本"""
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    # 规范化换行
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip()


def extract_problem_type(text: str) -> Optional[str]:
    """提取题型"""
    patterns = [
        r'Problem\s+([A-F])\s*:',
        r'MCM\s+Problem\s+([A-F])',
        r'ICM\s+Problem\s+([D-F])',
        r'\b([A-F])\s+题',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
    return None


def extract_year(text: str) -> Optional[int]:
    """提取年份"""
    patterns = [
        r'20[2-9][0-9]\s+MCM',
        r'20[2-9][0-9]\s+ICM',
        r'MCM\s+20[2-9][0-9]',
        r'ICM\s+20[2-9][0-9]',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            year_match = re.search(r'20[2-9][0-9]', match.group())
            if year_match:
                return int(year_match.group())
                
    return None


def extract_title(text: str) -> Optional[str]:
    """提取标题"""
    patterns = [
        r'Problem\s+[A-F]\s*:\s*(.+?)(?:\n|Background)',
        r'MCM\s+Problem\s+[A-F]\s*:\s*(.+?)(?:\n|Background)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            title = match.group(1).strip()
            # 清理标题
            title = re.sub(r'\s+', ' ', title)
            return title[:200]  # 限制长度
            
    return None


def extract_background(text: str) -> str:
    """提取背景"""
    patterns = [
        r'Background\s*:?\s*(.+?)(?:Your team|Requirements|Tasks)',
        r'Introduction\s*:?\s*(.+?)(?:Your team|Requirements|Tasks)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            background = match.group(1).strip()
            return background[:2000]  # 限制长度
            
    return ''


def extract_questions(text: str) -> List[Dict]:
    """提取子问题"""
    questions = []
    
    # 常见的问题列表模式
    patterns = [
        r'(?:^|\n)\s*(\d+)[.)]\s*(.+?)(?=(?:\n\s*\d+[.)]|\Z))',
        r'(?:^|\n)\s*([a-z])[.)]\s*(.+?)(?=(?:\n\s*[a-z][.)]|\Z))',
        r'(?:Task|Question|Requirement)\s*(\d+)\s*:?\s*(.+?)(?=(?:Task|Question|Requirement|\Z))',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        if matches:
            for i, (num, content) in enumerate(matches):
                question = {
                    'id': i + 1,
                    'description': content.strip()[:500],
                    'type': classify_question_type(content)
                }
                questions.append(question)
            break
            
    return questions


def classify_question_type(content: str) -> str:
    """分类问题类型"""
    content_lower = content.lower()
    
    if any(word in content_lower for word in ['model', 'develop', 'create', 'build']):
        return 'modeling'
    elif any(word in content_lower for word in ['analyze', 'analysis', 'evaluate', 'assess']):
        return 'analysis'
    elif any(word in content_lower for word in ['memo', 'letter', 'report', 'write']):
        return 'communication'
    elif any(word in content_lower for word in ['predict', 'forecast', 'estimate']):
        return 'prediction'
    elif any(word in content_lower for word in ['optimize', 'maximize', 'minimize']):
        return 'optimization'
    else:
        return 'general'


def extract_data_requirements(text: str) -> Optional[Dict]:
    """提取数据要求"""
    result = {
        'files': [],
        'description': ''
    }
    
    # 查找数据文件引用
    file_patterns = [
        r'(?:file|data|dataset|spreadsheet)\s*:?\s*([A-Za-z0-9_.-]+\.(?:csv|xlsx|xls|txt|json))',
        r'([A-Za-z0-9_.-]+\.(?:csv|xlsx|xls|txt|json))',
    ]
    
    for pattern in file_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            result['files'] = list(set(matches))
            break
            
    # 查找数据描述
    data_patterns = [
        r'(?:Data|Dataset)\s*:?\s*(.+?)(?:Your team|Requirements|\n\n)',
        r'(?:provided|given|attached)\s+data\s*:?\s*(.+?)(?:\n\n|\Z)',
    ]
    
    for pattern in data_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            result['description'] = match.group(1).strip()[:500]
            break
            
    if result['files'] or result['description']:
        return result
    return None


def extract_constraints(text: str) -> List[str]:
    """提取约束条件"""
    constraints = []
    
    # 常见约束关键词
    constraint_keywords = [
        'must', 'should', 'cannot', 'limited', 'constraint',
        'requirement', 'assume', 'given that', 'no more than',
        'at least', 'at most', 'within'
    ]
    
    sentences = re.split(r'[.!?]', text)
    for sentence in sentences:
        sentence = sentence.strip()
        if any(keyword in sentence.lower() for keyword in constraint_keywords):
            if 10 < len(sentence) < 300:
                constraints.append(sentence)
                
    return constraints[:10]  # 限制数量


def extract_keywords(text: str) -> List[str]:
    """提取关键词"""
    # 简单的关键词提取
    # 在实际使用中可以用更高级的NLP方法
    
    # 移除停用词
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as',
        'into', 'through', 'during', 'before', 'after', 'above', 'below',
        'and', 'or', 'but', 'if', 'while', 'although', 'because', 'since',
        'this', 'that', 'these', 'those', 'it', 'its', 'your', 'our',
        'you', 'we', 'they', 'their', 'team', 'model', 'problem'
    }
    
    # 提取词语
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    
    # 统计词频
    word_freq = {}
    for word in words:
        if word not in stopwords:
            word_freq[word] = word_freq.get(word, 0) + 1
            
    # 排序并返回前20个
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_words[:20]]


if __name__ == '__main__':
    # 测试代码
    test_text = """
    2026 MCM Problem A: Optimizing Solar Panel Placement
    
    Background: Solar energy is becoming increasingly important as the world
    transitions to renewable energy sources. Efficient placement of solar panels
    can significantly impact energy production.
    
    Your team should:
    1. Develop a mathematical model to determine optimal panel angles based on
       geographic location and seasonal variations.
    2. Analyze the impact of different panel configurations on energy output.
    3. Write a one-page memo to a solar company summarizing your recommendations.
    
    Data: Use the provided solar_data.csv file containing historical solar
    radiation measurements.
    """
    
    result = parse_problem_text(test_text)
    
    import json
    print(json.dumps(result, indent=2))
