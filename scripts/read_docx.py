import docx2txt
import os

def read_docx(file_path):
    """读取DOCX文件内容"""
    try:
        content = docx2txt.process(file_path)
        return content
    except Exception as e:
        print(f"读取文档时出错: {str(e)}")
        return None

def save_content_to_file(content, output_file="document_content.txt"):
    """将内容保存到文本文件"""
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"内容已保存到文件: {output_file}")
        return True
    except Exception as e:
        print(f"保存内容时出错: {str(e)}")
        return False

def extract_toc(content):
    """尝试提取文档的目录结构"""
    # 简单方法: 查找1-5级标题
    import re
    
    # 匹配类似 "1. 引言" 或 "4.2 技术栈选型" 这样的标题模式
    title_pattern = r'\n\s*(\d+(\.\d+)*)\s+([^\n]+)'
    
    titles = re.findall(title_pattern, content)
    toc = []
    for num, _, title in titles:
        depth = len(num.split('.'))
        indent = "  " * (depth - 1)
        toc.append(f"{indent}{num} {title.strip()}")
    
    return "\n".join(toc)

def display_sections(content, sections=5):
    """显示文档内容的多个部分"""
    if not content:
        return
    
    length = len(content)
    section_size = length // sections
    
    for i in range(sections):
        start = i * section_size
        end = min((i + 1) * section_size, length)
        print(f"\n\n==== 第{i+1}部分 (字符 {start}-{end}) ====\n")
        print(content[start:end][:2000])  # 每部分显示前2000个字符
        print("...")

if __name__ == "__main__":
    file_path = "Untitled (1).docx"
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
    else:
        content = read_docx(file_path)
        if content:
            print(f"文档总长度: {len(content)} 字符")
            
            # 提取目录
            print("\n文档目录结构:")
            toc = extract_toc(content)
            print(toc)
            
            # 保存内容到文件
            save_content_to_file(content)
            
            # 显示前2000个字符的预览
            print("\n文档开头预览:")
            print(content[:2000])
            
            # 显示文档内容的多个部分
            display_sections(content) 