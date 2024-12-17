import os
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

import requests

# 定义模式列表
md_sec_patt = [
    r"第[零一二三四五六七八九十百0123456789]+章",
    r"第[零一二三四五六七八九十百0123456789]+[条節]",
]


def read_data_from_pdf(pdf_path: str):
    url = 'http://pdf-parser:8000/convert/'
    files = {'file': open(pdf_path, 'rb')}
    response = requests.post(url, files=files)
    if response.status_code == 200:
        # print("成功转换:", response.json())
        return format_sections(response.json()['text'])
    else:
        print("错误:", response.text)
        return None
    

def format_sections(markdown_content):
    # 将每一行作为一个元素存储在列表中
    markdown_content = re.sub(r'\s+', ' ', markdown_content)
    markdown_content = re.sub(r'-+', ' ', markdown_content)
    lines = markdown_content.split('\n')
    # 定义一个空列表存储处理后的内容
    md_formatted_lines = []
    for line in lines:
        # 针对每种模式进行匹配和处理
        if re.match(md_sec_patt[0], line):  # 章节标题
            line = '## ' + line.strip()
        elif re.match(md_sec_patt[1], line):  # 条节标题
            line = '### ' + line.strip()
        md_formatted_lines.append(line)
    # 将处理后的行重新组合成字符串
    formatted_content = '\n'.join(md_formatted_lines)
    return formatted_content


def get_text_chunks(text: str):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["#", "##", "###", "\n\n\n\n",  "\n\n\n", "\n\n", "\n"], # チャンクの区切り文字リスト
        keep_separator=False,      # チャンクの区切り文字を残す
        chunk_size=512,           # チャンクの最大文字数
        chunk_overlap=100,         # チャンク間の重複する文字数
        length_function=len,      # 文字数で分割
        is_separator_regex=False, # separatorを正規表現として扱う場合はTrue
    )
    chunks = text_splitter.split_text(text)
    return chunks 


if __name__ == "__main__":
    for pdf_file in os.listdir("../assets"):
        pdf_path = f"../assets/{pdf_file}"
        print(pdf_path)
        
        md_text = read_data_from_pdf(pdf_path)
        # print(md_text)
        # formatted_content = format_sections(md_text)
        chunks = get_text_chunks(md_text)
        print(len(chunks))
        # break
        # chunks = get_text_chunks(text)
        # points += get_points(embed_client, chunks, is_openai=IS_OPENAI_EMBEDDING)
    
    
'''
elasticsearch_url = "http://es01:9200/"
client = Elasticsearch(elasticsearch_url, basic_auth=("elastic", "sns-bot"))
COLLECTION_NAME = "knowledge_base"
retriever = ElasticSearchBM25Retriever(client=client, index_name=COLLECTION_NAME)
retriever.add_texts(chunks)
'''