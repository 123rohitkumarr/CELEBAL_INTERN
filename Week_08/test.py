from dotenv import load_dotenv

load_dotenv()

from app.tools.keyword_tool import extract_keywords

text = """
Artificial Intelligence is transforming healthcare
by improving diagnosis and reducing costs.
"""

result = extract_keywords(text)
print(result)