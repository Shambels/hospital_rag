from openai import OpenAI
import config

llm_client = OpenAI(base_url='{llm_url}', api_key='NOKEY')