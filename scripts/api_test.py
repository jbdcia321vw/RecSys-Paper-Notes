import os
from openai import OpenAI

api_key = os.environ.get("DEEPSEEK_API_KEY")
print("当前 DEEPSEEK_API_KEY：", api_key if api_key else "❌ 未设置")
if not api_key:
    exit(1)

client = OpenAI(
    api_key=api_key,
    base_url="https://sub.zdyhn.cn/v1",   # ← 换成中转地址
)

MODEL = "deepseek-v4.1-flash"  # 若 /models 里名字不同，改成实际 id

try:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "请只回复 OK"}],
        max_tokens=10,
        stream=False,
    )
    print("✅ Key 有效，模型调用成功。")
    print("返回：", resp.choices[0].message.content)
except Exception as e:
    print("❌ 请求失败：", e)