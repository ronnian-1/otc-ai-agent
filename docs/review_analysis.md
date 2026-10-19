# 独立规则评论解析

`app/analysis/review_parser.py`按关键词输出情感、标签和正负要点，没有模型、数据库或电商请求；未注册成导购Agent工具。`app/review_analysis/`是扩展占位，不应把其入口当作已实现的评论分析应用。

本公开版data/reviews.json有两条独立编写的虚构评论，无真实顾客、账号或来源平台。字段为review_id、sku_id、content；函数需要text，因此显式映射，不修改原文件。

从项目根目录执行纯本地演示：

```python
import json
from pathlib import Path
from app.analysis.review_parser import parse_reviews

rows = json.loads(Path("data/reviews.json").read_text(encoding="utf-8"))
inputs = [{"review_id": row["review_id"], "text": row["content"]} for row in rows]
print(json.dumps(parse_reviews(inputs), ensure_ascii=False, indent=2))
```

可在Python交互解释器粘贴以上代码。直接 `python -m app.analysis.review_parser` 没有CLI输出入口；不要期待它自动读取评论文件。

## 输出与局限

- sentiment取positive、negative或neutral；按正负关键词命中计数比较，同分为neutral。
- tags只来自物流、包装、价格、产品体验和服务的固定词表；同义改写、否定和语境未作通用理解。
- analysis中的positive_points / negative_points为匹配关键词，不是模型总结、医学结论或用户真实评价证据。
- 空输入返回空列表，非字典项过滤；无text时按空文本处理。测试中的text字典不依赖评论文件，公开版另补文件映射一致性检查。

文件样例正负倾向只用于展示确定性规则，不声称情感准确率、真实评论语义质量、商品效果或生产适用性。测试及独立验收边界见[README](../README.md#测试方式)。
