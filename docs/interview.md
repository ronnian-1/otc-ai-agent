# 模块与接口说明

本说明对应当前公开版代码；商品与评论为独立合成样例。接口存在不等于已验证真实模型能力。

## 调用入口

`python -m app.main` 调用 `app.main.main()`。默认 `build_agent(llm=None, memory=None, timeouts=None)` 读取模型配置；测试注入支持 `bind_tools` 的脚本模型，避免provider。`run_agent_turn(runtime, user_input)` 返回最终展示字符串，可选输入/输出审核回调供受控测试。底层Agent invoke不是完整安全入口。

运行时包含agent、memory、execution_policy等对象；last_result、last_reply_plan、last_fact_check是诊断，不是可直接信任的商品数据。不要共享带原始咨询或候选的诊断。

## 工具与数据

`search_product.invoke({"query": "S001"})` 按SKU精确或名称片段检索。`compare_product.invoke({"sku_ids": "S001,S002"})` 输出basic_difference、suitable_crowd_difference、taboo_difference、product_position_difference。工具直接使用根目录data/sku.json，不依赖外部交付路径。

基础SKU每条11键：sku_id/name/category/spec/price/stock/delivery/suitable_crowd/taboo/product_desc/forbidden_words；前10项为字符串，最后为数组。search返回前10项，forbidden_words目前未被app读取。基础库没有ingredients字段，不能解释为已知成分。

工具模糊搜索与Agent身份绑定不同：完整唯一规范化名称才能绑定；局部名即使唯一仍澄清。名称空白和全半角归一化不是语义别名识别。

## 状态和事实

`SessionMemory.get_history()`、`get_state()`、`get_last_turn()`用于读取副本；`clear()`清空会话及绑定runtime诊断。显式共享Memory意味着共享会话，不保证跨调用方隔离。

`SessionFacts.get_query()`读取查询诊断，`get_fact()`受字段/来源/时效约束，`reusable_result()`只允许有限精确查询，`discard_queries()`隔离中止轮次涉及的范围。默认source_path在函数定义时绑定；临时测试重定向loader时须显式为SessionFacts传同一来源路径。

`SessionState.resolve_name()`校验当前待解析query；模型文字不能自行调用它建立身份。比较条件、顺序、撤回和用户预算来自明确状态更新。

## 执行与安全

`TimeoutPolicy`控制单工具、单模型、整轮等待和工具worker数；模型8、工具12、无效3、错误2是常量阈值。成功不会清空本轮工具错误数。已运行同步请求不能强制中断，后台迟到结果不得写事实或历史。同一runtime并发完整轮次被拒绝；不同runtime各有容量，不是全局服务限流器。

`check_user_input()`、`check_ai_output()`、`apply_fact_boundary()`是规则边界，完整轮次会组合使用。输入拒答保留受限安全上下文；被拒候选只可存在诊断，不进入正常助手历史或商品事实。固定声明见[README](../README.md#safety-guard-说明)。

## 评论与日志

`parse_reviews(reviews)`接受含text的字典列表；data/reviews.json的content需由调用方映射。扩展`analyze_reviews()`不是已完成分析服务，详见[评论说明](review_analysis.md)。

日志默认控制台输出，无业务数据库或应用日志文件配置；运行依赖可有额外行为。CLI异常使用固定阶段提示，但普通会话诊断、调用者保存副本和依赖日志仍需单独审查。

测试、边界和独立结果以[README](../README.md#测试方式)为准，没有真实模型、真实平台或干净环境保证。
