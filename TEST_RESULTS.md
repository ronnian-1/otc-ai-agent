# 已有离线测试记录

本文件于2026-10-18作发布文档同步，引用同日最新已有的政策拒绝／停止契约验收记录。**本次没有重新运行任何测试。** 原始日志、精确机器命令及隔离启动器保存在项目外审查证据中，未复制进公开目录。

## 已有报告结果

|阶段|passed|failed|xfailed|skipped|退出码|
|---|---:|---:|---:|---:|---:|
|原T1单独定向|1|0|0|0|0|
|政策拒绝／熔断／缓存|138|0|0|0|0|
|相关Agent／安全／工具|245|0|1|0|0|
|跨模块回归|337|0|0|0|0|
|全量tests/|541|0|1|0|0|

全量收集542项，各阶段有重叠，不能累加为更多覆盖。原有strict xfail为tests/test_sku_search.py::test_search_none_query_current_langchain_validation_behavior，检查LangChain字符串Schema前置验证拒绝None的现有行为，未删除或弱化断言。

既有报告环境为Python3.10.22，pytest9.1.1、langchain1.4.5、langchain-core1.6.10、langchain-openai1.7.1、pydantic2.14.1、python-dotenv1.2.4；这不等于其他环境或所有声明版本组合均可复现。没有执行干净环境安装。

## 选集和可观察契约

- T1：tests/test_public_scenarios.py::test_T1_policy_proposals_never_execute_or_reuse。相同物流输入“发货需要等多久，保税仓税费能确定吗？”；两个商品提议被执行前拒绝，实际商品工具0、错误2、停止原因为tool_error_budget、模型调用1；无法确认回复，状态及既有S001缓存逐字段保持。
- 政策／熔断／缓存：test_shipping_policy_boundary.py、test_final_acceptance.py、test_execution_control.py、test_facts.py、test_memory.py。
- 相关模块：test_final_acceptance.py、test_public_scenarios.py、test_agent_safety.py、test_guard.py、test_agent_flow.py、test_compare.py、test_sku_search.py。
- 跨模块：test_agent_flow.py、test_agent_grounding.py、test_agent_safety.py、test_agent_context.py、test_product_identity.py、test_memory.py、test_session_state.py、test_session_state_integration.py、test_guard.py、test_execution_control.py。
- test_shipping_policy_boundary.py共39项，包含前轮32项和后追加7项。后7项覆盖3种双拒绝后的缓存保留／S002恢复、工具JSON不能伪装可信政策门、重复call_id计数及政策拒绝混合真实错误仍隔离。原七个历史失败用例在最新全量中全部通过。

所有文件均在tests/中。已执行的完整入口测试使用脚本／Mock模型和本地合成工具；预算单元测试是局部流程，不冒充完整Agent。真实模型能否自主作出同样决策未验证，医疗效果、真实平台、生产负载与任意语言泛化均未验证。

## 以后可自行执行的命令

以下是普通便携命令，**本次未执行，不是历史验收的精确隔离命令**。普通pytest可能创建项目外临时文件，并可能因依赖导入进行IPv6能力探测；这些选项不提供网络、凭证或写入沙箱。

从项目根目录，安装运行依赖和pytest之后，在PowerShell执行：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -B -m pytest -p no:cacheprovider tests/test_public_scenarios.py::test_T1_policy_proposals_never_execute_or_reuse -q -ra
python -B -m pytest -p no:cacheprovider tests/test_shipping_policy_boundary.py tests/test_final_acceptance.py tests/test_execution_control.py tests/test_facts.py tests/test_memory.py -q -ra
python -B -m pytest -p no:cacheprovider tests/ -q -ra
```

命令路径、选项及入口本次仅静态核对。已有验收采用项目外专用隔离启动器，禁止连接、DNS、绑定、真实provider、凭证读取和项目写入，11项隔离自检与业务断言分别记录；普通命令不继承这些保证。

## 历史关系

最初公开版495 passed、7 failed、1 xfailed；上一轮物流识别修复只获得32项边界通过及原七项6过1失败，后续全量当时未执行。最新政策停止修复在保留错误预算的同时区分未执行政策提议，取得上表结果。内部旧交付版484 passed、1 xfailed属于另一份数据／版本，不作为公开版当前成绩。本次只改文档，不改验收过的实现、测试、数据或依赖。

代码静态契约、合成数据边界与公开授权状态分别见[README](README.md)、[数据来源](DATA_PROVENANCE.md)、[发布说明](RELEASE_NOTES.md)。
