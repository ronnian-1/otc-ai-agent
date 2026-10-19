# OTC Agent · 合成数据作品集

本项目是本地 OTC / 保健品导购 Agent 原型，展示商品工具、显式会话状态、规则式医疗安全、事实来源与执行控制。`data/sku.json` 和 `data/reviews.json` 均为本公开版独立编写的虚构样例，不代表真实商品、顾客、医疗事实或天猫数据。没有接入真实天猫、旺旺 API、商家账号、订单或物流服务。

本项目没有医学合规认证或生产环境效果证明。正常 CLI 咨询需要自行配置模型服务；离线测试使用脚本模型，不能证明真实模型的自主判断、安全性或回答质量。

**当前状态：技术检查完成，公开授权待确认。** 最新已有离线验收报告记录为 **541 passed、1 xfailed、0 failed、0 skipped**；本次仅同步文档和检查目录，没有重新运行测试。作者公开权、保密／委托条款与项目许可证仍待权利人确认，不构成发布批准。

## 功能与限制

- 按 SKU 或完整名称查询本地商品；完整唯一名称可绑定，局部名称与多候选需要澄清。名称归一化处理空白及全半角，不提供语义别名解析。
- 对两个商品比较价格、规格、人群、禁忌和定位。完整比较可按有效工具字段程序渲染；场景关联使用有限文字规则，不等于个体适用性判断。
- 输入、模型候选、事实替换回复及最终拒答经过规则审核。命中的治疗承诺、替代处方药、个性化用量等请求受到限制，规则可能漏报或误报。
- 缺失成分不得从商品名称、模型常识或其他商品补写；工具事实按来源版本、SKU、字段、参数和有效期管理。
- 规则能识别的纯物流/税费问题在商品工具执行和缓存读取前受阻；混合商品咨询可分流。库存、配送备注只是虚构本地记录，不能承诺真实库存、税费或到货时间。
- 历史、显式购物状态、工具事实与诊断分开保存在进程内。没有数据库、跨进程记忆、健康画像、实时平台规则或人工自动转接。

## 项目结构

```text
otc-agent/
├── app/
│   ├── main.py             # 交互 CLI
│   ├── agent/              # agent/state/facts/memory/reply/grounding/execution
│   ├── safety/guard.py      # 输入与输出规则
│   ├── tools/              # search_product / compare_product / data_loader
│   ├── analysis/           # 独立规则评论解析
│   ├── review_analysis/    # 扩展占位，非完整评论分析应用
│   └── utils/              # 日志工具
├── data/sku.json           # 两条独立合成商品，实际工具数据
├── data/reviews.json       # 两条独立合成评论，仅文档示例读取
├── tests/                 # 单元、脚本模型流程与公开场景回归
├── docs/                  # 展示、接口、场景与边界
├── BUSINESS_ANALYSIS.md     # 初始构建时期的业务分析，验收状态已过时
├── TEST_RESULTS.md          # 最新已有离线记录与证据边界
├── DATA_PROVENANCE.md       # 合成数据与测试夹具来源说明
├── RELEASE_NOTES.md         # 当前候选状态及历史文档说明
├── .env.example            # 空凭证模板
├── .gitignore
└── requirements.txt        # 运行依赖，pytest需另装
```

## 安装与配置

要求 Python 3.10+。既有离线验收使用已安装的 Python 3.10.22 环境；本次没有安装、启动或测试。干净环境安装、其他平台和所有依赖版本组合仍未验证。`requirements.txt` 使用版本范围，没有锁文件。

以下安装步骤会创建文件并访问软件源，供使用者自行执行；本轮未执行安装或联网。

PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install pytest
if (-not (Test-Path -LiteralPath .env)) { Copy-Item .env.example .env }
```

macOS / Linux：

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install pytest
[ -f .env ] || cp .env.example .env
```

只在需要连接模型时填写自己的本地 `.env`，不提交真实凭证。配置由 `load_llm_from_env()` 调用 `load_dotenv()` 读取；已有进程环境变量默认不会被覆盖。

| 变量 | 代码行为 |
| --- | --- |
| `OPENAI_API_KEY` | 默认 ChatOpenAI 初始化必填；注入脚本模型的测试不需要真实 Key。 |
| `OPENAI_MODEL` | 未设置时为 `gpt-4.1-mini`，用户需确认服务端支持所选模型。 |
| `OPENAI_BASE_URL` | 可选兼容服务地址，未设置时使用 SDK 默认；本轮未验证任何真实服务。 |
| `LOG_LEVEL` | 默认 INFO，未知值回退 INFO；CLI 在加载 .env 前初始化日志，需在启动前设置进程变量。 |
| `APP_ENV` | 模板保留字段，业务代码未读取，不是隔离开关。 |

模型默认 temperature=0、max_retries=0、关闭模型缓存；这些参数不保证真实模型正确。输入审核通过的上下文可能发送给所配置供应商，可能产生费用。会话诊断可保留原始输入及被拦截候选，不能作为已脱敏日志分享。

## 启动与使用

在项目根目录，配置好模型后运行：

```bash
python -m app.main
```

输入 exit / quit 退出，空输入忽略；EOF 和输入阶段 KeyboardInterrupt 退出。CLI 初始化/单轮异常使用固定提示，不能据此声称所有依赖日志均已脱敏。没有离线 CLI `--mock`、HTTP API 或 UI。启动提示中的平台名称沿用历史项目定位，不表示官方关联、授权或平台服务已接入。

合成数据查询示例（说明性输入，不是模型实测记录）：

```text
S001多少钱？
演示星屿营养片的规格是什么？
比较S001和S002的价格
改看S002，它的成分能确认吗？
发货需要等多久，保税仓税费能确定吗？
```

本地合成记录：S001 为 176 元、48 粒/瓶；S002 为 84 元、22g/支。基础库没有成分字段和具体标签用量。示例名称不是实际品牌，不能据样例购买或自行用药。

## 架构与接口

```text
run_agent_turn → 输入规则 → SessionState / 名称解析 / ReplyPlan
  → Agent + Execution（按任务允许工具、预算与来源检查）
  → 本地工具 / SessionFacts → 候选 → 输出规则 → Grounding
  → 替换复审 / 安全兜底 → 最终回复 → 历史
```

`build_agent(llm=None, memory=None, timeouts=None)` 构造 runtime；`run_agent_turn(runtime, user_input)` 是同步完整轮次入口。直接调用底层 `runtime["agent"].invoke` 不包含完整最终审核，异步执行不受支持。同一 runtime 不允许并发重入。

`SessionState` 管理商品、比较顺序、字段、条件和明确预算，不从模型文字绑定商品。`SessionFacts` 只在有效工具边界写入；完整精确SKU查询可有限复用，比较/模糊请求等重新查询。`memory.clear()` 清空应用持有的历史、状态、事实及诊断，不清除调用者副本或强制终止后台线程。

## Safety Guard 说明

保留的固定声明是程序契约，不是对合成商品的医疗或监管证明：

> 本产品为膳食补充剂 / OTC 外用产品，不具备治疗疾病功效，不能替代药物，身体不适建议咨询医师。

拒答还包含医生或药师、人工客服和专业人士引导；没有自动转接。模型自称安全文本不能获得固定拒答专用复审豁免。医疗和事实审核都是有限规则，不能代替专业诊断，也不能保证覆盖所有表达。

## 执行控制与恢复

| 默认每轮限制 | 值 |
| --- | --- |
| 模型调用 | 8次 |
| 工具调度尝试 | 12次，不等于真实工具执行数 |
| 无效调用累计阈值 | 3次 |
| 工具错误累计阈值 | 2次，成功不清零；之后停止下一轮模型 |
| 工具 / 模型等待 | 5 / 30秒 |
| 整轮时间 | 60秒，合作式检查，不是硬实时保证 |
| 每runtime只读工具 / 模型worker容量 | 默认4 / 1 |

通过 `TimeoutPolicy` 可设等待时间和工具worker数，调用次数阈值是代码常量。超时停止等待、尝试取消未运行任务并隔离迟到结果；已运行同步函数不能强行终止，可能继续占用容量。中止轮次触及的查询事实会被隔离，无关既有事实保留。执行前的纯物流政策拒绝仍累计错误并可触发两次错误停止；由于商品工具和缓存读取均未执行，这些提议不清理既有查询记录。若停止仅由这类拒绝造成，使用无法确认时效／税费的固定回复；实际工具错误或混合错误仍保留原隔离与停止行为。新轮次预算重置，但不承诺所有故障自动恢复。

## 测试方式

从项目根目录使用同一解释器运行；自动插件加载关闭可减少环境插件干扰，但以下命令本身不是网络/凭证/写入沙箱：

PowerShell：

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
python -B -m pytest -p no:cacheprovider tests/test_final_acceptance.py -q -ra
python -B -m pytest -p no:cacheprovider tests/ -q -ra
```

POSIX shell：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest -p no:cacheprovider tests/ -q -ra
```

依赖导入可能做本地IPv6能力探测，普通pytest并非完全无副作用。此公开候选的独立验收使用项目外受审查启动器：阻断网络/DNS/绑定、凭证读取、真实provider和项目写入，仅导入urllib3时提供IPv6不可用结果，随后恢复socket能力标志。该机器隔离脚本和内部证据不纳入公开仓库，普通命令不自动提供其保证。

### 本公开版独立验收

以下为 **2026-10-18 最新已有验收报告记录**，不是本次文档同步的运行结果。本次没有执行pytest、CLI、安装或真实模型调用。

| 范围 | 已有报告记录 |
| --- | --- |
| 原 T1 单独定向 | 1 passed |
| 政策拒绝／熔断／缓存 | 138 passed |
| 相关 Agent／安全／工具 | 245 passed、1 xfailed |
| 跨模块回归 | 337 passed |
| 全量 tests/ | **541 passed、1 xfailed、0 failed、0 skipped**，542项，退出码0 |

阶段范围重叠，不累加通过数。原有严格xfail针对LangChain字符串Schema在函数执行前拒绝None输入，不能计作通过。

已有报告中的T1输入为“发货需要等多久，保税仓税费能确定吗？”。在脚本模型提出两个无关商品工具时，实际商品执行0次；返回无法确认；两次错误仍触发停止，未进入下一轮模型；既有商品状态和查询缓存保留。相邻双拒绝后重新查询S002的恢复测试也通过。这些是指定样本下的程序行为，不能外推到全部自然语言表达。

早期公开版495 passed、7 failed、1 xfailed及其物流失败描述是历史；后续两次最小修复已在上述最新记录中复验。内部旧版484 passed、1 xfailed也不是当前公开版结果。详细阶段、环境、覆盖与局限见[测试记录](TEST_RESULTS.md)，版本关系及保留的历史文档说明见[发布说明](RELEASE_NOTES.md)。

这些是离线脚本／Mock模型配合本地合成商品工具及规则的流程证据，不能证明真实模型自主选工具、理解上下文、稳定遵守规则或生产效果。当前业务源码、测试、数据与依赖在本次文档同步中未改；对其与已有验收清单的对应仅做静态哈希核验。

## 数据来源与公开边界

已有构建来源审计记录将两份data文件记为独立编写的合成样例；本次核对文件、字段及派生夹具路径，未在所用原资料完整长字段匹配规则下发现残留。该有限检查不等于法律原创性证明，也不能据此认定原题资料、继承代码或固定声明已获公开授权。

测试临时标签、同名目录、错误价格和危险候选来自合成基库或显式测试字典，用于验证边界，不是商品成分、用量或医学事实。保留的SKU格式、字段、通用场景词和固定程序声明属于既有接口／测试契约；来源和未确认事项见[数据来源说明](DATA_PROVENANCE.md)。

没有附带原始任务JSON、真实模型权重、外部商品库、真实用户评论、账号或密钥。当前没有LICENSE；作者／雇佣／委托／贡献者权利、原任务知识产权与保密限制均需人工确认，不能将构建许可等同于公开许可。依赖包和模型服务分别适用其自身条款。

当前候选的相对文件清单和SHA-256保存在项目外发布审查材料中，不包含内部交付包、原始任务数据或历史证据。该候选清单用于技术追溯，**不是已批准上传的文件列表**；静态扫描也不能保证绝无秘密或权利问题。未经确认不发布。

当前说明：[测试记录](TEST_RESULTS.md)、[数据来源](DATA_PROVENANCE.md)、[发布说明](RELEASE_NOTES.md)、[模块接口](docs/interview.md)、[业务边界](docs/business_thinking.md)、[评论解析](docs/review_analysis.md)。

初始构建时期材料：[业务分析](BUSINESS_ANALYSIS.md)、[作品展示](docs/project_showcase.md)、[人工合成场景](docs/demo_logs.md)。这些文件保留首轮验收失败等历史表述，不是当前验收结论；本次未改这些文件，当前状态以本README、测试记录和发布说明为准。
