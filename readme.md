# Mini-project 2: Selecting Reliable LLM Reasoning

**姓名：HE YUXIN；学号：3036897306。**

本目录是最终 **v14 提交版**，包含离线运行、训练复现、必要评测证据和报告。模型由题目—解答低秩交互评分器（权重 0.75）与基线 v2（权重 0.25）组成。

| 数据集 | 答对题数 | 准确率 |
|---|---:|---:|
| 验证集 | 753 / 798 | 94.3609022556391% |
| 测试集 | 388 / 467 | **83.08351177730192%** |

测试相对 v2 增加 8 题（+1.7131 个百分点）。这是一组课程数据上的实测提升，配对置信区间包含零。最终配置在本次测试前按验证集选择并冻结，没有根据测试结果继续调参。

## 1. 文件说明

```text
MP2_Submission_v14/
  readme.md                     本运行指引
  report.pdf                    黑白学术报告，最多 10 页；最后为 AI 辅助声明
  requirements.txt              五个固定依赖
  selector.py                   最终推理入口 predict(bank, arrays, checkpoint)
  features.py                   96 维原有数值/文本特征
  math_consistency.py            40 维数学一致性及组内相对特征
  common.py                     原始数据读取与数字解析器（未改动）
  predict.py                    原始预测工具（未改动）
  score_predictions.py          原始独立评分器（未改动）
  check_data.py                 原始数据完整性检查（未改动）
  evaluate.py                   四线程冷启动预测 + 独立评分
  verify.py                     资产哈希、接口与置换检查
  analyze.py                    冻结后的验证可靠性与配对分析
  train.py                      从训练数据重建最终选中权重
  search.py                     复现 v14 的内部参数搜索和验证融合
  training_utils.py             训练标签、标准化与基线学习器
  test_interactions.py          交互模型及梯度测试
  test_math_consistency.py       算术解析边界测试
  checkpoint/                   最终推理资产，全部六个文件均需保留
  data/                         课程原始题库、固定嵌入、许可和数据清单
  evidence/                     官方分数、资源、内部搜索、冻结与对照记录
  MANIFEST.json                 本提交文件的 SHA-256 清单
```

没有包含虚拟环境、IDE 配置、旧版模型、淘汰 checkpoint、临时图片或开发过程中的旧报告。训练和核验代码、数据与必要证据用于满足课程复现要求。

## 2. 安装环境

推荐 **Python 3.12**；原实验使用 3.12.14。固定依赖：

```text
numpy==2.1.3
scipy==1.15.3
scikit-learn==1.6.1
joblib==1.4.2
threadpoolctl==3.5.0
```

Windows PowerShell：先进入本目录，再运行：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

如果 `py -3.12` 找不到解释器，先安装 Python 3.12，或在 PyCharm 中选择本机 Python 3.12 新建环境。已有匹配环境时可直接使用，不必重复创建。

Linux/macOS：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

不需要 GPU、CUDA、PyTorch、CatBoost、API key 或下载额外预训练模型。安装依赖之后，预测与评分离线运行。

## 3. 直接复现准确率（不需要训练）

**所有命令的工作目录都是本文件所在的提交目录，不是原开发项目的 code 目录。**

Windows：

```powershell
# 数据与接口检查
.\.venv\Scripts\python.exe check_data.py
.\.venv\Scripts\python.exe verify.py

# 复现验证集与测试集，分别生成分数和资源 JSON
.\.venv\Scripts\python.exe evaluate.py --split validation --method selector.py --checkpoint checkpoint --output-dir results
.\.venv\Scripts\python.exe evaluate.py --split test --method selector.py --checkpoint checkpoint --output-dir results
```

Linux/macOS 将解释器替换为 `.venv/bin/python`。若环境已激活，也可将解释器部分替换为 `python`。

测试评分的关键字段应为：

```json
{
  "split": "test",
  "correct": 388,
  "n": 467,
  "accuracy": 83.08351177730192
}
```

新输出位于 `results/`：每个 split 有 `*-predictions.json`、`*-score.json`、`*-resources.json`。它们不覆盖 `evidence/` 内的提交记录。

`evaluate.py` 使用同一个解释器启动独立预测和评分进程，将 OMP/OpenBLAS/MKL/NumExpr 线程数限制为 4。完整耗时为资源 JSON 的 `total_seconds`，不能只报告模型内部耗时。原开发机器 AMD Ryzen 7 5800H 上完整测试为 13.48 秒；新机器和精简目录的时间会不同。课程参考 Xeon CPU 上的结果决定最终时间资格。

推理资产是 `checkpoint/` 的全部六个文件，共 **6,246,341 字节（5.957 MiB）**。不要遗漏 `checkpoint/v2/`；数据和普通源码不计入额外模型资产。

## 4. 在 PyCharm 中运行

1. 用 File → Open 打开 **MP2_Submission_v14 本身**。
2. 在 Python Interpreter 中选择 Python 3.12 环境。可新建 `.venv`，也可使用原项目已安装依赖的环境。
3. Run → Edit Configurations → + → Python，建立：

| 字段 | 内容 |
|---|---|
| Name | `MP2_Final_Test` |
| Script path | `$PROJECT_DIR$/evaluate.py` |
| Parameters | `--split test --method selector.py --checkpoint checkpoint --output-dir results` |
| Working directory | `$PROJECT_DIR$` |
| Python interpreter | 安装了 requirements.txt 的 Python 3.12 环境 |

环境变量可设为：

```text
OMP_NUM_THREADS=4;OPENBLAS_NUM_THREADS=4;MKL_NUM_THREADS=4;NUMEXPR_NUM_THREADS=4;PYTHONUNBUFFERED=1
```

选择绿色 Run 进行计时复现。不要直接运行仅定义函数的 `selector.py`。验证集配置只需把 `--split test` 改成 `--split validation`。

若 Terminal 未激活正确环境，显式使用 `.\.venv\Scripts\python.exe`。若报 `No module named sklearn`，核对运行配置中的解释器，并在该解释器中安装依赖。若找不到 `predict.py` 或 checkpoint，先核对工作目录。

## 5. 复现训练和最终搜索

### 固定配置重训

```powershell
.\.venv\Scripts\python.exe train.py --output retraining/checkpoint
.\.venv\Scripts\python.exe verify.py --checkpoint retraining/checkpoint
.\.venv\Scripts\python.exe evaluate.py --split validation --method selector.py --checkpoint retraining/checkpoint --output-dir retraining/results
```

`train.py` 仅加载训练题及参考答案。它按已选定的配置重建：C=.01 的 pairwise 线性模型、64 隐藏单元/20 轮的神经基线、共享 TF-IDF/SVD，以及 rank=32/L2=.01/40 轮的低秩交互模型。不会读取验证或测试数据，不会覆盖已提交的 `checkpoint/`。

训练统计保存为 `retraining/training_summary.json`。本机固定重训已核验；不同 BLAS/操作系统可能导致细微数值或字节差异，官方分数以交付的冻结 checkpoint 为准。

### 重放 v14 开发搜索

```powershell
.\.venv\Scripts\python.exe search.py --baseline checkpoint/v2 --output retraining/search
```

该过程与记录的最终 v14 搜索保持一致：训练集内部按题目 80/20 分层划分；TF-IDF、SVD 和标准化只拟合内部训练部分；比较 12 个线性配置、4 个低秩配置及其内部 epoch；选好后在全部训练集重训三类模型，再用正式验证集选择融合权重。共 19 次监督拟合和 4 次预处理拟合，种子均为 7506。搜索不读取测试数据。输出在 `retraining/search/`，不会改动提交 checkpoint。

早期 v2 的超参数选择和后续未胜出的探索记录见 `evidence/baseline_training.json`、`historical_tuning.json` 和报告。精简包保留最终选中基线的训练配方与最终阶段完整搜索，不附带淘汰模型和全部早期探索脚本。

不要根据测试分数反复选择重训产物；开发新方法应使用训练/验证数据。

## 6. 代码测试与新审计题库

```powershell
.\.venv\Scripts\python.exe test_interactions.py
.\.venv\Scripts\python.exe test_math_consistency.py
.\.venv\Scripts\python.exe verify.py
```

检查包括解析梯度、算术格式边界、空输入、Python 整数输出、新 ID、批处理以及候选置换。可在不重新训练的情况下评测新审计输入：

```text
python predict.py --method selector.py --checkpoint checkpoint --split audit --data-dir /path/to/audit_inputs --output results/audit-predictions.json
```

直接运行 predict.py 时，运行环境也应将上面四个线程变量设为 4。该工具不读取参考答案。模型不使用题目 ID 查表或缓存预测。

## 7. 数据、版本与实验记录

- `data/` 是课程提供的原始 train/validation/test 输入、参考答案和固定嵌入，未更改；它使本目录能离线评测和从头复现训练。
- 原始 `common.py`、`predict.py`、`score_predictions.py`、`check_data.py` 保持逐字节一致，哈希见 `evidence/official_tools.json`。
- `evidence/asset_manifest.json` 记录六个模型资产的 SHA-256；`freeze.json` 保留原模型在测试前的冻结记录。
- `evidence/decision.json`、`internal_selection.json` 记录最终网格与选择；评分、资源和配对比较 JSON 是实际实验记录。
- `MANIFEST.json` 标识这个精简提交目录的文件内容。原开发记录可能含原开发路径；精简目录的运行结果另有核验记录。
- v1/v2 的测试成绩在后续开发前已知。v3-v13 未超过 v2 时没有新的测试搜索。v14 先选择验证最优配置、冻结资产，再做一次新的正式测试评分；没有测试后调参。

数据题目来自 GSM8K，保留 `data/LICENSE.txt` 和 `data/NOTICE.txt`；候选文本与固定嵌入由课程用 Qwen3-0.6B 制备。没有引入额外预训练权重。算法实现使用 NumPy、SciPy 和 scikit-learn，报告列出课程讲义及论文参考。

## 8. 报告与正式提交

`report.pdf` 为英文黑白学术报告，涵盖任务、方法、头部训练与参数微调、消融、失败尝试、准确率、可靠性、资源和复现过程，最后附 AI 辅助声明。

发布到个人公开 GitHub 仓库时，应保留本目录、完整 checkpoint 和数据许可，并使用固定提交或 release 链接。提交字段的 Accuracy 为 `83.08351177730192`，报告链接指向 `report.pdf`，checkpoint 链接指向完整 `checkpoint/`。

姓名和学号已按本人提供的信息填写为 HE YUXIN、3036897306；个人公开仓库地址仍需本人提供。本包没有代建 GitHub issue。课程网页准备提交链接后，仍需在 GitHub 实际创建 issue 才算完成外部提交。课程截止时间是 2026 年 10 月 31 日 23:59（UTC+8）；当前准备发生在截止后，是否接受补交需由课程方确认。

课程指引：https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md

## AI 辅助声明

本项目使用 OpenAI Codex 辅助检索改进方法与相关论文，并辅助报告的组织、撰写和语言编辑。AI 还参与了训练/推理代码的实现与调试、实验执行、结果核验和分析。未将 AI 辅助范围仅描述为文字润色。模型训练和推理没有调用在线 LLM，没有使用外部训练数据或新增预训练权重。学生应理解并核查实现和结果，并对提交内容承担责任。
