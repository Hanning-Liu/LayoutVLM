# `eval_benchmark_gen` 使用说明

本目录提供一个生成脚本，用 `eval-dataset/` 里的 `jsonl`（读取每行的 `text`）+ `eval-dataset/extracted_rooms.json`（读取房间 `exterior` 顶点）生成 **`benchmark_tasks/bedroom/bedroom_0.json` 同结构**的任务 JSON。

生成脚本：
- `eval_benchmark_gen/build_benchmark_from_eval_dataset.py`

## 输出 JSON 格式

每个输出文件为一个 JSON，包含字段：
- **`task_description`**：来自 jsonl 每行的 `text`
- **`layout_criteria`**：使用 `prompts-lhn/prompts.py` 中 `PROMPT_CRITERIA` 调用 LLM 生成
- **`boundary`**
  - `floor_vertices`: 来自 `extracted_rooms.json` 的 `room.exterior`（2D）转为 `[[x,y,0], ...]`，并平移到 `min(x)=0, min(y)=0`
  - **按 jsonl 行序绑定房间**：第 1 条非空行使用 `rooms[room_key][room_index]`，第 2 条非空行使用 `rooms[room_key][room_index+1]`，以此类推（跳过空行不计入行序）。`room_key` 由 `--eval-subdir` / `--extracted-room-key` 决定。
  - `wall_height`: CLI 参数 `--wall-height`（默认 2.5）
- **`assets`**：形如 `{"<uid>-<idx>": {}}` 的字典（UID 来自 `--asset-dir` 下扫描到的 `data.json`）

## 运行前准备

### 1) Python

使用 `python3` 运行。

### 2) 模型 API Key

脚本通过 OpenAI-compatible 的 DashScope/Qwen 接口发起请求（复用仓库里的 `src/layoutvlm/qwen_dashscope_client.py`）。

你需要提供 API Key（二选一）：
- 环境变量 **`DASHSCOPE_API_KEY`**（推荐）
- 或环境变量 **`OPENAI_API_KEY`**

可选：在仓库根目录放一个 `.env`，脚本会自动尝试加载。

### 3) 本地资产目录

默认使用 `objaverse_processed/`（可用 `--asset-dir` 覆盖）。目录布局与 `main.py` 的 `prepare_task_assets` 一致：
- `<asset_dir>/<uid>/data.json`
- `<asset_dir>/test_asset_dir/<uid>/data.json`

## 快速开始

在仓库根目录运行：

### 按 eval 子目录生成（推荐）

```bash
python3 -m eval_benchmark_gen.build_benchmark_from_eval_dataset \
  --eval-subdir bathroom \
  --room-index 0 \
  --out-dir out/bench_bathroom \
  --candidates-cache out/_candidates_cache.json \
  --llm-timeout-s 60 \
  --skip-missing \
  --skip-existing \
  --workers 4
```

### 直接指定 jsonl 文件生成

```bash
python3 -m eval_benchmark_gen.build_benchmark_from_eval_dataset \
  --jsonl-path eval-dataset/kitchen/kitchen.jsonl \
  --room-index 2 \
  --out-dir out/bench_kitchen \
  --candidates-cache out/_candidates_cache.json \
  --llm-timeout-s 60 \
  --skip-missing \
  --skip-existing \
  --workers 4
```

### 只验证几何与输出结构（不调用 LLM、不选 UID）

```bash
python3 -m eval_benchmark_gen.build_benchmark_from_eval_dataset \
  --eval-subdir livingroom \
  --room-index 0 \
  --out-dir out/dry_run \
  --dry-run --limit 3
```

### 一次性跑完 eval-dataset 所有子目录（示例）

```bash
cd /home/ubuntu/LayoutVLM

CACHE="out/_candidates_cache.json"

# 自动遍历 eval-dataset 下所有 <subdir>/<subdir>.jsonl
for jsonl in eval-dataset/*/*.jsonl; do
  sub="$(basename "$(dirname "$jsonl")")"
  python3 -m eval_benchmark_gen.build_benchmark_from_eval_dataset \
    --eval-subdir "$sub" \
    --room-index 0 \
    --out-dir "out/bench_${sub}" \
    --candidates-cache "$CACHE" \
    --llm-timeout-s 60 \
    --skip-missing \
    --skip-existing \
    --workers 4
done
```

## 常用参数说明

- **输入相关**
  - `--eval-subdir`: `eval-dataset/` 下的子目录名（会自动使用 `eval-dataset/<subdir>/<subdir>.jsonl`）
  - `--jsonl-path`: 直接指定 `.jsonl` 路径（与 `--eval-subdir` 二选一）
  - `--limit`: 仅处理前 **N 条非空** jsonl 记录（便于小规模测试）

- **房间边界相关**
  - `--extracted-rooms`: `extracted_rooms.json` 路径（默认 `eval-dataset/extracted_rooms.json`）
  - `--room-index`: **起始房间下标**：jsonl 中第 1 条非空记录对应 `rooms[room_key][room_index]`，第 2 条对应 `rooms[room_key][room_index+1]`，依此类推。若「非空行数 + 起始下标」超出该房型在 `extracted_rooms` 中的房间数量，脚本会报错退出。
  - `--extracted-room-key`: 覆盖 `rooms` 的 key（默认根据 `--eval-subdir` 映射；常见为与目录同名，另支持 `esports`、`childrenroom` 等，未知目录需显式传该参数）
  - `--wall-height`: 输出 JSON 的 `boundary.wall_height`（默认 2.5）

- **资产/UID 相关**
  - `--asset-dir`: Objaverse processed 资产根目录（默认 `objaverse_processed`）
  - `--candidates-cache`: 候选资产缓存文件（JSON）。第一次运行会扫描 `--asset-dir` 并写入缓存；后续运行直接加载缓存，避免重复扫描（推荐给循环跑多个子目录时使用）。
  - `--skip-missing`: 某个家具短语找不到候选或 Judge 全拒时，跳过该类（否则直接报错停止）
  - **`--skip-existing`（断点续跑）**：若 `--out-dir/<id>.json` 已存在，则跳过该行（不调 LLM、不覆盖）。**仍会占用一行序号**：下一行仍对应下一个 `rooms[room_key][…]`，与 jsonl 顺序一致。
  - `--max-dim-ratio / --min-dim-ratio / --max-total-area-ratio`: 尺寸启发式过滤参数（用于避免明显过大/过小的资产）

- **LLM 相关**
  - `--text-model`: 用于 `layout_criteria`、`furniture list`、以及无图时的 `judge`（默认优先读环境变量，否则 `qwen3.6-plus`）
  - `--vision-model`: 有预览图时用于 `PROMPT_FUR_JUDGE` 的视觉模型（默认 `qwen-vl-max`，也可用环境变量 `DASHSCOPE_VISION_MODEL`）
  - `--judge-no-image {text,skip}`: 如果 UID 目录找不到预览图，使用纯文本 judge 或直接跳过候选
  - `--llm-timeout-s`: 单次 LLM 调用的超时时间（秒）。网络/接口卡住时可避免“无输出一直挂住”。
  - **`--workers` / `-j`**：并行**进程**数（默认 `1` 即串行）。大于 1 时，多条 jsonl 记录在多个进程中并发调用 LLM；每个子进程会单独创建 API 客户端，并在内存中持有一份候选列表副本（与进程数成正比）。注意 DashScope/Qwen 可能有 **RPM/TPM 限流**，过大并发易触发 429 或失败；建议从 2～4 逐步加大。
  - `--dashscope-api-key / --dashscope-base-url`: 显式指定 Key / Base URL（否则用环境变量）

## 输出文件命名

脚本对 jsonl 每行读取 `id` 字段，输出为：
- `<out-dir>/<id>.json`

例如 `bathroom.jsonl` 的第一行 `id=bath_001` 会输出 `out/bench_bathroom/bath_001.json`。

## 注意事项

- **成本与速度**：每行至少调用 2 次 LLM（criteria + fur-list），再对候选 UID 做多次 judge；建议先用 `--limit` 小规模跑通。需要提速时可加大 `--workers`（进程数），但以服务商限流为准；并行不会减少「单行」内部的 judge 次数。
- **房间数量**：`extracted_rooms` 里该 `room_key` 的房间个数必须 **≥** 「非空 jsonl 行数 + `--room-index`」所需的末尾下标，否则会报错（不做循环取模）。
- **断点续跑**：批量生成中断后，用**相同** `--out-dir` 与 `--room-index` 加上 `--skip-existing` 重跑即可跳过已生成的 `<id>.json`。若要强制重做某几条，请删除对应 JSON 文件或换新的输出目录（脚本不提供 `--force`）。
- **预览图缺失**：如果你的 `objaverse_processed` 里多数 UID 没有任何 `.jpg/.png`，建议先用 `--judge-no-image text`，或配合 `--skip-missing` 降低失败率。

