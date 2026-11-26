# BoardGame Ark · 桌游方舟智能推荐系统

> **在线体验（Live Demo）**：👉 [https://boardgame-ark.onrender.com](https://boardgame-ark.onrender.com) *（Render 免费实例，首次冷启动约需 20–40 秒唤醒）*  
> **NUS School of Computing Summer Workshop (SWS) · 团队第 2 名 / 15 组**  
> 基于 **29,506 款** BoardGameGeek (BGG) 桌游数据与混合推荐算法构建的多步骤引导式个性化推荐 Web 应用。

## 项目亮点与我的分工

本项目完成于新加坡国立大学（NUS）计算机学院暑期项目（4 人团队，全英文调研与结项答辩）。

- **个人职责**：负责**产品冷启动流程设计（4 步引导 Wizard）、游戏机制抽象映射、特征与排序推荐逻辑设计以及前端交互页面落地**。
- **团队协作**：原始数据清洗、部分协同过滤优化、离线 NDCG 评估脚本与结项海报由队友协作完成。

---

## 核心产品与策略设计

### 1. 冷启动体验重构（100+ 专业机制 → 8 大直觉维度）
- 面向新手看不懂 BGG 100+ 晦涩桌游机制的痛点，将其抽象映射为**策略规划、团队合作、卡牌收集、领土争夺、建造大师、角色扮演、幸运冒险、反应社交**等直觉体验维度；
- 设计 **4 步引导流（经典游戏偏好 → 机制偏好 → 场景领域 → 人数/时长/复杂度）** 快速构建显式用户画像，补齐账户收藏留存与封面加载降级兜底。

### 2. 混合推荐与双层兜底筛选策略
- **284 维特征空间 + 混合策略**：结合 **60% 内容过滤（Cosine Similarity）+ 40% 协同过滤（Item-Based CF）**，并引入 `0.5×相似度 + 0.3×评分 + 0.2×流行度` 加权排序。
- **双层兜底筛选解决“推荐位空缺”**：高门槛过滤（评分人数 ≥ 500）曾导致“猜你喜欢”仅召回 7 款、填不满 12 个页面坑位；通过扩展 4 倍候选池 + 双层兜底（优先 500+、不足以 100+ 去重补齐），稳定输出满额 12 条高质推荐。
- **云端轻量化工程优化**：将 399 MB 原始用户评分数据离线预计算为 Top-K 稀疏协同过滤索引（`data/cf_topk.json.gz`），保留完整混合推荐能力的同时实现 **< 2 秒冷启动与低内存云端部署**。

---

## 离线评估结果（团队评估报告）

详见 [NDCG_Evaluation_Report.md](./NDCG_Evaluation_Report.md)：

| 推荐策略 | NDCG@12 | Top-12 命中率 | 相对混合策略差距 |
|---|---:|---:|---|
| **混合推荐（60% 内容 + 40% 协同）** | **0.7267** | **91.7%** | Baseline（最优） |
| 纯协同过滤 | 0.6688 | 75.0% | 混合推荐 +8.7% |
| 纯内容过滤 | 0.6312 | 66.7% | 混合推荐 +15.2% |

---

## 技术栈与云端部署

`Python 3.11` · `Flask` · `Scikit-learn` · `Pandas / NumPy / SciPy` · `SQLite / SQLAlchemy` · `HTML / CSS / JS` · `Docker` · `Gunicorn`

### 本地一键运行
```bash
pip install -r requirements.txt
python app.py
# 访问 http://localhost:8080
```

### 一键云端部署（支持 Hugging Face Spaces Docker / Render）
- **Docker / Hugging Face Spaces**：仓库根目录已内置标准 `Dockerfile`（默认监听 `PORT=7860`），推送至 Docker Space 即可自动构建上线。
- **Render**：已内置 `render.yaml` Blueprint，连接本 GitHub 仓库即可一键部署。
