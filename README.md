# PrivBayes on UCI Adult — NTUST 114-2 資訊安全期中

對 UCI Adult 資料集套用 PrivBayes 產生差分隱私 (Differential Privacy) 合成資料，
比較不同 ε 下的合成品質與下游機器學習效用，並與 HW1 (k-anonymity)、HW2 (DP-SGD) 對照。

## 環境需求

- Python 3.11
- [uv](https://docs.astral.sh/uv/) 套件管理
- 主要依賴：`datasynthesizer==0.1.13`、`pandas`、`scikit-learn`、`matplotlib`、`seaborn`

## 安裝

```bash
uv sync
```

## 執行流程

```bash
# 1. 從 HW1 的 adult.data 清洗資料 → data/original/original_clean.csv
uv run python src/prepare_data.py

# 2. 對 ε ∈ {0.1, 0.5, 1, 2, 4} × seed ∈ {42, 123, 456} 跑 PrivBayes
#    產出 15 個合成 CSV (約 6 分鐘)
uv run python src/run_privbayes.py

# 3. 算 TVD / 相關係數差 / TSTR (LR + RF) → utility_summary.csv + 圖表
uv run python src/evaluate.py

# 4. 與 HW1 K-anonymity 量化比較 + 寫 HW2 階段對照表
uv run python src/compare.py
```

`prepare_data.py` 會去讀 `../ntust-114-2-security-hw1/data/adult.data`；如果路徑不同
請改 `src/prepare_data.py` 裡的 `HW1_DATA_PATH`。

## 產出結構

```
data/
  original/original_clean.csv             # 30,162 列 × 10 欄
  synthetic/synthetic_eps{ε}_seed{S}.csv  # 15 個合成資料
results/
  metrics/
    utility_summary.csv                   # 主結果表 (mean ± std)
    utility_detail.json                   # 含每欄 TVD
    description_eps{ε}_seed{S}.json       # PrivBayes 學到的貝氏網路
    privbayes_timing.json                 # 每次跑的耗時
  figures/
    utility_vs_epsilon.png                # 主結果圖 (4 子圖)
    correlation_eps{ε}.png                # 5 張並排相關矩陣熱圖
    marginal_eps{ε}/marginals_eps{ε}.png  # 5 個邊際分布圖資料夾
    hw1_vs_privbayes.png                  # HW1 vs PrivBayes 並排比較
  comparisons/
    hw1_vs_privbayes.csv                  # 同資料同指標的對照表
    hw1_kanon_info_loss.csv               # HW1 自身的 C_DM/C_AVG
    hw2_dp_stages.md                      # HW2 階段對照 (含量化對齊)
    hw2_dp_stages.csv
    hw2_quant_alignment.csv
    comparison_detail.json
```

## 設定說明

- 欄位裁減：原本 14 欄裁到 10 欄。保留 5 個數值準識別碼（與 HW1 相同：`age`,
  `education-num`, `hours-per-week`, `capital-gain`, `capital-loss`）+ 4 個重要類別欄位
  （`workclass`, `education`, `marital-status`, `sex`）+ 敏感屬性 `income`。
- 貝氏網路 degree `k=2`：每個節點最多 2 個父節點。
- ε 拆分：DataSynthesizer 0.1.13 內部對結構與條件分布各拿 ε/2（5:5 拆分），
  與 Zhang et al. 2017 PrivBayes (TODS) 主實驗一致。
- 所有實驗 seed 固定為 `{42, 123, 456}`，結果取 mean ± std。

## 限制

1. 欄位從 14 裁到 10（捨棄 `fnlwgt`, `occupation`, `relationship`, `race`,
   `native-country`），因為這些欄位在 PrivBayes 中要嘛無預測意義（fnlwgt），
   要嘛類別數太多會稀釋條件機率表。
2. `k=2` 貝氏網路無法捕捉高階關聯，即使 ε=4，TSTR Random Forest 準確率仍比真實
   訓練（TRTR）差約 5.7 pp。
3. ε=0.1 時合成資料效用明顯崩潰（avg TVD ≈ 0.43），這是 DP 在極端低預算下的預期
   現象，並非實作錯誤。
4. 與 HW2 (DP-SGD on MNIST) 因任務不同，只能比 ε-效用曲線形狀，不能比絕對準確率。
