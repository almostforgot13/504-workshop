# A 的 15 分钟预测源码

`forecast_model.py` 来自 2026-10-12 用户提供的 `aic` 目录。它读取仓库 `data/processed/A_15min_virtual_2017_2019.csv`，以 2017 年训练、2018 年验证、2019 年测试，训练 q10、q50、q90 三个 LightGBM 模型，并输出每 15 分钟发起、未来 96 步的预测。

依赖：Python、NumPy、pandas、LightGBM、Matplotlib。原始提供目录没有依赖锁定文件、训练完成的模型权重、`metrics.json` 或全年预测 CSV；因此此处仅为可审查的源码，尚未完成独立重训验收。

在此目录运行 `python forecast_model.py train` 会训练并生成模型、指标、预测和图。训练输出会写入本目录，提交前应先核对结果及文件体积。B 的 48 小时交付已经包含对应的短预测片段，位于 `../../dispatch/b_delivery_2026-10-05/data/A_48h_forecasts.csv`。
