# Git commit assistant  
[English](README.md)  
Git commit assistant 是 CLI 工具，它會自動分析已暫存的變更（`git add --all`），並使用經過訓練的 ONNX 模型預測適當的提交類型。  

## Feature  

* **差異處理**：讀取已暫存的差異並擷取統計資料：  
    * 更改的文件數量  
    * 插入和刪除  
    * 檔案副檔名分佈  
* **ONNX 推理**：使用 ONNX 執行時間 (ORT 1.16.3) 對提交類型進行分類。  
* **可選項**：  
    * `--only-added-lines`：僅保留差異中新增的行  
    * `--max-chars`：限制差異文字的最大長度  
    * `--threshold`：顯示結果的機率閾值  
    * `--topk`：顯示前 N 個最可能的類別  