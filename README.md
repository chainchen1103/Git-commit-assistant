# Git Commit Assistant  
[中文](README_CH.md)  
Git Commit Assistant is a CLI tool that automatically analyzes your staged changes (`git add --all`) and predicts the appropriate commit type using a trained ONNX model.  

## Features  

* **Diff Processing**: Reads the staged diff and extracts statistics:  
  * Number of changed files  
  * Insertions and deletions  
  * File extension distribution  
* **ONNX Inference**: Uses ONNX Runtime (ORT 1.16.3) to classify the commit type.  
* **Flexible Options**:  
  * `--only-added-lines`: keep only added lines from the diff  
  * `--max-chars`: cap maximum diff text length  
  * `--threshold`: probability threshold for showing results  
  * `--topk`: display top-N most probable classes  