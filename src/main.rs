//! gca — Git Commit Assistant
//!
//! Automatically analyzes the staged changes in the current Git repository,
//! and uses an ONNX Runtime inference model to predict the commit message type
//! (feat/fix/docs/...).
//!
//! Features:
//!   - Automatically runs `git add --all` to collect all staged diffs
//!   - Optionally analyze only added lines, with maximum diff length limit
//!   - Supports models exported from sklearn → ONNX (with zipmap=False), outputs class probabilities
//!   - Reads `labels.txt` to map probabilities to class names
//!   - Configurable probability threshold, or show top-K most likely classes
//!
//! Usage examples:
//! ```bash
//! # Inside a repo directory, run directly:
//! gca commit --model model.onnx
//!
//! # Only analyze added lines, show top 3 most likely classes:
//! gca commit --only-added-lines --topk 3
//!
//! # Use a custom labels.txt
//! gca commit --labels custom_labels.txt
//!
//! # Set probability threshold (only output classes ≥ 0.7)
//! gca commit --threshold 0.7
//! ```
//!
//! Example output:
//! ```text
//! === gca commit: inference result ===
//! files_changed: 3, additions: 25, deletions: 10, add_div: 2.50
//!   82.341%  feat
//!   12.772%  refactor
//!    4.887%  docs
//! ```
use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use git2::{DiffFormat, IndexAddOption, Repository};
use ndarray::{arr2, Array2, CowArray};
use ort::environment::Environment;
use ort::session::{Session, SessionBuilder};
use ort::value::Value;
use ort::GraphOptimizationLevel;
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Parser)]
#[command(name = "gca", author, version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Commit {
        #[arg(long, default_value = "model.onnx")]
        model: PathBuf,
        #[arg(long, default_value = "")]
        labels: String,
        #[arg(long)]
        only_added_lines: bool,
        #[arg(long, default_value_t = 20000)]
        max_chars: usize,
        #[arg(long, default_value_t = 0.5)]
        threshold: f32,
        #[arg(long, default_value_t = 0)]
        topk: usize,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Commit {
            model,
            labels,
            only_added_lines,
            max_chars,
            threshold,
            topk,
        } => run_commit(model, labels, only_added_lines, max_chars, threshold, topk),
    }
}

fn run_commit(
    model_path: PathBuf,
    labels_path_opt: String,
    only_added: bool,
    max_chars: usize,
    threshold: f32,
    topk: usize,
) -> Result<()> {
    let repo = Repository::discover(".").context("Not a Git repository")?;
    {
        let mut index = repo.index()?;
        index.add_all(["*"].iter(), IndexAddOption::DEFAULT, None)?;
        index.write()?;
    }

    let (diff_text_full, files_changed, additions, deletions, top_exts) =
        staged_diff_and_stats(&repo)?;

    let diff_proc = {
        let s = if only_added {
            extract_added_lines(&diff_text_full)
        } else {
            diff_text_full
        };
        cap_text(&s, max_chars).to_string()
    };
    let add_div: f32 = (additions as f32) / ((deletions as f32) + 1.0);
    let exts_proc = if top_exts.is_empty() {
        String::new()
    } else {
        top_exts.join(" ")
    };

    let env: Arc<Environment> = Arc::new(
        Environment::builder()
            .with_name("gca-commit")
            .build()
            .context("建立 ORT Environment 失敗")?,
    );
    let session: Session = SessionBuilder::new(&env)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file(&model_path)
        .with_context(|| format!("載入 ONNX 失敗: {}", model_path.display()))?;

    let diff_cow = CowArray::from(Array2::from_shape_vec((1, 1), vec![diff_proc.clone()])?.into_dyn());
    let exts_cow = CowArray::from(Array2::from_shape_vec((1, 1), vec![exts_proc.clone()])?.into_dyn());
    let files_cow = CowArray::from(arr2(&[[files_changed as f32]]).into_dyn());
    let add_div_cow = CowArray::from(arr2(&[[add_div]]).into_dyn());

    let diff_tensor = Value::from_array(session.allocator(), &diff_cow)?;
    let exts_tensor = Value::from_array(session.allocator(), &exts_cow)?;
    let files_tensor = Value::from_array(session.allocator(), &files_cow)?;
    let add_div_tensor = Value::from_array(session.allocator(), &add_div_cow)?;

    let mut diff_v = Some(diff_tensor);
    let mut exts_v = Some(exts_tensor);
    let mut files_v = Some(files_tensor);
    let mut add_div_v = Some(add_div_tensor);

    let mut input_values: Vec<Value> = Vec::with_capacity(session.inputs.len());
    for spec in &session.inputs {
        match spec.name.as_str() {
            "diff_proc" => input_values.push(
                diff_v.take().ok_or_else(|| anyhow!("Repeatedly  input: diff_proc"))?
            ),
            "exts_proc" => input_values.push(
                exts_v.take().ok_or_else(|| anyhow!("Repeatedly  input: exts_proc"))?
            ),
            "files_changed" => input_values.push(
                files_v.take().ok_or_else(|| anyhow!("Repeatedly  input: files_changed"))?
            ),
            "add_div" => input_values.push(
                add_div_v.take().ok_or_else(|| anyhow!("Repeatedly  input: add_div"))?
            ),
            other => return Err(anyhow!("Unknown model input: {other}")),
        }
    }

    let outputs = session.run(input_values)?;
    let probs = find_probs_2d(&outputs)
        .ok_or_else(|| anyhow!("Can't find floating point probability output for [1, C]"))?;

    let labels_path = if labels_path_opt.trim().is_empty() {
        model_path.parent().unwrap_or(Path::new(".")).join("labels.txt")
    } else {
        PathBuf::from(labels_path_opt)
    };
    let labels = load_labels(&labels_path)?;
    if labels.len() != probs.len_of(ndarray::Axis(1)) {
        eprintln!(
            "Warning: The number of labels ({}) does not match the model output dimension ({})",
            labels.len(),
            probs.len_of(ndarray::Axis(1))
        );
    }

    println!("=== gca commit: inference result ===");
    let row = probs.index_axis(ndarray::Axis(0), 0);
    let mut scored: Vec<(String, f32)> = (0..row.len())
        .map(|i| (labels.get(i).cloned().unwrap_or_else(|| format!("class_{}", i)), row[i]))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!(
        "files_changed: {}, additions: {}, deletions: {}, add_div: {:.3}",
        files_changed, additions, deletions, add_div
    );

    if topk > 0 {
        for (name, p) in scored.into_iter().take(topk) {
            println!("{:>8.3}%  {}", p * 100.0, name);
        }
    } else {
        for (name, p) in scored {
            if p >= threshold {
                println!("{:>8.3}%  {}", p * 100.0, name);
            }
        }
    }

    Ok(())
}

/// Extract staged diff text and statistics
fn staged_diff_and_stats(
    repo: &Repository,
) -> Result<(String, usize, usize, usize, Vec<String>)> {
    let head_tree = repo
        .head()
        .ok()
        .and_then(|h| h.target())
        .and_then(|oid| repo.find_commit(oid).ok())
        .and_then(|c| c.tree().ok());

    let mut idx = repo.index()?;
    let tree = head_tree.as_ref();
    let diff = repo.diff_tree_to_index(tree, Some(&mut idx), None)?;

    let mut files_changed = 0usize;
    let mut additions = 0usize;
    let mut deletions = 0usize;

    let mut ext_count: BTreeMap<String, usize> = BTreeMap::new();

    if let Ok(st) = diff.stats() {
        files_changed = st.files_changed() as usize;
        additions = st.insertions() as usize;
        deletions = st.deletions() as usize;
    }

    let mut buf = Vec::new();
    diff.print(DiffFormat::Patch, |_delta, _hunk, line| {
        buf.extend_from_slice(line.content());
        true
    })?;

    diff.foreach(
        &mut |delta, _| {
            if let Some(path) = delta.new_file().path().or_else(|| delta.old_file().path()) {
                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    let e = ext.to_lowercase();
                    *ext_count.entry(e).or_insert(0) += 1;
                }
            }
            true
        },
        None,
        None,
        None,
    )?;

    let mut exts: Vec<_> = ext_count.into_iter().collect();
    exts.sort_by(|a, b| b.1.cmp(&a.1));
    let top_exts = exts.into_iter().map(|(e, _)| e).take(5).collect::<Vec<_>>();

    let diff_text = String::from_utf8(buf).unwrap_or_default();
    Ok((diff_text, files_changed, additions, deletions, top_exts))
}

fn extract_added_lines(diff_text: &str) -> String {
    let mut out = String::new();
    for line in diff_text.lines() {
        if line.starts_with("+++ ") {
            continue;
        }
        if line.starts_with('+') {
            out.push_str(&line[1..]);
            out.push('\n');
        }
    }
    out
}

fn cap_text(s: &str, max_chars: usize) -> &str {
    if max_chars > 0 && s.len() > max_chars {
        &s[..max_chars]
    } else {
        s
    }
}

/// Extract a [1, C] f32 probability matrix from ORT outputs
/// (v1: use `try_extract::<f32>()` + `into_dimensionality`)

fn find_probs_2d(outputs: &[Value]) -> Option<ndarray::Array2<f32>> {
    for v in outputs {
        if let Ok(tensor) = v.try_extract::<f32>() {
            let view = tensor.view(); // ArrayViewD<'_, f32>
            if let Ok(view2) = view.clone().into_dimensionality::<ndarray::Ix2>() {
                if view2.shape().len() == 2 && view2.shape()[0] == 1 && view2.shape()[1] >= 2 {
                    return Some(ndarray::Array2::<f32>::from(view2.to_owned()));
                }
            }
        }
    }
    None
}

fn load_labels(path: &Path) -> Result<Vec<String>> {
    let s = fs::read_to_string(path)
        .with_context(|| format!("讀取 labels 失敗: {}", path.display()))?;
    let labels = s
        .lines()
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>();
    Ok(labels)
}
