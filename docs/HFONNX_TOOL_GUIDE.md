# HFOnnxTool Guide

A command-line utility for working with Hugging Face models and converting them to ONNX format for use with Plugin.Maui.ML.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Commands](#commands)
  - [inspect](#inspect-command)
  - [fetch](#fetch-command)
  - [convert](#convert-command)
- [Common Workflows](#common-workflows)
- [Example: Biomedical NER Model](#example-biomedical-ner-model)
- [Example: Sentence Transformer Model](#example-sentence-transformer-model)
- [Troubleshooting](#troubleshooting)

## Overview

HFOnnxTool is a .NET command-line tool that simplifies working with Hugging Face models in your Plugin.Maui.ML projects. It provides three main capabilities:

1. **Inspect** - Browse files in a Hugging Face repository
2. **Fetch** - Download pre-existing ONNX models from a repository
3. **Convert** - Convert Hugging Face models to ONNX format using Python's `optimum` library

## Prerequisites

### Required for All Commands

- .NET 8.0 or later
- Internet connection (to access Hugging Face Hub)

### Required for Convert Command Only

The `convert` command requires Python and several packages to export models to ONNX:

```bash
# Python 3.8 or later
python --version

# Install required packages
pip install "optimum[exporters]" onnx onnxruntime transformers
```

**Note:** The `inspect` and `fetch` commands do NOT require Python. Only use the `convert` command if you need to export a model that doesn't already have ONNX files in its repository.

### Supported Architectures for the Convert Command

Optimum library handles the export of PyTorch models to ONNX in the exporters.onnx module. It provides classes, functions, and a command line interface to perform the export easily.

The complete list of supported architectures from Transformers, Diffusers, Timm & Sentence Transformers is available at the following link:

https://huggingface.co/docs/optimum-onnx/onnx/overview

For any unsupported architecture, detailed instructions for adding support for such architectures can be found here:

https://huggingface.co/docs/optimum-onnx/onnx/usage_guides/contribute

## Installation

### Option 1: Build from Source

```bash
# Navigate to the tool directory
cd tools/HFOnnxTool

# Build the tool
dotnet build

# Run the tool
dotnet run -- <command> [options]
```

### Option 2: Install as Global Tool (Optional)

```bash
cd tools/HFOnnxTool
dotnet pack
dotnet tool install --global --add-source ./nupkg HFOnnxTool
```

Then you can use `hfonnx` directly from anywhere:

```bash
hfonnx <command> [options]
```

## Commands

### inspect Command

List all files in a Hugging Face model repository to see what's available.

**Usage:**
```bash
hfonnx inspect --repo <repo-name>
```

**Options:**
- `--repo <REPO>` - (Required) Hugging Face repo in format `org/name` or full URL
- `--revision <REV>` - Git revision (branch/tag/sha). Default: `main`
- `--token <TOKEN>` - Optional HF access token for private repos (can also use `HF_TOKEN` env var)

**Example:**
```bash
# Inspect a sentence transformer model
hfonnx inspect --repo sentence-transformers/all-MiniLM-L6-v2

# Inspect with specific revision
hfonnx inspect --repo d4data/biomedical-ner-all --revision main
```

**Output:**
The command displays a table showing:
- File type (file/directory)
- File path
- File size
- Count of ONNX files found

### fetch Command

Download pre-existing ONNX models from a Hugging Face repository.

**Usage:**
```bash
hfonnx fetch --repo <repo-name> --output <directory>
```

**Options:**
- `--repo <REPO>` - (Required) Hugging Face repo in format `org/name` or full URL
- `--output <DIR>` - Destination directory. Default: `./downloaded`
- `--revision <REV>` - Git revision (branch/tag/sha). Default: `main`
- `--token <TOKEN>` - Optional HF access token for private repos
- `--pick-first` - Automatically pick the first ONNX file if multiple exist
- `--raw-dir <DIR>` - Optional: Copy the ONNX file directly to a MAUI `Resources/Raw` directory

**Example:**
```bash
# Download ONNX model to ./models directory
hfonnx fetch --repo sentence-transformers/all-MiniLM-L6-v2 --output ./models

# Download and copy directly to MAUI project
hfonnx fetch --repo sentence-transformers/all-MiniLM-L6-v2 \
  --output ./models \
  --raw-dir ./samples/MauiSample/Resources/Raw
```

**Interactive Selection:**
If a repository contains multiple ONNX files and `--pick-first` is not specified, you'll be prompted to select which file to download.

### convert Command

Export a Hugging Face model to ONNX format using Python's `optimum.exporters.onnx` module. Use this when the repository doesn't already contain ONNX files.

**Usage:**
```bash
hfonnx convert --repo <repo-name> --task <task-type> --output <directory>
```

**Options:**
- `--repo <REPO>` - (Required) Hugging Face repo in format `org/name` or full URL
- `--task <TASK>` - (Required) Model task type. Common values:
  - `token-classification` - For NER, POS tagging, etc.
  - `text-classification` - For sentiment analysis, categorization, etc.
  - `question-answering` - For QA models
  - `feature-extraction` - For embedding models
  - `text2text-generation` - For seq2seq models
- `--output <DIR>` - Output directory for ONNX files. Default: `./onnx-out`
- `--opset <N>` - ONNX opset version. Default: `17`
- `--python <PATH>` - Python executable path. Default: `python`
- `--revision <REV>` - Git revision. Default: `main`
- `--token <TOKEN>` - Optional HF access token for private repos
- `--skip-existing` - Skip conversion if output directory already has model.onnx
- `--maui-raw <DIR>` - Optional: Copy the ONNX files to a MAUI `Resources/Raw` directory
- `--no-precheck` - Skip Python dependency check (use if check fails incorrectly)

**Example:**
```bash
# Convert a biomedical NER model
hfonnx convert --repo d4data/biomedical-ner-all \
  --task token-classification \
  --output ./onnx-models

# Convert with specific opset and copy to MAUI project
hfonnx convert --repo d4data/biomedical-ner-all \
  --task token-classification \
  --output ./onnx-models \
  --opset 17 \
  --maui-raw ./samples/MauiSample/Resources/Raw
```

**Behind the Scenes:**
The convert command executes:
```bash
python -m optimum.exporters.onnx --model <repo> --task <task> --opset <opset> <output-dir>
```

## Common Workflows

### Workflow 1: Using a Model with Pre-existing ONNX Files

This is the **simplest workflow** when the model repository already contains ONNX files (like `sentence-transformers/all-MiniLM-L6-v2`).

```bash
# Step 1: Inspect to see what's available
hfonnx inspect --repo sentence-transformers/all-MiniLM-L6-v2

# Step 2: Download the ONNX model
hfonnx fetch --repo sentence-transformers/all-MiniLM-L6-v2 --output ./models

# Step 3: Use in your MAUI app
# Copy the model file to your project's Resources/Raw folder or load it directly
```

### Workflow 2: Converting a Model Without ONNX Files

Use this when the repository doesn't have ONNX files (like `d4data/biomedical-ner-all`).

```bash
# Step 1: Inspect to confirm no ONNX files exist
hfonnx inspect --repo d4data/biomedical-ner-all

# Step 2: Convert the model to ONNX
hfonnx convert --repo d4data/biomedical-ner-all \
  --task token-classification \
  --output ./onnx-models

# Step 3: Copy to your MAUI project
cp ./onnx-models/*.onnx ./samples/MauiSample/Resources/Raw/
```

### Workflow 3: All-in-One with MAUI Integration

Directly download or convert and copy to your MAUI project:

```bash
# For models with existing ONNX:
hfonnx fetch --repo sentence-transformers/all-MiniLM-L6-v2 \
  --raw-dir ./samples/MauiSample/Resources/Raw

# For models requiring conversion:
hfonnx convert --repo d4data/biomedical-ner-all \
  --task token-classification \
  --maui-raw ./samples/MauiSample/Resources/Raw
```

## Example: Biomedical NER Model

The [d4data/biomedical-ner-all](https://huggingface.co/d4data/biomedical-ner-all) model is a BERT-based model for Named Entity Recognition (NER) in biomedical text. It **does not** have pre-existing ONNX files, so we need to convert it.

### Step-by-Step Guide

#### 1. Install Python Dependencies

```bash
pip install "optimum[exporters]" onnx onnxruntime transformers
```

#### 2. Inspect the Repository

```bash
hfonnx inspect --repo d4data/biomedical-ner-all
```

This will show that the repository contains PyTorch model files (`pytorch_model.bin`, `config.json`, etc.) but no `.onnx` files.

#### 3. Convert to ONNX

```bash
hfonnx convert --repo d4data/biomedical-ner-all \
  --task token-classification \
  --output ./biomedical-ner-onnx
```

**What happens:**
- The tool downloads the model from Hugging Face
- Converts it to ONNX format using the `optimum` library
- Saves the ONNX files to `./biomedical-ner-onnx/`

**Expected output files:**
- `model.onnx` - The main model file
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer configuration
- Other supporting files

#### 4. Copy to Your MAUI Project

```bash
# Copy just the ONNX file
cp ./biomedical-ner-onnx/model.onnx ./samples/MauiSample/Resources/Raw/biomedical_ner.onnx

# Or copy all files for tokenizer support
cp ./biomedical-ner-onnx/*.onnx ./samples/MauiSample/Resources/Raw/
cp ./biomedical-ner-onnx/*.json ./samples/MauiSample/Resources/Raw/
cp ./biomedical-ner-onnx/vocab.txt ./samples/MauiSample/Resources/Raw/
```

#### 5. Use in Your Code

```csharp
// Load the model
await mlInfer.LoadModelFromAssetAsync("biomedical_ner.onnx");

// Use for NER tasks
// See MedicalNlpService.cs in the samples for a complete example
```

### Understanding the Task Type

For NER models, use `--task token-classification` because:
- NER assigns a label to each token in the input
- The model outputs a classification for each token position
- Common entity types: diseases, drugs, symptoms, procedures, etc.

## Example: Sentence Transformer Model

The [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model generates sentence embeddings. This model **already has** ONNX files in the repository.

### Step-by-Step Guide

#### 1. Inspect the Repository

```bash
hfonnx inspect --repo sentence-transformers/all-MiniLM-L6-v2
```

This will show that the repository already contains ONNX files in the `onnx/` directory. **No conversion needed!**

#### 2. Fetch the ONNX Model

```bash
hfonnx fetch --repo sentence-transformers/all-MiniLM-L6-v2 --output ./models
```

**What happens:**
- The tool lists available ONNX files
- If multiple files exist, you can select which one to download
- The selected file is downloaded to `./models/`

#### 3. Copy to Your MAUI Project

```bash
cp ./models/*.onnx ./samples/MauiSample/Resources/Raw/sentence_encoder.onnx
```

Or use the `--raw-dir` option to copy directly:

```bash
hfonnx fetch --repo sentence-transformers/all-MiniLM-L6-v2 \
  --output ./models \
  --raw-dir ./samples/MauiSample/Resources/Raw
```

#### 4. Use in Your Code

```csharp
// Load the model
await mlInfer.LoadModelFromAssetAsync("sentence_encoder.onnx");

// Generate embeddings
var inputs = new Dictionary<string, Tensor<long>>
{
    ["input_ids"] = tokenIds,
    ["attention_mask"] = attentionMask
};

var outputs = await mlInfer.RunInferenceLongInputsAsync(inputs);
var embeddings = outputs["last_hidden_state"]; // or "pooler_output"
```

### Why This Model Has ONNX Files

Sentence-transformers models often include pre-converted ONNX files because:
- They're commonly used for production deployments
- ONNX Runtime provides excellent performance for these models
- The maintainers want to make it easy to use with ONNX

## Troubleshooting

### Python Not Found (convert command)

**Error:** `Failed to start python process`

**Solution:**
1. Ensure Python is installed: `python --version`
2. Specify the full path: `--python /usr/bin/python3`
3. On Windows, use: `--python python.exe` or `--python C:\Python39\python.exe`

### Missing Python Packages (convert command)

**Error:** `Missing python packages: optimum, transformers`

**Solution:**
```bash
pip install "optimum[exporters]" onnx onnxruntime transformers
```

If the pre-check fails incorrectly, use `--no-precheck` to skip it.

### No ONNX Files Found (fetch command)

**Error:** `No ONNX files present. Use 'convert' command to export.`

**Solution:**
The repository doesn't have pre-existing ONNX files. Use the `convert` command instead:

```bash
hfonnx convert --repo <repo-name> --task <task-type> --output ./onnx
```

### Wrong Task Type (convert command)

**Error:** Model conversion fails or produces incorrect outputs

**Solution:**
Choose the correct task type for your model:
- Check the model card on Hugging Face for the intended task
- Common mappings:
  - NER/Entity Recognition → `token-classification`
  - Sentiment Analysis → `text-classification`
  - Embeddings → `feature-extraction`
  - Q&A → `question-answering`

### Access Denied / Private Repo

**Error:** `HF API error: 401 Unauthorized`

**Solution:**
1. Create a Hugging Face access token: https://huggingface.co/settings/tokens
2. Use the token:
   ```bash
   hfonnx inspect --repo <repo> --token <your-token>
   # Or set environment variable
   export HF_TOKEN=<your-token>
   hfonnx inspect --repo <repo>
   ```

### Model Too Large

**Note:** Some models are very large (several GB). Ensure you have:
- Sufficient disk space
- A stable internet connection
- Patience - large models take time to download and convert

### Opset Version Issues

If you encounter errors related to ONNX operators:

```bash
# Try a different opset version
hfonnx convert --repo <repo> --task <task> --opset 14 --output ./onnx
```

Common opset versions:
- `17` - Latest, best compatibility (default)
- `14` - Good compatibility with older runtimes
- `11` - For legacy systems

## Additional Resources

- [Hugging Face Model Hub](https://huggingface.co/models)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [Optimum Documentation](https://huggingface.co/docs/optimum/index)
- [Plugin.Maui.ML Documentation](../README.md)
- [Platform Backend Guide](./PLATFORM_BACKENDS.md)

## Getting Help

If you encounter issues not covered here:

1. Check the [main README](../README.md) for general Plugin.Maui.ML usage
2. Open an issue on [GitHub](https://github.com/Clinical-Support-Systems/Plugin.Maui.ML/issues)
3. Include:
   - The exact command you ran
   - The full error message
   - Your Python version (if using `convert`)
   - Your .NET version
