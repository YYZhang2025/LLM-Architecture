- [Baseline Model](#baseline-model)



<h1 align="center">LLM: Architecture</h1>
<p align="center"><b>Implement different SOTA LLM Architecture using PyTorch</b></p>

I write a note about corresponding to the code structure, please check [LLM Part1: Architecture](https://yuyang.info/posts/Blogs/LLM/LLM-Architecture/post.html).

<h2 align="center">Environment Prepare</h2>
First, clone the repository and navigate into it:

```Shell
git clone https://github.com/YYZhang2025/LLM-Architecture.git
cd LLM-Architecture
```

Then, we need to install dependencies, here we are using `uv`

```Shell
wget -qO- https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

uv sync 
source .venv/bin/activate
uv pip install -e .
```

It will automatically create a virtual environment and install the required dependencies.
And we also need download the `tinystories` dataset through

```Shell
chmod +x download.sh
./download.sh
```

Than, we also need train a tokenizer and pre-tokenize it and save as numpy array to speed up training.
```Shell
python pre_tokenize.py
```

After above code, you should have following files in the `data` directory:
```Text 
📂 data  
└── 📂 tinystories  
    ├── 📄 tinystories_eval_tokens.npy  
    ├── 📄 tinystories_train_tokens.npy  
    ├── 📘 TinyStoriesV2-GPT4-train.txt  
    ├── 📘 TinyStoriesV2-GPT4-valid.txt  
    └── 🧩 tokenizer-bpe.json  
```

<h2 align="center">About This Repository</h2>

This repository implements different SOTA LLM architectures using PyTorch. The current implemented models include:
- Position Encoding:
  - Learned Positional Encoding
  - Sinusoidal Positional Encoding
  - Relative Positional Encoding 
  - Rotary Positional Encoding (RoPE)
- Attention Mechanisms:
  - Standard Multi-Head Attention
  - Multi-Query Attention / Grouped-Query Attention



<h2 align="center">Experiments</h2>

The dataset we are using is [TinyStoriesV2-GPT4](https://huggingface.co/datasets/roneneldan/TinyStories). We have pre-tokenized the dataset and saved it as numpy arrays for faster loading during training. The tokenizer used is a BPE tokenizer trained using the `tokenizers` library. For the tokenization part, please refer to the `pre_tokenize.py` script.

All the model defined in the `llm/models` directory. 


```Text
📂 llm
├── 📂 models
│   ├── 📂 baseline.py
```

Each model has a corresponding training script in the `train_scripts` directory.
```Text
📂 train_scripts
├── 📄 train_baseline.py
```


<h3 align="center">Baseline Model</h3>
