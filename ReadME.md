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

