
Environment Prepare: 
```Shell
git clone https://github.com/YYZhang2025/LLM-Architecture.git
cd LLM-Architecture
```


Install dependencies, here we are using `uv`
```Shell
wget -qO- https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv --version

uv sync 
source .venv/bin/activate
```


It will automatically create a virtual environment and install the required dependencies.

And we also need download the `tinystories` dataset through
```Shell
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