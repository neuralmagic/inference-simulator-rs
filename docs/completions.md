# Shell completions

`vllm-vcr completions <shell>` prints a completion script to stdout for `bash`,
`zsh`, `fish`, `powershell`, or `elvish`. Install it wherever your shell loads
completions, for example:

```bash
# fish
vllm-vcr completions fish > ~/.config/fish/completions/vllm-vcr.fish

# bash (current shell)
source <(vllm-vcr completions bash)

# zsh (into a directory on $fpath)
vllm-vcr completions zsh > ~/.zfunc/_vllm-vcr
```

The script is generated from the live command tree, so it always matches the
subcommands and flags of the binary you ran it from.
