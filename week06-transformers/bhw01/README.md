# Big homework 1. BLMs (Boutique LMs)

<p align="center"> <img src="./tiny-llama.jpg" style="width: 30%;"> </p>
<p align="center">This tiny lama is real</p>

## Task description

You need to implement and train a small (but strong) language model on
a high quality synthetic dataset from the
[TinyStories](https://arxiv.org/abs/2305.07759) paper **from scratch**
(no code-templates, no trainers, no checkpoints).

Then you need to write a detailed report about your reproduction with
implementation details and at least two comparisons (qualitative and
quantitative) of your model with the
[GPT2-XL](https://huggingface.co/gpt2-xl) in story generation,
demonstrating your model is better.

## Data

The dataset can be loaded from HuggingFace:

```bash
wget --quiet --show-progress "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
```

## Code & Advice

> [!IMPORTANT]
> The only dependencies you can use to implement and
> train your BLM in this homework are in `env.yaml` example below. If
> your code meaningfully crashes in this env we'll lower the grade
> to 0.

```yaml
# env.yaml
name: b
channels:
- https://conda.anaconda.org/nvidia
- https://conda.anaconda.org/conda-forge
- nodefaults
dependencies:
- pytorch::pytorch=2.1
- pytorch::pytorch-cuda=12.1
- sentencepiece
- requests
- wandb
- tensorboard
- tqdm
```

For evaluation and comparison you can use [mlc](https://llm.mlc.ai/),
[llama.cpp](https://github.com/ggerganov/llama.cpp),
[transformers](https://huggingface.co/docs/transformers/index) (only
for GPT2-XL inference) or any closed model API.

Some advice:
- We recommend you to start (and keep everything simple). With the
default pre-norm transformer decoder. Don't overengineer.
- You can look up typical hyperparameters like embedding dimension,
  num. heads, sizes of FFNs, in the
  [TinyStories](https://arxiv.org/abs/2305.07759) and
  [Chinchilla](https://arxiv.org/abs/2203.15556) (last page) papers.
- **Important**: run the final training in bfloat16 on an A100,
  maximize the GPU utilization by maximizing the batch size. Look at
  the power usage, or estimate it (like in
  [PaLM](https://arxiv.org/pdf/2204.02311.pdf#appendix.B) appendix B,
  for example).
- Gradient updates for Medium size'd LMs are typically done for every
  100k tokens (e.g. for sequence length 256 and batch size 64, you
  might have to use gradient accumulation with ~6 steps or increase
  the batch size to ~512 to reach >= 100k tokens per gradient step). ⚠️
  This is just a commonly used heuristic, use with caution ⚠️.
- Train for long enough (see Chinchilla paper for example). 10-20B
  tokens should be good enough for sub 1B models.
- We used [RoPE](https://blog.eleuther.ai/rotary-embeddings/) and
  [RMSNorm](https://arxiv.org/abs/1910.07467v1), but the defaults should
  also suffice.
- General good advice on experimenting with DL and training models https://github.com/google-research/tuning_playbook


## Report

Report should contain all the details about your reimplementation: log
experiments, final configuration parameters and describe the process
in detail (better than the TinyStories paper).

> [!IMPORTANT]
> Report should also contain at least **two** pieces of
> evidence that your trained model outperforms the GPT2-XL in story
> generation. One could be just qualitative examples. Second could be
> whatever you want. Automatic evaluation with more powerfull LLMs, 
> reproducing experiments from TinyStories paper, propose your own experiments 
> introspecting the obtained model.


## Scoring

- `5 pts.` for the trained model
- `5 pts.` for experiments and comparison with GPT2-XL
- `2 pts.` extra for excellent reports.

## Compute Resources

We provide around 7700₽ for two large homework. You should manage
resources yourself. In datasphere pricing this roughly equals to 15
full A100 (g2.1) hours - enough for both big homework's final runs
after debugging.


## Dates and deadlines

There will be two deadlines (So that you don't try to cramp everything into the last week). 

**Checkpoint Deadline**. ⚠️ Only hard deadline ⚠️. By `20.11.2023 11:00` you
should submit working training code. Working = it doesn't crash and
seemingly trains the model. **If nothing is submitted** here, **we will subtract 2 points from the final grade**.

**Final Deadline**. ⚠️ Only hard deadline ⚠️. By `04.12.2023 11:00` you should
submit the final training code, model checkpoint and the detailed
report (In either [typst](https://typst.app), [quarto](https://quarto.org), [wandb](https://wandb.ai/site/reports) or LaTeX).

## Plagiarism

Sharing code of your solution with fellow students is prohibited. If
you have discussed any parts of the assignment with other students or
used materials from PyTorch help/tutorials, make sure to state this in
Anytask when submitting the assignment. Copying code from any source
other than PyTorch help or tutorials is not allowed.
