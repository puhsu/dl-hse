# Big homework 2. Generative models for images.

<p align="center"> <img src="./face.png" style="width: 30%;"> </p>

## Task description

In this assignment, you must implement and train an image generative
model on **your** dataset of choice. The choice of the model is also
yours, it could be either WGAN-GP/-SN, StyleGAN, or even DDPM. 

If you don't know what dataset to use, you can use `anime_faces`, but
you'll miss out on some fun and bonus points.

What a good result should look like?
If you or your classmates cannot distinguish between a sample from the dataset and your model's output - this is a good result.
A clear 64x64 anime face that you can proudly share in tg chat or use as an avatar (let's ignore the size, shall we?).
If there are some artifacts like a lesser number of eyes, more than one mouth, etc - model quality is poor. 

## Data

You can get an example dataset mentioned in the description from
kaggle, but once again: you can pick whatever dataset you want, even
generate one yourself (e.g. [The Five DollarModel](https://arxiv.org/abs/2308.04052))

```bash
https://www.kaggle.com/datasets/soumikrakshit/anime-faces/data
```

## Code & Advice

> [!IMPORTANT]
> The only dependencies you can use to implement and
> Train your BLM in this homework in `env.yaml` example below. If
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
- pytorch::torchvision
- photosynthesis-team::piq  # for FID and SSIM
- requests
- matplotlib
- wandb
- tensorboard
- tqdm
```

Advice: TBD (tomorrow)

## Report

The report should contain all the details about your reimplementation: log
experiments, final configuration parameters and describe the process
in detail (as in the first big homework). 

## Scoring

- `5 pts.` You prepared the code for the model and the training script. Up to `3 pts` for model. `1 pts` for the correct training script and `1 pts` for the evaluation part of the training script. 
- `5 pts.` You trained and evaluated the model. Compute FID and SSIM on evaluation and add a plot to the report. Up to `3 pts` for model quality. 1 point for good metrics graphs and 1 for the gif with the model's evolution.
- `2 pts.` Bonus. Extra points were given for excellent reports, comparison of GANs and Diffusion models, visualizations, and interesting custom datasets.

## Dates and deadlines

There will be one deadline

**Final Deadline**. By `18.12.2023 11:00` you should submit the model, the training script, and the detailed report (In either [typst](https://typst.app), [quarto](https://quarto.org), [wandb](https://wandb.ai/site/reports) or LaTeX).

## Plagiarism

Sharing the code of your solution with fellow students is prohibited. If
you have discussed any parts of the assignment with other students or
used materials from PyTorch help/tutorials, make sure to state this in
Anytask when submitting the assignment. Copying code from any source
other than PyTorch help or tutorials are not allowed.