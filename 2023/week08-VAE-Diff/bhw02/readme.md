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

## Papers
StyleGAN https://arxiv.org/abs/1812.04948 <br>
StyleGAN3 https://arxiv.org/abs/2106.12423 <br>
WGAN-GP https://arxiv.org/abs/1704.00028 <br> 
WGAN-SN https://arxiv.org/abs/1802.05957<br>
DDPM https://arxiv.org/abs/2006.11239 <br>

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
in detail (as in the first big homework). The details you listed in the report must be sufficient to 
reproduce all your results. Also, don't forget about the conclusion section, where you reflect on your thoughts about the results you've got
and possible reasons why (why the results are good or not as good). Is there are any ideas left untested (because you didn't have enough time/resources to test them all)? <br>
Also, besides human evaluation, reflect your opinion on the numerical evaluation. Does FID and SSIM describe the quality improvement throughout the training process well enough? <br>
Would you change anything for the evaluation step, and if yes, why? <br>
Add a general conclusion about work as a whole.

## Scoring

- `5 pts.` You prepared the code for the model and the training script. Up to `3 pts` for model. `1 pts` for the correct training script and `1 pts` for the evaluation part of the training script. 
- `5 pts.` You trained and evaluated the model. Compute FID and SSIM on evaluation and add a plot to the report. Up to `3 pts` for model quality. 1 point for good metrics graphs and 1 for the gif with the model's evolution.
- `2 pts.` Bonus. Extra points were given for excellent reports, comparison of GANs and Diffusion models, visualizations, and interesting custom datasets.

## Dates and deadlines

There will be one deadline

**Final Deadline**. By `18.12.2023 23:30` you should submit the model, the training script, and the detailed report (In either [typst](https://typst.app), [quarto](https://quarto.org), [wandb](https://wandb.ai/site/reports) or LaTeX).

## Plagiarism

Sharing the code of your solution with fellow students is prohibited. If
you have discussed any parts of the assignment with other students or
used materials from PyTorch help/tutorials, make sure to state this in
Anytask when submitting the assignment. Copying code from any source
other than PyTorch help or tutorials are not allowed.
