
# VSTAR: Generative Temporal Nursing for Longer Dynamic Video Synthesis

:fire:  Official implementation of "VSTAR: Generative Temporal Nursing for Longer Dynamic Video Synthesis" (ICLR2025)

🚀**TL;DR**: VSTAR enables *pretrained* text-to-video models to generate longer videos with dynamic visual evolution in a **single** pass, **without finetuning needed**.


<table class="center">
  <td><img src=docs/lava.gif width="320"></td>
  <td><img src=docs/boy_girl.gif width="320"></td>
  <tr>
  <td><img src=docs/beach.gif width="320"></td>
  <td><img src=docs/superman.gif width="320"></td>
  <tr>
</table >

<br />


## Getting Started

Our environment is built on top of [VideoCrafter2](https://github.com/AILab-CVC/VideoCrafter):
```
conda create -n vstar python=3.10.6 pip jupyter jupyterlab matplotlib
conda activate vstar
pip install -r requirements.txt
```
Download pretrained Videocafter2 320x512 checkpoint from [here](https://huggingface.co/VideoCrafter/VideoCrafter2/blob/main/model.ckpt) and store it in the [checkpoint](checkpoint) folder.

## Inference
Run [inference_VSTAR.ipynb](inference_VSTAR.ipynb) for testing.


## License

This project is open-sourced under the AGPL-3.0 license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in this project, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).


## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. 


## Contact     

Please feel free to open an issue or contact personally if you have questions, need help, or need explanations. Don't hesitate to write an email to the following email address:
liyumeng07@outlook.com


