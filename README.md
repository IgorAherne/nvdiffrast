Allows to build a `.whl` for windows, cuda 11.8<br>
This way, users don't need to have VisualStudio nor Cuda Toolkit installed.<br>
They can just use this wheel after pulling it via pip.

Steps:

1) Make sure Cuda Toolkit is installed, v 11.8
2) Make sure you have Visual Studio 2022.
3) create a virtual environment: `python -m venv venv` then activate it: `.\venv\Scripts\activate`
4) Verify that python is exactly 3.11.<br>
   If not, remove the `venv` folder and redo like so: `& "C:\Program Files\PATH-TO-YOUR-PYTHON\Python311\python.exe" -m venv venv` **and activate it again** `.\venv\Scripts\activate`

5) If doing through powershell / vscode terminal, ensure environment variables are setup, so that cuda and the visual studio can be found.
   <br>So, check the filepaths make sense/exist on your pc. Then, write inside the terminal:
   `$Env:CUDA_HOME="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"`
   `$Env:PATH="$Env:PATH;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.40.33807\bin\Hostx86\x64"`

   Also, setup the env variables by running:
   `& "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"`

   Note the `\x64` and the `64` in the above 2 commands. We'll be building for x64.

6) This should have made `cl.exe` discoverable, which is important for compilation.

6) `pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118`
7) `python setup.py clean`
8) `python setup.py build_ext --inplace -v`

   now you should have a `.whl` for nvdiffrast, which can be easily used by Windows users, with Cuda 11.8 and Python 3.11

Original repo description:

--------------------------------

## Nvdiffrast &ndash; Modular Primitives for High-Performance Differentiable Rendering

![Teaser image](./docs/img/teaser.png)

**Modular Primitives for High-Performance Differentiable Rendering**<br>
Samuli Laine, Janne Hellsten, Tero Karras, Yeongho Seol, Jaakko Lehtinen, Timo Aila<br>
[http://arxiv.org/abs/2011.03277](http://arxiv.org/abs/2011.03277)

Nvdiffrast is a PyTorch/TensorFlow library that provides high-performance primitive operations for rasterization-based differentiable rendering.
Please refer to &#x261E;&#x261E; [nvdiffrast documentation](https://nvlabs.github.io/nvdiffrast) &#x261C;&#x261C; for more information.

## Licenses

Copyright &copy; 2020&ndash;2024, NVIDIA Corporation. All rights reserved.

This work is made available under the [Nvidia Source Code License](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt).

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/)

We do not currently accept outside code contributions in the form of pull requests.

Environment map stored as part of `samples/data/envphong.npz` is derived from a Wave Engine
[sample material](https://github.com/WaveEngine/Samples-2.5/tree/master/Materials/EnvironmentMap/Content/Assets/CubeMap.cubemap)
originally shared under 
[MIT License](https://github.com/WaveEngine/Samples-2.5/blob/master/LICENSE.md).
Mesh and texture stored as part of `samples/data/earth.npz` are derived from
[3D Earth Photorealistic 2K](https://www.turbosquid.com/3d-models/3d-realistic-earth-photorealistic-2k-1279125)
model originally made available under
[TurboSquid 3D Model License](https://blog.turbosquid.com/turbosquid-3d-model-license/#3d-model-license).

## Citation

```
@article{Laine2020diffrast,
  title   = {Modular Primitives for High-Performance Differentiable Rendering},
  author  = {Samuli Laine and Janne Hellsten and Tero Karras and Yeongho Seol and Jaakko Lehtinen and Timo Aila},
  journal = {ACM Transactions on Graphics},
  year    = {2020},
  volume  = {39},
  number  = {6}
}
```
