
<h2 align="center">GASN: Grain Adaptive Segmentation Network for Dense Scenarios (Updating)</h2>
<p align="center">
  <a href='https://www.sciencedirect.com/science/article/abs/pii/S0168169924011633'><img src='https://img.shields.io/badge/Paper-GASN-blue' alt='Paper PDF'></a>
  <a href=#><img src='https://img.shields.io/badge/License-MIT-green' alt='License'></a>
</p>

![image](res/img/gasn.png)

## News
- [2024.05.28] 🔥 Release GIDAS dataset.
- [2024.12.02] 🎉 GASN **accepted** by ***Computers and Electronics in Agriculture***. 🎉
- [2024.12.02] 🔥 Update Apache 2.0 License.
- [2025.01.16] 🔥 Release model, checkpoint and code. Update usage document.

## TODO List
- [x] Dataset
- [x] Release model, checkpoint and demo code. 
- [x] Usage document.

## Environment and Platform
- Pytorch 1.13
- RTX 3060
- Windows 10 and Ubuntu 18.04
## Tutorial
### Step 1: Dataset and Checkpoint download 
Images and density maps can be downloaded from here:

[Google Drive](https://drive.google.com/file/d/1p6Rawe30yahceD1q8RrGfhGnSCgxYgFn/view?usp=drive_link)
and
[Baidu Netdisk](https://pan.baidu.com/s/1T9JRfQwx7daYTcYL4gAL9w?pwd=pzp5)

Backbone chexkpoint is in here:

We Use SAM [Vit-H](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints).

Use ```exp/vis_dataset.py``` to show annotations of GIDAS. 
```shell
python ./exp/vis_dataset.py
```

### Step 2: Preprocess dataset (if you download all dataset, this step can be skiped)
Use ```exp/preprocess.py``` process dataset.

Firstly, we generate ground truth density map for GIDAS.

```python
# generate GIDAS density map step 1, transform file type to mat
# it need only to run once
generate_density_mat()
# generate GIDAS density map step 2
# if set different sigma, it must be rerun
generate_density_map()
```

Secondly, we generate image imbedding for GIDAS to speed up calculations (adapt to 3060)

```python
# generate GIDAS image embedding
gidas_preprocess()
```

### Step 3: Run GASN Demo
Use ```exp/gasn_demo.py``` to run demo.

Select one of GIDAS to show effect, such as 103.jpg. (The 103.jpg is default.) 

Open ```exp/dataset_gidas.py``` to set image in class ```GIDASDataset``` function ```__init__```.
```python
self.image_list = ["103.jpg"]
```
Then, you can run it.
```shell
python ./exp/gasn_demo.py
```
Some configurations is in ```exp/config.py```.



### Step 4: Test GASN
Use ```exp/gasn_test.py``` to test gasn.

Set all of GIDAS to test. 

Open ```exp/dataset_gidas.py``` to set image in class ```GIDASDataset``` function ```__init__```.
```python
self.image_list = os.listdir(self.image_dir)
```
Then, you can test it.
```shell
python exp/gasn_test.py
```
### Step 5: Train PDDN (GASN is train-free)
```shell
python exp/train.py
```
