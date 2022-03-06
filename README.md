# Shading-aware Unsupervised Learning for Intrinsic Image Decomposition
 
 
# Requirement
- Python: 3.9.6
- CUDA: 11.2.0
# Installation
 ```
 git clone https://github.com/tiruota/ShadinAware-USI3D
 cd ShadinAware-USI3D
 pip install -r requirements.txt
 ```
 
データセットをダウンロード・配置([dataset.zip](https://drive.google.com/file/d/13D3FxHKFu7HZ2DLZ82Yg6HXumN05szTQ/view?usp=sharing))
```
ShadinAware-USI3D/dataset/
```
## Usage
学習
```
python train.py -c [path_to_config] -o [output_dir] -g [gpu_id]
```
学習の例
```
python train.py -c configs/configs.yaml -o checkpoints/ -g 0
```
テスト([学習済みモデル](https://drive.google.com/file/d/1iQELnZg-WAwgO73oP5jqveARcCSqEhR9/view?usp=sharing))
```
python test.py -c [path_to_configs] -p [path_to_checkpoint] -o [output_dir] -i [test_list] -g [guided_image_dir] -e [edge_image_dir]
```
テストの例
```
python test.py -c configs/configs.yaml -p checkpoints/gen.pt -o results/ -i dataset/test-input.txt -g dataset/L1smooth/ -e dataset/edge_iiw_dexi/
```
MSE評価(Ground-truthがある場合)
```
python eval.py -r [predict_reflectance_dir] -s [predict_shading_dir] -t [target_reflectance_dir] -u [target_shading_dir]
```

WHDR評価
```
python calc_whdr.py -i [predict_reflectance_dir] -j [judgement_dir]
```