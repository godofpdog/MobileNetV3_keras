# MobileNet V3
***
## Requirments
```
numpy 1.15.1
keras 2.2.4
tensorflow-gpu 1.9.0
opencv-python 3.4.3.18
imgaug 0.2.8
```

## Training
* You can define taining learning rate schedule by edit `src/learning_rate_schedule.py`.
```
python train.py -c config/train.ini
```

### Arguments
* data
* model
* train
* gpu

### **data**
Argument|Description|Type |Default
---|---|-----|---
train|Training dataset directory.|str|None
valid|Validation dataset directory.|str|None

### **model**
Argument|Description|Type|Default
---|---|---|---
input_size|Input size of MobileNet V3 model.|int|224
model_size|"large" or "small" version of MobileNet V3 model.|str|large
pooling_type|Pooling type of MobileNet V3 model. (avg or depthwith)|str|avg
num_classes|Number of classes.|int|1000

### **train**
Argument|Description|Type|Default
---|---|---|---
epochs|Maximun number of training epochs.|int|200
batch_size|Batch size of data generator.|int|32
save_path|Saved weights path.|str|weights/*.h5
pretrained_path|Pre-trained model path of MobileNet V3 model.|str|None

### **gpu**
Argument|Description|Type|Default
---|---|---|---
gpu|Specify a GPU.|str|-1

***
## bottleneck structure configuation
* You can define custom bottleneck structure by edit ***large_config_list*** and ***small_config_list*** in `MobileNet_V3.py`

Argument|Description|Type|Code
---|---|---|---
out_dim|Output chennal dimension.|int|out
kernel|Kernel size of filter.|tuple|kernel
strides|Strides of the converlutional operation.|tuple|stride
expansion_dim|Expansion dimension of the bottleneck block.|int|exp
is_use_bias|Use bias or not.|bool|bias
res|Use shortcut operation or not.|bool|res
is_use_se|Use SE block or not.|bool|se
activation|Activative functions. ('RE' or 'HS')|str|active
num_layers|Layer index number.|int|id 

### example
```python 
# NOTE               out   kernel  stride  exp  bias   res    se     active id  
large_config_list = [[16,  (3, 3), (1, 1), 16,  False, False, False, 'RE',  0],
                     [24,  (3, 3), (2, 2), 64,  False, False, False, 'RE',  1],
                     [24,  (3, 3), (1, 1), 72,  False, True,  False, 'RE',  2],
                     [40,  (5, 5), (2, 2), 72,  False, False, True,  'RE',  3],
                     [40,  (5, 5), (1, 1), 120, False, True,  True,  'RE',  4],
                     [40,  (5, 5), (1, 1), 120, False, True,  True,  'RE',  5],
                     [80,  (3, 3), (2, 2), 240, False, False, False, 'HS',  6],
                     [80,  (3, 3), (1, 1), 200, False, True,  False, 'HS',  7],
                     [80,  (3, 3), (1, 1), 184, False, True,  False, 'HS',  8],
                     [80,  (3, 3), (1, 1), 184, False, True,  False, 'HS',  9],
                     [112, (3, 3), (1, 1), 480, False, False, True,  'HS', 10],
                     [112, (3, 3), (1, 1), 672, False, True,  True,  'HS', 11],
                     [160, (5, 5), (1, 1), 672, False, False, True,  'HS', 12],
                     [160, (5, 5), (2, 2), 672, False, True,  True,  'HS', 13],
                     [160, (5, 5), (1, 1), 960, False, True,  True,  'HS', 14]]
```
