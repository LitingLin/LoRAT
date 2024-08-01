## Custom Dataset

This page describes how to create a custom dataset for training and evaluation.

### Add A New Dataset
To add a new dataset, you need to create a new `{DatasetName}.py` file under the `trackit/datasets/SOT/datasets` directory. This file should define a new class `{DatasetName}_Seed` that inherits from `trackit.datasets.common.seed.BaseSeed` and implements the `__init__` and `construct` methods.

Assuming the new dataset is named `MyDataset`, the following is an example of the `MyDataset.py` file:
```python
# trackit/datasets/SOT/datasets/MyDataset.py

import os
import numpy as np

from trackit.datasets.common.seed import BaseSeed
from trackit.datasets.SOT.constructor import SingleObjectTrackingDatasetConstructor

class MyDataset_Seed(BaseSeed):
    def __init__(self, root_path: str = None):
        if root_path is None:
            # get the path from `consts.yaml` file
            root_path = self.get_path_from_config('MyDataset_PATH') 
        super(MyDataset_Seed, self).__init__(
            'MyDataset', # dataset name
            root_path,   # dataset root path
        )

    def construct(self, constructor: SingleObjectTrackingDatasetConstructor):
        # Implement the dataset construction logic here
        
        sequence_names = ['seq1', 'seq2', 'seq3']
        
        # Set the total number of sequences (Optional, for progress bar)
        constructor.set_total_number_of_sequences(len(sequence_names))
        
        # Set the bounding box format (Optional, 'XYXY' or 'XYWH', default for XYWH)
        constructor.set_bounding_box_format('XYWH')
        
        # get root_path
        root_path = self.root_path
        
        for sequence_name in sequence_names:
            '''
            The following is an example of the dataset structure:
            root_path
            ├── seq1
            │   ├── frames
            │   │   ├── 0001.jpg
            │   │   ├── 0002.jpg
            │   │   └── ...
            │   └── groundtruth.txt
            ├── seq2
            ...            
            '''
            with constructor.new_sequence() as sequence_constructor:
                sequence_constructor.set_name(sequence_name)
                
                sequence_path = os.path.join(root_path, sequence_name)
                # groundtruth.txt: the path of the bounding boxes file
                boxes_path = os.path.join(sequence_path, 'groundtruth.txt')
                frames_path = os.path.join(sequence_path, 'frames')
                
                # load bounding boxes using numpy
                boxes = np.loadtxt(boxes_path, delimiter=',')
                
                for frame_id, box in enumerate(boxes):
                    # frame_path: the path of the frame image, assuming the frame image is named as 0001.jpg, 0002.jpg, ...
                    frame_path = os.path.join(frames_path, f'{frame_id + 1:04d}.jpg')
                    
                    with sequence_constructor.new_frame() as frame_constructor:
                        # set the frame path and image size 
                        # image_size is optional (will be read from the image if not provided)
                        frame_constructor.set_path(frame_path, image_size=(1920, 1080))
                        # set the bounding box
                        # validity is optional (False for fully occluded or out-of-view or not annotated)
                        frame_constructor.set_bounding_box(box, validity=True)
```

Note: The new dataset can also be Multi-Object Tracking (MOT) dataset or Detection (DET) dataset. See the `trackit/datasets/MOT` and `trackit/datasets/DET` directories for examples.

### Add the new dataset for training and evaluation

#### Set the Dataset Path in `consts.yaml`
```yaml
MyDataset_PATH: '/path/to/MyDataset'
```

#### Set the New Dataset for Training

Create a new mixin config file `config/LoRAT/_mixin/my_dataset_train.yaml`:

```yaml
# config/LoRAT/_mixin/my_dataset_train.yaml
- path: "run.data.train.source.parameters.datasets"
  value:
    - name: "MyDataset"
      type: "SOT"
- path: "run.data.train.sampler"
  value:
    type: "random"
    samples_per_epoch: 131072  # set a new value for samples per epoch
    # weight: multi-dataset sampling weight is not required, since there is only one dataset
```

Start training with the new dataset:
```shell
./run.sh LoRAT dinov2 --mixin my_dataset_train --output_dir /path/to/output
```

note: mixin config file can be in `config/_mixin`, `config/method_name/_mixin` or `config/method_name/config_name/mixin`, each with a different visibility scope.

#### Set the New Dataset for Evaluation:

Create a new mixin config file `config/LoRAT/_mixin/my_dataset_test.yaml`:

```yaml
# config/LoRAT/_mixin/my_dataset_test.yaml
- path: "run.data.eval.source.parameters.datasets"
  value:
    - name: "MyDataset"
      type: "SOT"
```

Start evaluation with the new dataset:
```shell
./run.sh LoRAT dinov2 --mixin my_dataset_test --mixin evaluation --weight_path /path/to/weight.bin --output_dir /path/to/output
```

Train with the default setting and evaluate with the new dataset:
```shell
./run.sh LoRAT dinov2 --mixin my_dataset_test --output_dir /path/to/output
```

Train and evaluate with the new dataset:
```shell
./run.sh LoRAT dinov2 --mixin my_dataset_train --mixin my_dataset_test --output_dir /path/to/output
```
