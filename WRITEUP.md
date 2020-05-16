# Project Write-Up

The model I have used in my project is 'person-detection-retail-0013'.
This model was taken from website: https://docs.openvinotoolkit.org/2019_R3/_models_intel_index.html

The model suitable for my device was: https://docs.openvinotoolkit.org/2019_R3/_models_intel_person_detection_retail_0013_description_person_detection_retail_0013.html

I have used the following commands to download the model:
- `cd /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader`
- `ls`
- `sudo ./downloader.py --name person-detection-retail-0013 --precisions FP16 -o /home/workspace`

***This helped me to get `.xml` and `.bin` files required for the project and converting it into Inference model***

Then, I used the following command to run main.py:

```
python main.py -i resources/Pedestrian_Detect_2_1_1.mp4 -m person-detection-retail-0013.xml -l /opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.6 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm
```

## Explaining Custom Layers

The process behind converting custom layers involves depends on frameowrks we use either it is tensorflow,caffee or kaldi. We can find it here:
https://docs.openvinotoolkit.org/2019_R3/_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer.html

The example model is fed to the Model Optimizer that loads the model with the special parser, built on top of caffe.proto file. In case of failure, Model Optimizer asks you to prepare the parser that can read the model.
Model Optimizer extracts the attributes of all layers. In particular, it goes through the list of layers and attempts to find the appropriate extractor. In order of priority, Model Optimizer checks if the layer is registered as:

  - CustomLayersMapping.xml
  - Model Optimizer extension
  - Standard Model Optimizer layer

Custom layers are important to use because:
  - If your layer output shape depends on dynamic parameters, input data or previous layers parameters, calculation of output shape of the layer via model used can be incorrect. In this case, you need to patch it on your own.
  - If the calculation of output shape of the layer fails inside the framework, Model Optimizer is unable to produce any correct Intermediate Representation and you also need to investigate the issue in the implementation of layers and patch it.
  - You are not able to produce Intermediate Representation on any machine that does not have model installed. If you want to use Model Optimizer on multiple machines, your topology contains Custom Layers and you use CustomLayersMapping.xml to fallback on it, you need to configure on each new machine.

Some of the potential reasons for handling custom layers is to optimize our pre-trained models and convert them to a intermediate representation(IR) without a lot of loss of accuracy and shrink and speed up the Performance so that desired output is resulted.

You have two options for TensorFlow* models with custom layers as I have tried 3 models below:

  - Register those layers as extensions to the Model Optimizer. In this case, the Model Optimizer generates a valid and optimized Intermediate Representation.
  - If you have sub-graphs that should not be expressed with the analogous sub-graph in the Intermediate Representation, but another sub-graph should appear in the model, the Model Optimizer provides such an option. This feature is helpful for many TensorFlow models.

There are majorly two custom layer extensions required-
- Custom Layer Extractor
  Responsible for identifying the custom layer operation and extracting the parameters for each instance of the custom layer. The layer parameters are stored per instance and used by the layer operation before finally appearing in the output IR. Typically the input layer parameters are unchanged, which is the case covered by this tutorial.

- Custom Layer Operation
  Responsible for specifying the attributes that are supported by the custom layer and computing the output shape for each instance of the custom layer from its parameters. The `--mo-op` command-line argument shown in the examples below generates a custom layer operation for the Model Optimizer.

Each device plugin includes a library of optimized implementations to execute known layer operations which must be extended to execute a custom layer. The custom layer extension is implemented according to the target device:

- Custom Layer CPU Extension
  A compiled shared library (`.so` or `.dll` binary) needed by the CPU Plugin for executing the custom layer on the CPU.

- Custom Layer GPU Extension
  OpenCL source code (`.cl`) for the custom layer kernel that will be compiled to execute on the GPU along with a layer description file (`.xml`) needed by the GPU Plugin for the custom layer kernel.

- Using the model extension generator
  The Model Extension Generator tool generates template source code files for each of the extensions needed by the Model Optimizer and the Inference Engine.

The script for this is available here- 
`/opt/intel/openvino/deployment_tools/tools/extension_generator/extgen.py`

## Comparing Model Performance

My method(s) to compare models before and after conversion to Intermediate Representations
were its speed and its accuracy to give final output relatively.

The difference between model accuracy pre- and post-conversion was that it decreased as the model was shrinked and helped me to make it faster, although this will not give the model higher inference accuracy. In fact, there will be some loss of accuracy as a result of potential changes like lower precision. Therefore, these losses in accuracy are minimized.

I have majorly stressed on model accuracy and inference time, I have included model size as a secondary metric. I have stated the models I experimented with. For more information take a look at Model Research.

Model size
| |SSD MobileNet V2|SSD Inception V2|Faster rcnn Inception| |
Before Conversion |67 MB|98 MB|28 MB|
After Conversion |65 MB|96 MB|26 MB|

Inference Time
| |SSD MobileNet V2|SSD Inception V2|Faster rcnn Inception| |
Before Conversion |50 ms|150 ms|55 ms|
After Conversion |60 ms|155 ms|60 ms|

## Assess Model Use Cases

Some of the potential use cases of the people counter app are

- It provides valuable visitor analytics.

- It helps to improve in-store operations.

- Every business with a physical space should count customer traffic to see the bigger picture of what is going on in their business.

- Each of these use cases would be useful because every business whether you are a shopping center, retail chain, museum, library, sporting venue, bank, restaurant or other.People Counting data will help you make well-informed decisions about your business.

- It can help to optimize sales and conversions.

Each of these use cases would be useful because it helps to optimise our work and makes it easier to detect whatever we require.

## Assess Effects on End User Needs

Lighting, model accuracy, and camera focal length/image size have different effects on a deployed edge model. The potential effects of each of these are as follows:

Better the model accuracy, More are the chances to obtain the desired results through an app deployed at edge.

Focal length/image also have a effect as better be the pixel quality of image or better the camera focal length,more clear results ww will obtain.

Lighter the model, More faster it will get execute and more adequate results in faster time as compared to a heavier model.

## Model Research

[This heading is only required if a suitable model was not found after trying out at least three different models. However, you may also use this heading to detail how you converted 
a successful model.]

In investigating potential people counter models, I tried each of the following three models:

- Model 1: [faster_rcnn_inception_v2_coco_2018_01_28]
  - [Model Source](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments:
  
      - To download the model:
       `wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz`
      
      - To unzip the tar:
      `tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz`
      
      - To switch the directory:
      `cd faster_rcnn_inception_v2_coco_2018_01_28`
      
      - To convert model into IR:
      ```
      python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
      ```
      
      - Command to run the model:
      ```
      python main.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config -- reverse_input_channels --tensorflow_use_custom_operations_config/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/faster_rcnn_support.json
      ```
      - It was obtained in 136 seconds.
  
  - The model was insufficient for the app becuase when I tried it had many errors including server connection becuase it missed some files which even after resetting the data of workspace didnt change.
  - I tried to improve the model for the app by checking in documentation and found input is empty. So I added another input too but still it gave me errors.
      
- Model 2: [ssd_mobilenet_v2_coco]
  - [Model Source](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments:
  
      - To download the model:
      `wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz`
      
      - To unzip the model:
      `tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz`
      
      - To switch directory:
      `cd ssd_mobilenet_v2_coco_2018_03_29`
      `export MOD_OPT=/opt/intel/openvino/deployment_tools/model_optimizer`
      
      - To convert model into IR:
      ```
      python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
      ```
      
      - Command to run
      ```
      python main.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config -- reverse_input_channels --tensorflow_use_custom_operations_config/opt/intel/openvino/deployment_tools/model_optimizer/extensions/front/tf/ssd_v2_support.json
      ```
      
      - .xml and .bin files were obtained in 59.66 seconds
      
  - The model was insufficient for the app because it showed the error of not getting connected to the server even after running the mosca and ffmpeg severs correctly. I asked montors about the error and rechecked everything. Finally output was able to run on new tab but still the video was not able to load. As then suggested by a mentor, I switched the model and dropped this.
  - I tried to improve the model for the app by resetting the data 4 times and reloading everything as i couldnt see any error in my steps. I tried to change the video which I thought may be shown in output screen but didnt work. Also tried with an image but no results.

- Model 3: [ssd_inception_v2_coco_2018_01_28]
  - [Model Source](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
  - I converted the model to an Intermediate Representation with the following arguments:
      
      - To download the model:
      `wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz -Unpack the file`
      
      - To unzip the model:
      `tar -xvf ssd_inception_v2_coco_2018_01_28.tar.gz`
      
      - To switch the directory:
      `cd ssd_inception_v2_coco_2018_01_28`
      
      - To convert the model into IR:
      ```
      python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
      ```
      
      - To run the model:
      ```
      python mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config extensions/front/tf/ssd_v2_support.json
      ```
      
  - The model was insufficient for the app because It also has issues like first model where it failed to detect person in specific time period. It was sticking at some places where a person may leave the screen but still counted as 1 person on screen. Tried it with other video but same results. Count was not proper and also time duration which would affect the average duration with no accurate measurements exactly so I couldn't use this model too.
