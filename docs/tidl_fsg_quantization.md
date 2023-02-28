# TIDL-RT: Quantization 

[TOC]

# Introduction {#did_tidl_quantization_intro}
- A DNN inference engine using Floating point operations suffer power and cost efficiency. These floating point operations can be substituted with fixed point operations (8 or 16 bit) without losing much inference accuracy.
- TIDL Product provides state-of-art Post Training Quantization (PTQ) and calibration algorithms to provide the best accuracy.
- TIDL Product supports 8-bit, 16-bit and mixed precision inference. It is recommended to use 8-bit inference mode for best execution time and use mixed-precision and/or 16-bit if you observe any accuracy gap with 8-bit inference  

# Quantization Options {#did_tidl_quantization_Types}

TIDL provides the following quantization options to the user:
- A. Post Training Quantization (PTQ)
- B. Guidelines For Training To Get Best Accuracy With Quantization
- C. Quantization Aware Training (QAT)
- D. Native support for TF-Lite int8 PTQ Models*

Note : Option D is only available for certain devices (AM62A, AM68A & AM69A)

We recommend to use option 'A' for the network first, if the quantization accuracy loss is not acceptable, then user can try option 'B'. If the result with 'B' is also not acceptable, then user can use option 'C'. Option 'C' shall work most of the time. The only drawback of this solution is that it would need additional effort from the user to re-train the network. For devices which supports option 'D', user can directly use it.

 
## A. Post Training Quantization (PTQ) {#did_tidl_quantization_1}

- Training free Quantization – Most preferred option
- PTQ has the following options available:
	- Simple Calibration
	- Advanced Calibration
	- Mixed Precision
	- Future/Planned Improvements
	
### A.1. Simple Calibration
- This calibration can be enabled by setting accuracy_level = 0 during model compilation
	- Supports Power of 2 and Non Power of 2 scales for parameters, this can be controlled using advanced_options:quantization_scale_type parameter
	- Supports only power of 2 scales for feature maps
	- Scale selected based on min and max values in the given layer
	- Range for each feature maps are calibrated offline with few sample inputs
	- Calibrated range (Min and Mix) Values are used for Quantizing feature maps in target during inference (real time)
	- Observed accuracy drop less than 1% w.r.t floating point for many networks with 8-bits
		- For example models such as Resnets, SqueezeNet, VGG, etc ( especially models which don't use Depthwise convolution layers)

### A.2. Advanced Calibration
- TIDL product provides some advance options for calibration as listed below:

#### A.2.1. Advanced Bias calibration:
- This feature can be enabled by user by setting accuracy_level = 1 which is one of the optional parameter for model compilation. Typically no other parameter is required to be set as default parameters works for most of the cases. It is observed that using 50 or more number of images gives considerable accuracy boost.
- This feature applies a clipping to the weights and update the bias to compensate the DC errors introduced because of quantization. To understand details of this feature please refer the following <a href="https://github.com/TexasInstruments/edgeai-torchvision/blob/master/docs/pixel2pixel/Calibration.md">Link</a>
- User can also experiment with following parameters related to this option if required:
	- advanced_options:calibration_iterations: Number of iteration to be used for bias calibration.
	- advanced_options:calibration_frames: Number of input frames to be used for bias calibration.
   
#### A.2.2. Histogram based activation range collection:
- To enable this feature user needs to set accuracy_level = 9 and advanced_options:activation_clipping = 1/0 to enable/disable this feature.
- This feature uses the histogram of feature map activation ranges to remove outliers which can affect the overall range. This may help in reducing the accuracy loss due to quantization in some of the networks.


#### A.3. Mixed Precision :
- This feature allows user to run only certain layers in higher precision ( i.e. in 16 bit) whereas rest of the network runs in 8 bit. As the precision keeps changing throughout the network this feature is called as Mixed Precision.
- User can use this feature using following ways :
	- Manually selecting layers for mixed precision :
		- User can manually specify the layers which they want to run in higher precision ( i.e. in 16 bits) using advanced_options:output_feature_16bit_names_list and advanced_options:params_16bit_names_list parameters for model compilation. User has option to either increase only parameters/weights precision to 16 bit or to have both activation and parameters of a particular layer in 16 bit.
		- TIDL allows change of precision for certain set of layers ( mentioned below ). For layers not mentioned in this list, you cannot change the precision. This means that the layers which support change in precision can have input, output and parameters in different precision. Whereas the layers which do not support change in precision will always have input, output and parameters in same precision. The impact of this is that for a particular layer which doesn't support change in precision, the input, output and parameter's precision will be automatically determined based on the producer or consumer of the layer. For example, for the concat layer, which doesn't support change in precision, if the output is in 16 bit because of its consumer layer or because the user requested for the same, then it will change all its input to be in 16 bits as well.
		- If for a given layer output is already a floating point output (e.g. Softmax, DetectionOutputLayer etc) then increasing activation precision has no impact.

- Few Points to Note:
	- Currently following layers support change in precision and all the other layers cannot have input and output in different precision i.e. their precision is determined by their producer/consumer and both input and output will be in the same precision :
		- TIDL_ConvolutionLayer ( Except TIDL_BatchToSpaceLayer and TIDL_SpaceToBatchLayer) 
		- TIDL_BatchNormLayer
		- TIDL_PoolingLayer ( Excluding Max pooling layer) 
		- TIDL_EltWiseLayer

#### A.4.1 Automated Mixed Precision - Automatic selection of layers :
- This is an enhancement to the mixed precision feature. It enables automatic selection of layers to be set to 16 bit for improved accuracy
- The accuracy improvement with mixed precision comes with a performance cost. This feature accepts a parameter to specify the user-tolerable performance cost and accordingly sets the most impactful layers to 16 bit to meet the user specified performance constraint
- User can use this feature by setting the parameter advanced_options:mixed_precision_factor which is one of optional parameter for model compilation
	- Let the latency for network executing entirely in 8 bit precision be T_8 and let the latency for network executing with mixed precision be T_MP
	- We define mixedPrecisionFactor = T_MP / T_8
	- As an example, if the latency for 8 bit inference of a network is 5 ms, and if tolerable latency with mixed precision is 6 ms, then set mixedPrecisionFactor = 6/5 = 1.2
	- If set value of mixedPrecisionFactor does not provide desired accuracy, consider increasing its value, e.g. in above example, set mixedPrecisionFactor = 1.4 instead of 1.2 
- This method uses advanced bias calibration as part of the algorithm to do auto selection of layers. Recommended values are
		- Set accuracy_level = 1, calibration_frames = 50 and calibration_iterations = 50 which are optional parameters for model compilation
		- The algorithm uses "calibration_frames/4" frames and "calibration_iterations/4" iterations for auto selection of layers followed by bias calibration with "calibration_frames" frames and "calibration_iterations" iterations
- Note : The compilation time for running automated mixed precision is high, so recommended to use utilities like <a href="https://www.gnu.org/software/screen/manual/screen.html">screen</a> to run compilation without interruption

### A.5 Future/Planned Improvements
- The following options are not supported in current release but are planned for future TIDL releases (For AM62A, AM68A & AM69A):
  - Support for asymmetric & non power of 2 scales
  - Support for efficient per channel 
	

## B. Guidelines For Training To Get Best Accuracy With Quantization {#did_tidl_quantization_2}
- For best accuracy with post training quantization, we recommend that the training uses sufficient amount of regularization / weight decay. Regularization / weight decay ensures that the weights, biases and other parameters (if any) are small and compact - this is good for quantization. These features are supported in most of the popular training framework.
- The weight decay factor should not be too small. We have used a weight decay factor of 1e-4 for training several networks and we highly recommend a similar value. Using small values such as 1e-5 is not recommended.
- We also highly recommend to use Batch Normalization immediately after every Convolution layer. This helps the feature map to be properly regularized/normalized. If this is not done, there can be accuracy degradation with quantization. This especially true for Depthwise Convolution layers. However applying Batch Normalization to the very last Convolution layer (for example, the prediction layer in segmentation/object detection network) may hurt accuracy and can be avoided.
- To summarize, if you are getting poor accuracy with quantization, please check the following:
	- (a) Weight decay is applied to all layers / parameters and that weight decay factor is good.
	- (b) Ensure that all the Depthwise Convolution layers in the network have Batch Normalization layers after that - there is strictly no exception for this rule. Other Convolution layers in the network should also have Batch Normalization layers after that - however the very last Convolution layer in the network need not have it (for example the prediction layer in a segmentation network or detection network).


## C. Quantization Aware Training (QAT) {#did_tidl_quantization_3}

- Model parameters are trained to comprehend the 8-bit fixed point inference loss.
- This would need support/change in the training framework 
- Once a model is trained with QAT, the feature map range values are inserted as part of the model. There is no need to use advanced calibration features for a QAT model.
Example – CLIP, Minimum, PACT, RelU6 operators.
- This option has resulted in accuracy drop to be very close to zero for most of the networks.
- EdgeAI-TorchVision provides tools and examples to do Quantization Aware Training. With the tools provided, you can incorporate Quantization Aware Training in your code base with just a few lines of code change. For detailed documentation and code, please visit <a href="https://github.com/TexasInstruments/edgeai-torchvision/blob/master/docs/pixel2pixel/Quantization.md">Link</a>

## D. Native support for TensorFlow Lite int8 PTQ Models
- TF-Lite full-integer quantized models (limited to int8 ops) can be inferred directly on certain devices (AM62A, AM68A & AM69A) without further calibration
- Refer to <a href="https://www.tensorflow.org/lite/performance/post_training_quantization">Post Training Quantization | TensorFlow Lite</a> for further details on how to quantize TensorFlow Lite models