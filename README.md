# Alzheimers-Clock-Drawing

This repository reports the results of constructing a two-dimensional convolutional neural network to classify the RowanSOM Alzheimer's disease clock drawing dataset.

## Dataset

The Alzheimer's disease clock drawing dataset consists of 196 patients who have taken the clock drawing test. For each patient, there are 352 features, comprising 68 categorical features and 284 numerical features. The features are extracted from two tests: a "command" test and a "copy" test. The "command" test asks the patient to draw, from memory, an analog clock set to ten minutes after eleven. The "copy" test asks the patient to draw a copy of an analog clock set to ten minutes after eleven. 

Patients are classified as one of five levels of cognitive impairment:

| Code | Label |
| ---- | ----- |
|  1   |  SCI  |
|  2   |  AMCI  |
|  3   |  MMCI  |
|  4   |  AD   |
|  5   |  VAD  |

where the labels stand for Subjective Cognitive Impairment (SCI), Amnestic Mild Cognitive Impairment (AMCI), Mixed Mild Cognitive Impairment (MMCI), Alzheimer's Disease (AD), and Vascular Dementia (VAD). 

### Preprocessing

The features "comm_ClockFaceNonClockFaceNoNoiseLatency" and "CFNonCFNoNoiseTerminator" are removed from the dataset, as they appear in the "command" set of features but not the "copy" set of features. Except for two, all other features are shared between the two feature sets (the remaining features are "ClockFaceNonClockFaceLatency" in the command set and "copy_PostClockFaceLatencyNoNoise_A" in the copy set. These two features are each the fifth feature in their respective sets). 

The data is first split into a training set and a testing set, with the holdout ratio being 0.3. The training set is then augmented by oversampling the underrepresented class(es) with SMOTENC, a variant of the synthetic minority over-sampling technique that accounts for both numerical and categorical features in its generation of synthetic data. 

Over the entire augmented training set, each numeric feature is standardized to have zero mean and unit variance. The mean and standard deviation parameters obtained from the training dataset are then used to similarly scale the testing dataset.

Finally, for each patient, the remaining command and copy features are stacked on top of each other, to create a \[2 x 175\] feature map.

## Convolutional Neural Network

The two-dimensional neural network is constructed in PyTorch. It consists of an input head, two residual blocks with skip connections, one 1x1 convolution, and one fully-connected output node.

### Input Head

* 2D convolution (n_filters=24, kernel_size=(3, 5), stride=1, padding=(1, 2))
* Batch normalization
* ReLU
* 2D convolution (n_filters=8, kernel_size=(3, 5), stride=1, padding=(1, 2))
* Batch normalization
* ReLU

### Residual Block 1
* 2D convolution (n_filters=8, kernel_size=(3, 5), stride=1, padding=(1, 2))
* Batch normalization
* ReLU
* 2D convolution (n_filters=8, kernel_size=(3, 5), stride=1, padding=(1, 2))
* Batch normalization
* Addition layer summing the block input with the output of the previous BatchNorm layer
* ReLU
* 2D convolution (n_filters=8, kernel_size=(3, 5), stride=(1, 2), padding=(1, 2))
* Batch normalization
* ReLU

### Residual Block 2
* 2D convolution (n_filters=8, kernel_size=(3, 3), stride=1, padding=(1, 1))
* Batch normalization
* ReLU
* 2D convolution (n_filters=8, kernel_size=(3, 3), stride=1, padding=(1, 1))
* Batch normalization
* Addition layer summing the block input with the output of the previous BatchNorm layer
* ReLU
* 2D convolution (n_filters=8, kernel_size=(3, 3), stride=(1, 2), padding=(1, 1))
* Batch normalization
* ReLU

### 1x1 Convolution
* 2D convolution (n_filters=2, filter_size=(1, 1), stride=1, padding=0)
* Batch normalization
* ReLU

### Fully-connected Output
* One densely-connected output node (Linear activation)

## Training
The neural network is trained using cross-entropy loss. The learning rate is optimized using the Adam optimizer with an initial learning rate of 0.001, betas=(0.9, 0.999), and an L2 regularization parameter 0.0005. The learning rate is reduced by a factor of ten after the 25th and 40th epochs. Training is performed over a total of 50 epochs.

## Results



