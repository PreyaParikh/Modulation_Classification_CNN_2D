function modClassNet = helperModClassCNN_VBX(modulationTypes, sps, spf)
%HELPERMODCLASSCNN_VBX Build the 2-D CNN used for modulation classification.
%   NET = HELPERMODCLASSCNN_VBX(MODULATIONTYPES, SPS, SPF) returns a
%   layer array, NET, describing a 2-D convolutional network for
%   modulation classification.
%
%   MODULATIONTYPES - Categorical list of modulation types to classify.
%   SPS             - Samples per symbol.
%   SPF             - Samples per frame.
%
%   Input format
%   ------------
%   The network expects input frames of size [2 x SPF x 1], where the
%   first dimension holds the in-phase (I) and quadrature (Q) components
%   and the second dimension holds the time samples. This 2-D layout
%   (as opposed to a 1-D I/Q vector) is used so the trained network can
%   be exported to ONNX and deployed with VectorBlox.

numModTypes = numel(modulationTypes);

netWidth = 1;
poolSize = 2;

modClassNet = [

    imageInputLayer([2 spf 1], ...
        Normalization="none", ...
        Name="InputLayer")

    convolution2dLayer([2 sps], 16*netWidth, ...
        Padding="same", ...
        Name="CNN1")
    batchNormalizationLayer(Name="BN1")
    reluLayer(Name="ReLU1")
    maxPooling2dLayer([1 poolSize], ...
        Stride=[1 poolSize], ...
        Name="MaxPool1")

    convolution2dLayer([2 sps], 32*netWidth, ...
        Padding="same", ...
        Name="CNN2")
    batchNormalizationLayer(Name="BN2")
    reluLayer(Name="ReLU2")
    maxPooling2dLayer([1 poolSize], ...
        Stride=[1 poolSize], ...
        Name="MaxPool2")

    convolution2dLayer([2 sps], 48*netWidth, ...
        Padding="same", ...
        Name="CNN3")
    batchNormalizationLayer(Name="BN3")
    reluLayer(Name="ReLU3")
    maxPooling2dLayer([1 poolSize], ...
        Stride=[1 poolSize], ...
        Name="MaxPool3")

    convolution2dLayer([2 sps], 64*netWidth, ...
        Padding="same", ...
        Name="CNN4")
    batchNormalizationLayer(Name="BN4")
    reluLayer(Name="ReLU4")
    maxPooling2dLayer([1 poolSize], ...
        Stride=[1 poolSize], ...
        Name="MaxPool4")

    convolution2dLayer([2 sps], 32*netWidth, ...
        Padding="same", ...
        Name="CNN5")
    batchNormalizationLayer(Name="BN5")
    reluLayer(Name="ReLU5")

    globalAveragePooling2dLayer(Name="AP1")

    fullyConnectedLayer(numModTypes, ...
        Name="FC1")

    softmaxLayer(Name="SoftMax")
];

end
