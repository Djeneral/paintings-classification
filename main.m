%%
train_imds = imageDatastore('dataset/train', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
val_imds = imageDatastore('dataset/validation', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
test_imds = imageDatastore('dataset/test', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%%
% layers = [
%     imageInputLayer([256, 256, 3], 'Name', 'Input')
%     
%     convolution2dLayer(3, 16, 'Padding', 'same', 'Stride', [1,1], 'Name', 'conv1')
%     reluLayer('Name','relu1')
%     maxPooling2dLayer(2, 'Stride', [2,2], 'Name', 'pool1')
%     
%     convolution2dLayer(3, 32, 'Padding', 'same', 'Stride', [1,1], 'Name', 'conv2')
%     reluLayer('Name','relu2')
%     maxPooling2dLayer(2, 'Stride', [2,2], 'Name', 'poo2')
%     
%     convolution2dLayer(3, 64, 'Padding', 'same', 'Stride', [1,1], 'Name', 'conv3')
%     reluLayer('Name','relu3')
%     maxPooling2dLayer(2, 'Stride', [2,2], 'Name', 'pool3')
%     
%     fullyConnectedLayer(128, 'Name', 'FC1')
%     reluLayer('Name','reluFC')
%     
%     fullyConnectedLayer(11, 'Name', 'FC2')
%     softmaxLayer('Name','SoftMax')
%     
%     classificationLayer('Name','class')
%     ];
% 
% lgraph = layerGraph(layers);
% analyzeNetwork(lgraph);
%% 
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'ValidationData', val_imds, ...
    'ValidationFrequency', 331, ...
    'ValidationPatience', 5, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress');
%%
pretrained_model = resnet50;

lgraph = layerGraph(pretrained_model);
lgraph = removeLayers(lgraph,{'fc1000_softmax', 'ClassificationLayer_fc1000'});

%analyzeNetwork(lgraph);

new_layers = [
    fullyConnectedLayer(11, 'Name', 'FC11')
    softmaxLayer('Name','SoftMax')
    
    classificationLayer('Name','class')
    ];

lgraph = addLayers(lgraph, new_layers);
lgraph = connectLayers(lgraph, 'fc1000', 'FC11');

analyzeNetwork(lgraph);
%%
net = trainNetwork(train_imds, lgraph, options);

%% 
pred = classify(net,test_imds);

figure,
plotconfusion(test_imds.Labels, pred);