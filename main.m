%% Read datasets
train_imds = imageDatastore('dataset_11/train', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
val_imds = imageDatastore('dataset_11/validation', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
test_imds = imageDatastore('dataset_11/test', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'ValidationData', val_imds, ...
    'ValidationFrequency', 331, ...
    'ValidationPatience', 5, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress');
%% Create network
pretrained_model = resnet50;

lgraph = layerGraph(pretrained_model);
lgraph = removeLayers(lgraph,{'fc1000_softmax', 'ClassificationLayer_fc1000'});

new_layers = [
    fullyConnectedLayer(11, 'Name', 'FC11')
    softmaxLayer('Name','SoftMax')
    
    classificationLayer('Name','class')
    ];

lgraph = addLayers(lgraph, new_layers);
lgraph = connectLayers(lgraph, 'fc1000', 'FC11');

analyzeNetwork(lgraph);
%% Train network
net = trainNetwork(train_imds, lgraph, options);

%% Confustion matrix
pred = classify(net,train_imds);
figure, plotconfusion(train_imds.Labels, pred);

pred = classify(net,val_imds);
figure, plotconfusion(val_imds.Labels, pred);

pred = classify(net,test_imds);
figure, plotconfusion(test_imds.Labels, pred);