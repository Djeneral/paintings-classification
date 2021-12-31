%% Read datasets
train_imds = imageDatastore('dataset_22/train', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
val_imds = imageDatastore('dataset_22/validation', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
test_imds = imageDatastore('dataset_22/test', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
%% Create network and options
[lgraph, options] = config_resnet50(21, val_imds);
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