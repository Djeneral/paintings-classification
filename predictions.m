number_of_classes = 22;

if number_of_classes == 11
    train_imds = imageDatastore('dataset_11/train', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    val_imds = imageDatastore('dataset_11/validation', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    test_imds = imageDatastore('dataset_11/test', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
else
    train_imds = imageDatastore('dataset_11/train', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    val_imds = imageDatastore('dataset_11/validation', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    test_imds = imageDatastore('dataset_11/test', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
end

if number_of_classes == 11
    results = [test_imds.Labels];
    net = load('results/GoogleNet/11 classes/net.mat').net;
    results = [results classify(net,test_imds)];
    net = load('results/ResNet50/11 classes/net.mat').net;
    results = [results classify(net,test_imds)];
    net = load('results/ResNet101/11 classes/net.mat').net;
    results = [results classify(net,test_imds)];
    net = load('results/DenseNet201/11 classes/net.mat').net;
    results = [results classify(net,test_imds)];
    writematrix(results,'11 classes validation.csv')
    
else
    results = [test_imds.Labels];
    net = load('results/GoogleNet/21 classes/net.mat').net;
    results = [results classify(net,test_imds)];
    net = load('results/ResNet50/21 classes/net.mat').net;
    results = [results classify(net,test_imds)];
    net = load('results/ResNet101/21 classes/net.mat').net;
    results = [results classify(net,test_imds)];
    net = load('results/DenseNet201/21 classes/net.mat').net;
    results = [results classify(net,test_imds)];
    writematrix(results','21 classes validation.csv')
    
end