function [lgraph, options] = config_resnet50(number_of_classes, val_imds)
    pretrained_model = resnet50;

    lgraph = layerGraph(pretrained_model);
    lgraph = removeLayers(lgraph,{'fc1000_softmax', 'ClassificationLayer_fc1000'});

    new_layers = [
        fullyConnectedLayer(number_of_classes, 'Name', 'FC11')
        softmaxLayer('Name','SoftMax')

        classificationLayer('Name','class')
        ];

    lgraph = addLayers(lgraph, new_layers);
    lgraph = connectLayers(lgraph, 'fc1000', 'FC11');
    
    options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'ValidationData', val_imds, ...
    'ValidationFrequency', 100, ...
    'ValidationPatience', 25, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress');
end

