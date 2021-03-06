function [lgraph, options] = config_resnet101(number_of_classes, val_imds)
    pretrained_model = resnet101;

    lgraph = layerGraph(pretrained_model);
    lgraph = removeLayers(lgraph,{'prob', 'ClassificationLayer_predictions'});

    new_layers = [
        fullyConnectedLayer(number_of_classes, 'Name', 'FC11')
        softmaxLayer('Name','SoftMax')

        classificationLayer('Name','class')
        ];

    lgraph = addLayers(lgraph, new_layers);
    lgraph = connectLayers(lgraph, 'fc1000', 'FC11');
    
    options = trainingOptions('adam', ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 8, ...
    'ValidationData', val_imds, ...
    'ValidationFrequency', 200, ...
    'ValidationPatience', 20, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress');
end

