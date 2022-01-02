function [lgraph, options] = config_vgg19(number_of_classes, val_imds)
    pretrained_model = vgg19;

    lgraph = layerGraph(pretrained_model.Layers);
    lgraph = removeLayers(lgraph,{'prob', 'output'});

    new_layers = [
        fullyConnectedLayer(number_of_classes, 'Name', 'FC11')
        softmaxLayer('Name','SoftMax')

        classificationLayer('Name','class')
        ];

    lgraph = addLayers(lgraph, new_layers);
    lgraph = connectLayers(lgraph, 'fc8', 'FC11');
    
    options = trainingOptions('adam', ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 16, ...
    'ValidationData', val_imds, ...
    'ValidationFrequency', 100, ...
    'ValidationPatience', 25, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress');
end

