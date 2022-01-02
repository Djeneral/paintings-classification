function [lgraph, options] = config_densenet201(number_of_classes, val_imds)
    pretrained_model = densenet201;

    lgraph = layerGraph(pretrained_model);
    lgraph = removeLayers(lgraph,{'fc1000', 'fc1000_softmax', 'ClassificationLayer_fc1000'});

    new_layers = [
        fullyConnectedLayer(number_of_classes, 'Name', 'FC_final')
        softmaxLayer('Name', 'SoftMax')

        classificationLayer('Name', 'class')
        ];

    lgraph = addLayers(lgraph, new_layers);
    lgraph = connectLayers(lgraph, 'avg_pool', 'FC_final');
    
    options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 16, ...
    'ValidationData', val_imds, ...
    'ValidationFrequency', 100, ...
    'ValidationPatience', 15, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress');
end

