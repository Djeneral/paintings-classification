function [lgraph, options] = config_googleNet(number_of_classes)
    pretrained_model = googlenet;

    lgraph = layerGraph(pretrained_model);
    lgraph = removeLayers(lgraph,{'output', 'prob', 'loss3-classifier'});

    new_layers = [
        fullyConnectedLayer(number_of_classes, 'Name', 'FC11')
        softmaxLayer('Name','SoftMax')

        classificationLayer('Name','class')
        ];

    lgraph = addLayers(lgraph, new_layers);
    lgraph = connectLayers(lgraph, 'pool5-drop_7x7_s1', 'FC11');
    
    options = trainingOptions('adam', ...
    'MaxEpochs', 40, ...
    'MiniBatchSize', 16, ...
    'ValidationData', val_imds, ...
    'ValidationFrequency', 100, ...
    'ValidationPatience', 100, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress');
end

