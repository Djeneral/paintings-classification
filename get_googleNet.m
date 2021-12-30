function [lgraph] = get_googleNet(number_of_classes)
    pretrained_model = googlenet;

    lgraph = layerGraph(pretrained_model);
    lgraph = removeLayers(lgraph,{'output', 'prob'});

    new_layers = [
        fullyConnectedLayer(number_of_classes, 'Name', 'FC11')
        softmaxLayer('Name','SoftMax')

        classificationLayer('Name','class')
        ];

    lgraph = addLayers(lgraph, new_layers);
    lgraph = connectLayers(lgraph, 'loss3-classifier', 'FC11');
end

