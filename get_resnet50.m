function [lgraph] = get_resnet50(number_of_classes)
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

end

