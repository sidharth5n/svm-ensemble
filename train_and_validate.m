function [models, M, models_name, params] = main(class, data_directory, dataset_directory, results_directory)
    % Trains the model for the given class and performs evaluation and
    % testing.

    % Copyright (C) 2011-12 by Tomasz Malisiewicz
    % All rights reserved.
    %
    % This file is part of the Exemplar-SVM library and is made
    % available under the terms of the MIT license (see COPYING file).
    % Project homepage: https://github.com/quantombone/exemplarsvm

    addpath(genpath(pwd));

    if ~exist('class','var')
        class = 'car';
    end

    if ~exist('data_directory','var')
        data_directory = 'VOC2007/VOCdevkit';
    end

    if ~exist('dataset_directory','var')
        dataset_directory = 'VOC2007';
    end

    if ~exist('results_directory', 'var') || isempty(results_directory)
        results_directory = sprintf('%s%s%s', dataset_directory, '/results/', class);
    end
    
    dataset_params = get_voc_dataset(dataset_directory, data_directory, results_directory);

    % Set exemplar-initialization parameters
    params = get_default_params;
    params.model_type = 'exemplar';
    params.dataset_params = dataset_params;

    % Initialize exemplar stream
    stream_params.stream_set_name = 'trainval';
    stream_params.stream_max_ex = 10000;
    stream_params.must_have_seg = 0;
    stream_params.must_have_seg_string = '';
    % must be scene or exemplar;
    stream_params.model_type = 'exemplar';
    stream_params.cls = class;

    % Create an exemplar stream (list of exemplars)
    e_stream_set = get_pascal_stream(stream_params, dataset_params);
    
    % Define negative exemplars
    neg_set = get_pascal_set(dataset_params, 'train', class, -1);

    % Choose a models name to indicate the type of training run we are doing
    models_name = [class '-' params.init_params.init_type '.' params.model_type];

    initial_models = initialize_exemplars(e_stream_set, params, models_name);

    % Perform Exemplar-SVM training
    train_params = params;
    train_params.detect_max_scale = 0.5;
    train_params.detect_exemplar_nms_os_threshold = 1.0;
    train_params.detect_max_windows_per_exemplar = 100;
    
    % Train the exemplars and get updated models name
    [models, models_name] = train_exemplars(initial_models, neg_set, train_params);
    
    % Validation parameters
    val_params = params;
    val_params.detect_exemplar_nms_os_threshold = 0.5;
    val_params.gt_function = @load_gt_function;
    
    % Define validation set
    val_set_name = ['trainval'];
    val_set = get_pascal_set(dataset_params, val_set_name);

    % Apply trained exemplars on validation set
    val_grid = detect_imageset(val_set, models, val_params, val_set_name);

    % Perform Platt calibration and M-matrix estimation
    M = perform_calibration(val_grid, models, val_params);
end

    
