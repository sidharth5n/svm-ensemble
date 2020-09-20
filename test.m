function test(models, M, models_name, params)
    % Evalues the models on the test set and shows top 20 detections.
    
    % Define test-set
    params.detect_exemplar_nms_os_threshold = 0.5;
    test_set_name = 'test';
    test_set = get_pascal_set(params.dataset_params, test_set_name);
    
    % Apply on test set
    test_grid = detect_imageset(test_set, models, params, test_set_name);

    % Apply calibration matrix to test-set results
    test_struct = pool_exemplar_dets(test_grid, models, M, params);

    % Show top 20 detections    
    show_top_dets(test_struct, test_grid, test_set, models, params,  20, test_set_name);

    % Perform the exemplar evaluation
    evaluate_pascal_voc(test_struct, test_grid, params, test_set_name, class, models_name);
    
end