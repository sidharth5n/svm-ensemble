function allbbs = show_top_dets(test_struct, ~, test_set, models, params, k, set_name)
    % Show top k detections for [models] where [grid] is the set of
    % detections from the set [test_set], [test_struct] contains final
    % boxes after pooling and calibration. If [dataset_params.localdir] is
    % present, then results are saved based on naming convention into a
    % "images" subfolder. maxk is the number of top detections we show
    %
    %
    % Copyright (C) 2011-12 by Tomasz Malisiewicz
    % All rights reserved.
    % 
    % This file is part of the Exemplar-SVM library and is made
    % available under the terms of the MIT license (see COPYING file).
    % Project homepage: https://github.com/quantombone/exemplarsvm
    
    allbbs = [];

    if ~exist('set_name','var')
        set_name = '';
    end

    %maxk is the maximum number of top detections we display
    if ~exist('maxk','var')
        k = 20;
    end
    
    final_boxes = test_struct.unclipped_boxes;

    bbs = cat(1, final_boxes{:});

    % sort detections by score
    try
        [~,bb] = sort(bbs(:,end), 'descend');
    catch
        bb = [];
    end

    counter = 1;

    for k = 1:min(k, length(bb))
        try
            corr = test_struct.rc(k);
        catch
            corr = 0;
        end

        dir_path = sprintf('%s/images/%s.%s%s/', params.dataset_params.localdir, set_name, models{1}.models_name, test_struct.calib_string);
        if ~exist(dir_path,'dir')
            mkdir(dir_path);
        end

        file_name = sprintf('%s/%05d.png', dir_path, k);
        if file_exists(file_name)
            counter = counter + 1;
            fprintf(1,'Already showed detection # %d, score=%.3f\n', k, bbs(bb(counter), end));
            continue
        end

        fprintf(1,'Showing detection # %d, score = %.3f\n', k, bbs(bb(counter), end));
        allbbs(k,:) = bbs(bb(counter),:);

        I = load_image(test_set{bbs(bb(counter),11)});

        % Use the raw detection
        allbb = bbs(bb(counter),:);

        figure(1)
        clf

        show_transfer_figure(I, models, allbb, corr);
        axis image
        drawnow
        snapnow

        print(gcf,'-dpng',file_name);
        counter = counter + 1;    
    end
end

function show_transfer_figure(I, models, topboxes, corr)
    % Shows a figure with the detections of the exemplar svm model
    
    if ndims(topboxes) == 2
        topboxes = topboxes(1);
    end
    
    exemplar_id = topboxes(6);
    Iex = get_exemplar_icon(models{exemplar_id}, topboxes(7), false);
    
    subplot(1, 3, 2);
    imagesc(Iex)
    title(sprintf("Exemplar Image %d", exemplar_id))
    axis image
    axis off
    
    hogpic = HOGpicture(models{exemplar_id}.model.w);
    hogpic = jettify(hogpic);
    if top_boxes(6) == 1
        hogpic = flip_image(hogpic);
    end
    hogpic = imresize(hogpic,[size(Iex1,1) size(Iex,2)], 'nearest');
    
    subplot(1, 3, 1);
    imagesc(hogpic)
    title(sprintf("Exemplar-SVM %s %d", models{exemplar_id}.cls, exemplar_id))
    axis image
    axis off
    
    if corr == 0
        curcolor = [1 0 0];
    else
        curcolor = [0 1 0];
    end
    
    clipped_top = clip_to_image(topboxes(1:4), [1 1 size(extraI,2) size(extraI,1)]);
    clipped_top([1 2]) = clipped_top([1 2]) + 2;
    clipped_top([3 4]) = clipped_top([3 4]) - 2;
    I(clipped_top(1):clipped_top(3), clipped_top(2):clipped_top(4), :) = I(clipped_top(1):clipped_top(3), clipped_top(2):clipped_top(4), :) + curcolor;
    
    subplot(1, 3, 3);
    imagesc(I);
    title(sprintf("%s: %0.3f}", models{exemplar_id}.cls, topboxes(end)))
    axis image
    axis off
    
end