function final = pool_exemplar_dets(grid, models, M, params)
    % Perform detection post-processing and pool detection boxes
    % (which will then be ready to go into the PASCAL evaluation code)
    % If there are overlap scores associated with boxes, then they are
    % also kept track of propertly, even after NMS.
    % 
    % If M is empty, then just NMS is performed
    % If M has neighbor_thresh defined, then we apply the
    % calibration-matrix
    % If M has betas defined, then do platt-calibration
    %
    % Copyright (C) 2011-12 by Tomasz Malisiewicz
    % All rights reserved.
    % 
    % This file is part of the Exemplar-SVM library and is made
    % available under the terms of the MIT license (see COPYING file).
    % Project homepage: https://github.com/quantombone/exemplarsvm

    bboxes = cell(1,length(grid));

    for i = 1:length(grid)  
        bboxes{i} = grid{i}.bboxes;
    end

    raw_boxes = bboxes;

    % Perform score rescaling
    %1. no scaling
    %2. platt's calibration (sigmoid scaling)
    %3. raw score + 1

    if exist('M', 'var') && ~isempty(M) && isfield(M, 'betas') && ~isfield(M, 'neighbor_thresh')
        fprintf(1,'Applying betas to %d images:', length(bboxes));
        for i = 1:length(bboxes)
            %if neighbor thresh is defined, then we are in M-mode boosting
            if size(bboxes{i}, 1) == 0
                continue
            end
            calib_boxes = calibrate_boxes(bboxes{i}, M.betas); 
            oks = find(calib_boxes(:, end) > params.calibration_threshold);
            calib_boxes = calib_boxes(oks,:);
            bboxes{i} = calib_boxes;
        end
    elseif exist('M','var') && ~isempty(M) && isfield(M,'neighbor_thresh')
        fprintf(1,'Applying M-matrix to %d images:',length(bboxes));
        starter = tic;
        nbrlist = cell(length(bboxes),1);
        for i = 1:length(bboxes)
            fprintf(1,'.');
            if size(bboxes{i},1) == 0
                continue
            end
            bboxes{i}(:,end) = bboxes{i}(:,end)+1;
            [xraw, nbrlist{i}] = get_M_features(bboxes{i}, length(models), M.neighbor_thresh);
            r2 = apply_M(xraw, bboxes{i}, M);
            bboxes{i}(:,end) = r2;
        end
        fprintf(1,'took %.3fsec\n',toc(starter));
    else
        fprintf(1,'No betas, No M-matrix, no calibration\n');
    end

    os_thresh = 0.3;
    fprintf(1, 'Applying NMS (OS thresh = %.3f)\n',os_thresh);
    for i = 1:length(bboxes)
        if size(bboxes{i}, 1) > 0
            bboxes{i}(:,5) = 1:size(bboxes{i}, 1);
            bboxes{i} = nms(bboxes{i}, os_thresh);
            if exist('nbrlist','var')
                nbrlist{i} = nbrlist{i}(bboxes{i}(:,5));
            end
            bboxes{i}(:,5) = 1:size(bboxes{i},1);
        end
    end
    
    % Propagate scores onto raw boxes
    if params.calibration_propagate_onto_raw && exist('M','var') && ~isempty(M) && isfield(M,'betas')
        fprintf(1,'Propagating scores onto raw detections\n');
        for i = 1:length(bboxes)
            if size(bboxes{i}, 1) > 0
                allMscores = bboxes{i}(:, end);
                calib_boxes = calibrate_boxes(raw_boxes{i}, M.betas);
                beta_scores = calib_boxes(:, end);
                overlap_score = iou(bboxes{i}, raw_boxes{i});
                for j = 1:size(overlap_score, 1)
                    curscores = (overlap_score(j, :) > 0.5) .* beta_scores';
                    [aa,bb] = max(curscores);
                    bboxes{i}(j, :) = raw_boxes{i}(bb, :);
                    bboxes{i}(j, end) = aa;
                end
                bboxes{i}(:, end) = allMscores;
            end
        end
    end

    % Return unclipped boxes for transfers
    final.unclipped_boxes = bboxes;
    
    % Clip boxes to image dimensions
    for i = 1:length(bboxes)
        bboxes{i} = clip_to_image(bboxes{i}, grid{i}.imbb);
    end
    final.final_boxes = bboxes;

    % Create a string which summarizes the pooling type
    calib_string = '';
    if exist('M','var') && ~isempty(M) && isfield(M,'betas')
        calib_string = '-calibrated';
    end

    if exist('M', 'var') && ~isempty(M) && isfield(M, 'betas') && isfield(M, 'w')
        calib_string = [calib_string '-M'];
    end

    final.calib_string = calib_string;    
end

function top = nms(boxes, overlap)
    % Performs non maximum suppression with the given overlap.

    if isempty(boxes)
        top = [];
        return;
    end

    x1 = boxes(:,1);
    y1 = boxes(:,2);
    x2 = boxes(:,3);
    y2 = boxes(:,4);
    s = boxes(:,end);

    area = (x2-x1+1) .* (y2-y1+1);
    [~, I] = sort(s);

    pick = s*0;
    counter = 1;
    while ~isempty(I)
        last = length(I);
        i = I(last);  
        pick(counter) = i;
        counter = counter + 1;

        xx1 = max(x1(i), x1(I(1:last-1)));
        yy1 = max(y1(i), y1(I(1:last-1)));
        xx2 = min(x2(i), x2(I(1:last-1)));
        yy2 = min(y2(i), y2(I(1:last-1)));

        w = max(0.0, xx2-xx1+1);
        h = max(0.0, yy2-yy1+1);

        o = w.*h ./ area(I(1:last-1));

        I([last; find(o>overlap)]) = [];
    end

    pick = pick(1:(counter-1));
    top = boxes(pick,:);
end

function bboxes = calibrate_boxes(bboxes, betas)
    % Applies learned Platt-calibration parameters onto raw detection
    % scores in boxes.
    %
    % Parameters
    % ----------
    % bboxes : Bounding box coordinates and associated attributes
    % betas  : Learned Platt-calibration parameters
    %
    % Returns
    % -------
    % bboxes : Bounding box coordinates with calibrated scores.
    
    if size(bboxes,1) == 0
        return;
    end

    exemplar_id = bboxes(:,6);
    
    if ~exist('betas','var') || isempty(betas)
        betas(exemplar_id, 1) = 1;
        betas(exemplar_id, 2) = 0;
    end
    
    scores = betas(exemplar_id, 1) .* (bboxes(:,end) - betas(exemplar_id,2));
    bboxes(:, end) = 1./(1 + exp(-scores));
end