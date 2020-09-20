function M = perform_calibration(grid, models, params)
    % 1. Perform LABOO calibration procedure and 2. Learn a combination
    % matrix M which multiplexes the detection results (by compiling
    % co-occurrence statistics on true positives) 
    %
    % Copyright (C) 2011-12 by Tomasz Malisiewicz
    % All rights reserved.
    % 
    % This file is part of the Exemplar-SVM library and is made
    % available under the terms of the MIT license (see COPYING file).
    % Project homepage: https://github.com/quantombone/exemplarsvm

    % Perform calibration
    betas = perform_platt_calibration(grid, models, params);

    % Estimate the co-occurrence matrix M
    if ~(isfield(params, 'SKIP_M') && params.SKIP_M == 1)
        M = estimate_M(grid, models, params);
    end

    %concatenate results
    M.betas = betas;
end

function [betas] = perform_platt_calibration(grid, models, params)
    % Perform calibration by learning the sigmoid parameters (linear
    % transformation of svm scores) for each model independently.
    
    if nargin < 1 || isempty(grid) || isempty(models)
        fprintf(1,'Not enough arguments or empty grid or model\n');
        betas = [];
        return;
    end

    models_name = '';
    if length(models) >= 1 && isfield(models{1}, 'models_name') && isstr(models{1}.models_name)
        models_name = models{1}.models_name;
    end

    final_dir = sprintf('%s/models',params.dataset_params.localdir);

    if ~exist(final_dir','dir')
        mkdir(final_dir);
    end

    final_file = sprintf('%s/%s-betas.mat', final_dir, models_name);
    if file_exists(final_file)
        fprintf(1,'Loading final file %s\n', final_file);
        res = load_keep_trying(final_file);
        betas = res.betas;
        return;
    end

    for i = 1:length(models)
        if ~isfield(models{i},'curid')
            models{i}.curid = '-1';
        end
    end

    targets = 1:length(models);

    targetc = models{1}.cls;

    for i = 1:length(grid)    
        if mod(i,100) == 0
            fprintf(1,'.');
        end
        cur = grid{i};
        %do not process grids with no bboxes
        if size(cur.bboxes, 1) == 0
            continue;
        end
        if size(cur.bboxes,1) >= 1
            cur.bboxes(:,5) = 1:size(cur.bboxes,1);    
            cur.coarse_boxes(:,5) = 1:size(cur.bboxes,1);    
            if ~isempty(cur.extras)
                cur.extras.os = cur.extras.maxos(cur.bboxes(:,5));
                try
                    cur.extras.os = cur.extras.os.* reshape(double(ismember(cur.extras.maxclass,targetc)), size(cur.extras.os));
                catch
                    keyboard
                end
            end
        end

        cur.bboxes(:,5) = grid{i}.index;
        cur.coarse_boxes(:,5) = grid{i}.index;

        bboxes{i} = cur.bboxes;

        %if we have overlaps, collect them
        if ~isempty(cur.extras)            
            % find the ground truth examples of the right category
            os{i} = cur.extras.maxos;
        else
            os{i} = zeros(size(bboxes{i},1),1);
        end
        scores{i} = cur.bboxes(:,7)';
    end

    ALL_bboxes = cat(1,bboxes{:});
    ALL_os = cat(1,os{:});
    
    % Pre-processing models for calibration
    for exid = 1:length(models)
        fprintf(1,'.');        
        hits = find((ALL_bboxes(:, 6) == exid));
        all_scores = ALL_bboxes(hits, end);
        all_os = ALL_os(hits, :);

        good_scores = all_scores(all_os >= 0.5);
        good_os = all_os(all_os >= 0.5);

        bad_scores = all_scores(all_os<.5);
        bad_os = all_os(all_os < 0.5);

        if length(good_os) <= 1 || isempty(bad_os)
            beta = [.1 100];
        else
            [~,bb] = sort(bad_scores, 'descend');
            curlen = min(length(bb), 10000*length(good_scores));
            bb = bb(round(linspace(1, length(bb), curlen)));

            bad_scores = bad_scores(bb);
            bad_os = bad_os(bb);
            all_scores = [good_scores; bad_scores];
            all_os = [good_os; bad_os];
            beta = learn_sigmoid(all_scores, all_os);
        end

        if beta(1) < 0.001
            fprintf(1, ['warning[perform_platt_calibration.m]: beta(1) is low']);
        end

        betas(exid,:) = beta;

        if (sum(ismember(exid,targets))==0)
            continue
        end 
    end

    fprintf('\nLoaded calibration parameters "betas", saving to %s\n', final_file);
    save(final_file, 'betas');
end

function M = estimate_M(grid, models, params)
    % Given a bunch of detections, learn the M boosting matrix, which
    % makes a final boxes's score depend on the co-occurrence of certain
    % "friendly" detections
    
    neighbor_thresh = params.calibration_neighbor_thresh;
    count_thresh = params.calibration_count_thresh;

    dir_path = sprintf('%s/models',params.dataset_params.localdir);
    file_path = sprintf('%s/%s-M.mat', dir_path, models{1}.models_name);

    if file_exists(file_path)
        fprintf(1,'Loading final file %s\n', file_path);
        res = load_keep_trying(file_path);
        M = res.M;
        return;
    end

    if isempty(grid)
        error('Found no images of type %s\n',results_directory)
    end

    excurids = cellfun2(@(x)x.curid,models);
    boxes = cell(1,length(grid));
    maxos = cell(1,length(grid));

    fprintf(1,' -Computing Box Features:');
    starter = tic;
    for i = 1:length(grid)
        current_id = grid{i}.curid;
        boxes{i} = grid{i}.bboxes;
        if size(boxes{i},1) == 0
            if ~isempty(grid{i}.extras)
                maxos{i} = [];
            end      
            continue
        end
        
        calib_boxes = boxes{i};
        calib_boxes(:, end) = calib_boxes(:, end) + 1;

        %Threshold at the target value specified in parameters
        oks = find(calib_boxes(:,end) >= params.calibration_threshold);
        boxes{i} = calib_boxes(oks,:);
        if ~isempty(grid{i}.extras)
            fprintf(fileid, 'grid[i].extras is not empty\n');
            maxos{i} = grid{i}.extras.maxos;
            maxos{i}(find(ismember(grid{i}.extras.maxclass, models{1}.cls)==0)) = 0;
            maxos{i} = maxos{i}(oks);
        else
            maxos{i} = zeros(size(boxes{i},1),1);
        end
        
        % Remove firings on self-image, these create artificially high
        % scores.
        badex = find(ismember(excurids, current_id));
        badones = ismember(boxes{i}(:,6), badex);
        boxes{i}(badones,:) = [];
        if ~isempty(maxos{i})
            maxos{i}(badones) = [];
        end
    end

    lens = cellfun(@(x)size(x,1),boxes);
    boxes(lens == 0) = [];
    maxos(lens == 0) = [];

    K = length(models);
    os = cat(1,maxos{:})';

    scores = cellfun2(@(x)x(:,end)',boxes);
    scores = [scores{:}];

    xraw = cell(length(boxes),1);
    allboxes = cat(1,boxes{:});

    for i = 1:length(boxes)
        fprintf(1,'.');
        xraw{i} = get_M_features(boxes{i}, K, neighbor_thresh);
    end
    x = [xraw{:}];

    exemplar_ids = allboxes(:, 6);
    exemplar_ids(allboxes(:, 7) == 1)= exemplar_ids(allboxes(:, 7) == 1) + length(models);

    fprintf(1,'took %.3fsec\n',toc(starter));

    fprintf(1,' -Learning M by counting: ');
    starter = tic;

    M = learn_M_counting(x, exemplar_ids, os, count_thresh);
    fprintf(1,'took %.3fsec\n',toc(starter));

    M.neighbor_thresh = neighbor_thresh;
    M.count_thresh = count_thresh;

    r = cell(length(xraw), 1);
    fprintf(1,' -Applying M to %d images: ',length(xraw));
    starter = tic;
    for j = 1:length(xraw)
        r{j} = apply_M(xraw{j}, boxes{j}, M);
    end

    r = [r{:}];
    [~, bb] = sort(r,'descend');
    goods = os > 0.5;

    res = cumsum(goods(bb))./(1:length(bb));
    M.score = mean(res);
    fprintf(1,'took %.3fsec\n', toc(starter));

    if params.dataset_params.display == 1
        display_calibration(scores, os, r)
    end

    fprintf(1,'Computed M, saving to %s\n',file_path);
    save(file_path,'M');
end

function M = learn_M_counting(x, exids, os, count_thresh)
    % Learn M matrix by counting activations on positives.

    N = size(x, 2);
    K = size(x, 1);
    C = zeros(K, K);

    for i = 1:N
        cur = find(x(:,i)>0);  
        C(cur, exids(i)) = C(cur, exids(i)) + os(i)*(os(i) >= count_thresh) / length(cur);
    end

    for i = 1:K
        M.w{i} = C(:,i);
        M.b{i} = 0;
    end

    M.C = sparse(C);
end

function beta = learn_sigmoid(scores, os)
    % Fits a sigmoid to the scores verus overlap scores.
    % 1 / (1 + exp(-beta1 * (x - beta2)))
    %
    % Examples with high scores and high os's will "fit" well.  This is a 
    % soft way of counting the number of "good" detections before the first 
    % bad one.
    %
    % Note : os >= 0.5 is positive, os <= 0.2 is negative, rest is don't
    % care.

    x = scores;
    y = os;

    % Overlap score greater than or equal to 0.5 is 1 and less than or 
    % equal to 0.2 is 0
    y(y >= 0.5) = 1;
    y(y <= 0.2) = 0;
    
    % Overlap score between 0.2 and 0.5 is don't care
    bads = y > 0.2 & y < 0.5;
    y(bads) = [];
    x(bads) = [];

    reg_constant = .000001;
    fun = @(beta)robust_loss(1./(1+exp(-beta(1)*(x-beta(2))))-y)+ reg_constant*beta(1).^2;

    guess2 = 100;
    if sum(y > 0.5) > 0
        guess2 = mean(x( y > 0.5));
    end

    beta = [3.0 guess2];
    beta = fminsearch(fun, beta, optimset('MaxIter',10000, 'MaxFunEvals',10000, 'Display','off'));
end

function r = robust_loss(d)
    r = mean(d.^2);
end

function display_calibration(scores, os, r)
    figure(4)
    subplot(1,2,1)
    plot(scores, os, 'r.','MarkerSize',12)
    xlabel('Detection Score')
    ylabel('OS wrt gt')
    title('w/o calibration')

    subplot(1,2,2)
    plot(r,os,'r.','MarkerSize',12)
    xlabel('Detection Score')
    ylabel('OS wrt gt')
    title('w/ M-matrix')
    drawnow
    snapnow

    figure(5)
    clf
    [~, bb] = sort(scores,'descend');
    plot(cumsum(os(bb)>.5)./(1:length(os)),'r-','LineWidth',3)
    hold on;
    [~, bb] = sort(r, 'descend');
    plot(cumsum(os(bb)>.5)./(1:length(os)),'b--','LineWidth',3)
    xlabel('#instances Recalled')
    ylabel('Precision')
    title('M-matrix estimation Precision-Recall');
    legend('no matrix','matrix')
    drawnow
    snapnow
end