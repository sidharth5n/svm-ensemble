function [m] = mine_train_iteration(m, training_function)
    % ONE ITERATION OF: Mine negatives until cache is full and update the current
    % classifier using training_function (do_svm, do_rank, ...). m must
    % contain the field m.train_set, which indicates the current
    % training set of negative images
    % Returns the updated model (where m.mining_queue is updated mining_queue)
    %
    % Copyright (C) 2011-12 by Tomasz Malisiewicz
    % All rights reserved.
    % 
    % This file is part of the Exemplar-SVM library and is made
    % available under the terms of the MIT license (see COPYING file).
    % Project homepage: https://github.com/quantombone/exemplarsvm

    % Start wtrace (trace of learned classifier parameters across
    % iterations) with first round classifier, if not present already
    
    if ~isfield(m.model,'wtrace')
        m.model.wtrace{1} = m.model.w;
        m.model.btrace{1} = m.model.b;
    end

    if isempty(m.mining_queue)
        fprintf(1,' ---Null mining queue, not mining!\n');
        return;
    end

    %If the skip is enabled, we just update the model
    if m.mining_params.train_skip_mining == 0
        [hard_negatives, m.mining_queue, mining_stats] = mine_negatives({m}, m.mining_queue, m.train_set, m.mining_params);
        m = add_new_detections(m, cat(2,hard_negatives.xs{1}{:}), cat(1,hard_negatives.bbs{1}{:}));
    else
        mining_stats.num_visited = 0;
        fprintf(1,'WARNING: train_skip_mining = 0, just updating model\n');  
    end

    m = update_the_model(m, mining_stats, training_function);

    if isfield(m,'dataset_params') && m.dataset_params.display == 1
        dump_figures(m);
    end
end

function [m] = update_the_model(m, mining_stats, training_function)
    % Updates the current SVM, keep max number of svs, and show the results
    
    if ~isfield(m,'mining_stats')
        m.mining_stats{1} = mining_stats;
    else
        m.mining_stats{end+1} = mining_stats;
    end

    m = training_function(m);

    % Append new w to trace
    m.model.wtrace{end+1} = m.model.w;
    m.model.btrace{end+1} = m.model.b;
end

function dump_figures(m)

    figure(2)
    clf
    Isv1 = show_det_stack(m,7);

    imagesc(Isv1)
    axis image
    axis off
    iter = length(m.model.wtrace) - 1;
    title(sprintf('Ex %s.%d.%s SVM-iter = %03d', m.curid, m.objectid, m.cls, iter))
    drawnow
    snapnow

    if m.mining_params.dump_images == 1 || m.mining_params.dump_last_image == 1 && m.iteration == m.mining_params.train_max_mine_iterations
        imwrite(Isv1, sprintf('%s/%s.%d_iter_I = %05d.png', m.mining_params.final_directory, m.curid, m.objectid, m.iteration), 'png');
    end
end

function m = add_new_detections(m, xs, bbs)
    % Adds current detections (xs, bbs) to the model struct (m)
    % pruning away duplicates and sorting by score.

    % First iteration might not have support vector information stored
    if ~isfield(m.model, 'svxs') || isempty(m.model.svxs)
        m.model.svxs = [];
        m.model.svbbs = [];
    end

    m.model.svxs = cat(2, m.model.svxs, xs);
    m.model.svbbs = cat(1, m.model.svbbs, bbs);

    % Create a unique string identifier for each of the supports
    names = cell(size(m.model.svbbs,1),1);
    for i = 1:length(names)
        bb = m.model.svbbs(i,:);
        names{i} = sprintf('%d.%.3f.%d.%d.%d',bb(11),bb(8), bb(9),bb(10),bb(7));
    end

    [~, subset, ~] = unique(names);
    m.model.svbbs = m.model.svbbs(subset, :);
    m.model.svxs = m.model.svxs(:, subset);

    [~, bb] = sort(m.model.w(:)'*m.model.svxs, 'descend');
    m.model.svbbs = m.model.svbbs(bb, :);
    m.model.svxs = m.model.svxs(:, bb);
end

function Isv = show_det_stack(m, K2, K1)
    % Creates a K1 x K2 image visualizing the detection windows and
    % information about trained exemplar m. The visualization shows top
    % negative support vectors as well as top detection from any set.
    %
    % The first row is exemplar image, w+, w-, mean 0, ... , mean N. 
    % Second row first icon starts the top detections.
    
    if ~exist('K2','var')
        K1 = 5;
        K2 = 5;
    elseif ~exist('K1','var')
        K2 = max(K2, 5);
        K1 = K2;
    else
        K1 = max(K1, 5);
        K2 = max(K2, 5);
    end
    
    if sum(m.model.w(:) < 0) == 0 || sum(m.model.w(:) > 0) == 0
        fprintf(1,'Note, squaring visualization\n');
        m.model.w = (abs(m.model.w)).^2;
    end

    % sort by score
    if isfield(m.model,'svxs') && numel(m.model.svxs) > 0
        if isfield(m.mining_params, 'dfun') && m.mining_params.dfun == 1
            r = m.model.w(:)'*bsxfun(@minus,m.model.svxs,m.model.x(:,1)).^2 - m.model.b;
        else
            r = m.model.w(:)'*m.model.svxs - m.model.b;
        end
        [~, indices] = sort(r,'descend');
        m.model.svbbs = m.model.svbbs(indices, :);
        m.model.svxs = m.model.svxs(:, indices);
    end

    N = min(size(m.model.svbbs,1), K1 * K2);

    svbbs = m.model.svbbs(1:N,:);

    if N > 0
        ucurids = unique(svbbs(:, 11));
    else
        ucurids = [];
    end
    
    svims = cell(N, 1);

    for i = 1:length(ucurids)
        Ibase = load_image(m.train_set{ucurids(i)});
        hits = find(svbbs(:,11) == ucurids(i));
        for j = 1:length(hits)
            cb = svbbs(hits(j), :);
            mypad = max([0, 1 - cb(1), 1 - cb(2), cb(3) - size(Ibase, 2), cb(4) - size(Ibase, 1)]);
            PADDER = round(mypad) + 2;
            I = pad_image(Ibase,PADDER);
            indices = round(cb + PADDER);
            svims{hits(j)} = I(indices(2):indices(4), indices(1):indices(3), :);
            if cb(7) == 1
                svims{hits(j)} = flip_image(svims{hits(j)});
            end
        end
    end

    %Get the exemplar frame icon
    Ibase = get_exemplar_icon(m);
    
    newsize = [size(Ibase, 1) size(Ibase, 2)];
    newsize = 100/newsize(1) * newsize;
    newsize = round(newsize) + 10;

    svims = cellfun2(@(x)max(0.0,min(1.0,imresize(x,newsize))),svims);
    svimstack = cat(4,svims{:});
    SSS = size(svimstack, 4)*~isempty(svimstack);
    
    NSHIFT_BASE = 3;
    NMS = K2 - NSHIFT_BASE;
    if NMS == 1
        cuts(1) = SSS;
    else
        cuts = round(linspace(1,SSS,NMS+1));
        cuts = cuts(2:end);
    end

    for i = 1:length(cuts)
        if SSS > 0
            mss{i} = mean(svimstack(:,:,:,1:cuts(i)),4);
        else
            mss{i} = zeros(newsize(1), newsize(2),3);
        end
    end
    
    PADSIZE = 5;
    for i = 1:numel(svims)
        svims{i} = pad_image(svims{i}, PADSIZE, [1 1 1]);
    end

    if length(svims) < K1*K2
        svims{K1*K2} = zeros(newsize(1) + PADSIZE*2, newsize(2) + PADSIZE*2, 3);
    end

    for j = (N+1):(K1*K2)
        svims{j} = zeros(newsize(1)+PADSIZE*2, newsize(2)+PADSIZE*2,3);
    end

    pos_picture = HOGpicture(m.model.w);
    neg_picture = HOGpicture(-m.model.w);

    pos_picture = jettify(imresize(pos_picture, newsize,'nearest'));
    neg_picture = jettify(imresize(neg_picture, newsize,'nearest'));

    NSHIFT = length(mss) + NSHIFT_BASE;
    svims((NSHIFT+1):end) = svims(1:end-NSHIFT);

    %ex goes in slot 1
    svims{1} = max(0.0, min(1.0,imresize(Ibase, [size(pos_picture,1), size(pos_picture, 2)])));
    svims{2} = pos_picture;
    svims{3} = neg_picture;

    svims(NSHIFT_BASE + (1:length(mss))) = mss;

    for q = 1:(NSHIFT_BASE+length(mss))
        svims{q} = pad_image(svims{q},PADSIZE,[1 1 1]);
    end

    svims = reshape(svims, K1, K2)';
    for j = 1:K2
        svrows{j} = cat(2,svims{j,:});
    end

    Isv = cat(1,svrows{:});
end