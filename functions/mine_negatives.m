function [hard_negatives, mining_queue, mining_stats] = mine_negatives(models, mining_queue, imageset, mining_params)
    % Finds hard negatives for the images in the stream or queue given the
    % models.
    % 
    % Input Data:
    % models: Kx1 cell array of models
    % mining_queue: the mining queue create from
    %    initialize_mining_queue(imageset)
    % imageset: the source of images (potentially already in pyramid feature
    %   format)
    % mining_params: the parameters of the mining/localization
    % procedure
    % 
    % Returned Data: 
    % hard_negatives: Kx1 cell array where hard_negatives{i} contains info for model i
    % hard_negatives contains:
    %   hard_negatives{:}.xs "features"
    %   hard_negatives{:}.bbs "bounding boxes"
    %
    % Copyright (C) 2011-12 by Tomasz Malisiewicz
    % All rights reserved.
    % 
    % This file is part of the Exemplar-SVM library and is made
    % available under the terms of the MIT license (see COPYING file).
    % Project homepage: https://github.com/quantombone/exemplarsvm
    
    if ~exist('mining_params','var')
        mining_params = get_default_params;
    end

    number_of_violating_images = 0;
    number_of_windows = zeros(length(models),1);

    violating_images = zeros(0,1);
    empty_images = zeros(0,1);

    mining_params.detect_save_features = 1;

    numpassed = 0;

    for i = 1:length(models)
        if ~isfield(models{i},'total_mines')
            models{i}.total_mines = 0;
        end
    end

    for i = 1:length(mining_queue)
        index = mining_queue{i}.index;
        I = load_image(imageset{index});
        [rs, ~] = detect(I, models, mining_params);

        if isfield(models{1}.mining_params, 'SOFT_NEGATIVE_MINING') && models{1}.mining_params.SOFT_NEGATIVE_MINING == 1
            for j=1:length(rs.bbs)
                if size(rs.bbs{j},1) > 0
                    top_det = rs.bbs{j}(1,:);
                    os = iou(rs.bbs{j},top_det);
                    goods = find(os < models{j}.mining_params.SOFT_NEGATIVE_MINING_OS);
                    rs.bbs{j} = rs.bbs{j}(goods,:);
                    rs.xs{j} = rs.xs{j}(goods);
                end
            end
        end

        numpassed = numpassed + 1;
        for q = 1:length(rs.bbs)
            if ~isempty(rs.bbs{q})
               rs.bbs{q}(:,11) = index;
            end
        end

        % Make sure we only keep 3 times the number of violating windows
        clear scores
        scores{1} = [];
        for q = 1:length(models)
            if ~isempty(rs.bbs{q})
                s = rs.bbs{q}(:, end);
                nviol = sum(s >= -1);
                [~, bb] = sort(s, 'descend');
                bb = bb(1:min(length(bb), ceil(nviol*mining_params.train_keep_nsv_multiplier)));
                rs.xs{q} = rs.xs{q}(bb);    
                scores{q} = cat(2,s);
            end
        end

        addon = '';
        supersize = sum(cellfun(@(x)length(x),scores));
        if supersize > 0
            addon = sprintf(', max = %.3f', max(cellfun(@(x)max_or_this(x,-1000),scores)));
        end
        fprintf(1,'Found %04d windows, image:%05d (#seen=%05d/%05d%s)\n', supersize, index, length(imageset)-length(mining_queue)+i, length(imageset), addon);

        % Increment how many times we processed this image
        mining_queue{i}.num_visited = mining_queue{i}.num_visited + 1;

        number_of_windows = number_of_windows + cellfun(@(x)length(x),scores)';

        clear curxs curbbs
        for q = 1:length(models)
            curxs{q} = [];
            curbbs{q} = [];
            if isempty(rs.xs{q})
                continue
            end

            goods = cellfun(@(x)numel(x),rs.xs{q})>0;
            curxs{q} = cat(2,curxs{q},rs.xs{q}{goods});
            curbbs{q} = cat(1,curbbs{q},rs.bbs{q}(goods,:));
        end

        Ndets = cellfun(@(x)size(x,2),curxs);
        % If no detections, just skip image because there is nothing to store
        if sum(Ndets) == 0
            empty_images(end+1) = index;
        end

        %an image is violating if it contains some violating windows,
        %else it is an empty image
        if max(cellfun(@(x)max_or_this(x,-1000),scores)) >= -1
            if mining_queue{i}.num_visited == 1
                number_of_violating_images = number_of_violating_images + 1;
            end 
            violating_images(end+1) = index;
        end

        for a = 1:length(models)
            xs{a}{i} = curxs{a};
            bbs{a}{i} = curbbs{a};
        end

        if (numpassed + models{1}.total_mines >= mining_params.train_max_mined_images) || (max(number_of_windows) >= mining_params.train_max_windows_per_iteration) || (numpassed >= mining_params.train_max_images_per_iteration)
            fprintf(1,['Stopping mining because we have %d windows from %d new violators\n'], max(number_of_windows), number_of_violating_images);
            break;
        end
    end

    if ~exist('xs','var')
        % If no detections from from any models, return an empty matrix
        for i = 1:length(models)
            hard_negatives.xs{i} = zeros(numel(models{i}.model.w),0);
            hard_negatives.bbs{i} = [];
        end
        mining_stats.num_violating = 0;
        mining_stats.num_empty = 0;
        return;
    end

    hard_negatives.xs = xs;
    hard_negatives.bbs = bbs;

    fprintf(1,'# Violating images: %d, #Non-violating images: %d\n', length(violating_images), length(empty_images));
    mining_stats.num_empty = length(empty_images);
    mining_stats.num_violating = length(violating_images);
    mining_stats.total_mines = mining_stats.num_violating + mining_stats.num_empty;

    % Update mining queue by removing already seen images
    mining_queue = update_mq_onepass(mining_queue, violating_images, empty_images);
    
end

function mining_queue = update_mq_onepass(mining_queue, violating_images, empty_images)
    % Takes the violating images and remove them from queue
    
    mover_ids = find(cellfun(@(x)ismember(x.index,violating_images), mining_queue));
    mining_queue(mover_ids) = [];

    % We now take the empty images and remove them from queue
    mover_ids = find(cellfun(@(x)ismember(x.index,empty_images), mining_queue));

    mining_queue(mover_ids) = [];
end