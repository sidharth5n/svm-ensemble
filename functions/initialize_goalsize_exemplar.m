function model = initialize_goalsize_exemplar(I, bbox, init_params)
    % Initialize the exemplar (or scene) such that the representation
    % which tries to choose a region which overlaps best with the given
    % bbox and contains roughly init_params.goal_ncells cells, with a
    % maximum dimension of init_params.MAXDIM
    %
    % Copyright (C) 2011-12 by Tomasz Malisiewicz
    % All rights reserved.
    %
    % This file is part of the Exemplar-SVM library and is made
    % available under the terms of the MIT license (see COPYING file).
    % Project homepage: https://github.com/quantombone/exemplarsvm

    if ~exist('init_params','var')
        init_params.sbin = 8;
        init_params.hg_size = [8 8];
        init_params.MAXDIM = 10;
    end

    if ~isfield(init_params,'MAXDIM')
        init_params.MAXDIM = 10;
        fprintf(1,'Default MAXDIM is %d\n',init_params.MAXDIM);
    end

    % Expand the bbox to main aspect ratio constraints
    bbox = expand_bbox(bbox, I);

    % Create a blank image with the exemplar inside
    Ibox = zeros(size(I, 1), size(I, 2));
    Ibox(bbox(2):bbox(4), bbox(1):bbox(3)) = 1;

    % Get the hog feature pyramid for the entire image
    clear params;
    params.detect_levels_per_octave = 10;
    params.init_params = init_params;
    [f_real, ~] = pyramid(I, params);

    % Extract the regions most overlapping with Ibox from each level in the pyramid
    [masker, sizer] = get_matching_masks(f_real, Ibox);

    % Now choose the mask which is closest to N cells
    [targetlvl, mask] = get_ncell_mask(init_params, masker, sizer);
    [uu,vv] = find(mask);
    curfeats = f_real{targetlvl}(min(uu):max(uu),min(vv):max(vv),:);

    model.init_params = init_params;
    model.hg_size = size(curfeats);
    model.mask = logical(ones(model.hg_size(1), model.hg_size(2)));

    fprintf(1,'initialized with HOG_size = [%d %d]\n',model.hg_size(1),model.hg_size(2));
    model.w = curfeats - mean(curfeats(:));
    model.b = 0;
    model.x = curfeats;

    % Fire inside self-image to get detection location
    [model.bb, model.x] = get_target_bb(model, I, init_params);

    % Normalized-HOG initialization
    model.w = reshape(model.x,size(model.w)) - mean(model.x(:));

    if isfield(init_params,'wiggle_number') && (init_params.wiggle_number > 1)
        model = get_model_wiggles(I, model, init_params.wiggle_number);
    end
end


function [targetlvl, mask] = get_ncell_mask(init_params, mask, size)
    % Get the mask and features, where mask is closest to NCELL cells
    % as possible
    
    for i = 1:size(mask)
        [u, v] = find(mask{i});
        if max(u) - min(u) + 1 <= init_params.MAXDIM && max(v) - min(v) + 1 <= init_params.MAXDIM
            targetlvl = i;
            mask = mask{targetlvl};
            return;
        end
    end
    
    ncells = prod(size, 2);
    [~, targetlvl] = min(abs(ncells - init_params.goal_ncells));
    mask = mask{targetlvl};
end

function [masker,sizer] = get_matching_masks(f_real, Ibox)
    % Finds the best matching region per level in the feature pyramid given 
    % a feature pyramid, and a segmentation mask inside Ibox.

    masker = cell(length(f_real),1);
    sizer = zeros(length(f_real),2);

    for i = 1:length(f_real)
        goods = double(sum(f_real{i}.^2,3)>0);
        masker{i} = max(0.0,min(1.0,imresize(Ibox,[size(f_real{i},1) size(f_real{i}, 2)])));
        [~, ind] = max(masker{i}(:));
        masker{i} = (masker{i} > 0.1) & goods;
        if sum(masker{i}(:)) == 0
            [a, b] = ind2sub(size(masker{i}), ind);
            masker{i}(a, b) = 1;
        end
        [a, b] = find(masker{a});
        masker{i}(min(a):max(a),min(b):max(b))=1;
        sizer(i,:) = [range(a)+1 range(b)+1];
    end
end

function bbox = expand_bbox(bbox,I)
    % Expand bounding box and satisfy the following constraints.
    % 1. Each dimension is at least 50 pixels
    % 2. Max aspect ratio is (.25, 4)
    % 3. Is inside the image
    
    for expandloop = 1:10000
      % Get initial dimensions
        w = bbox(3)-bbox(1)+1;
        h = bbox(4)-bbox(2)+1;
        if h > w*4 || w < 50
            % make wider
            bbox(3) = bbox(3) + 1;
            bbox(1) = bbox(1) - 1;
        elseif w > h*4 || h < 50
            % make taller
            bbox(4) = bbox(4) + 1;
            bbox(2) = bbox(2) - 1;
        else
            break;
        end
        bbox([1 3]) = cap_range(bbox([1 3]), 1, size(I,2));
        bbox([2 4]) = cap_range(bbox([2 4]), 1, size(I,1));
    end
end


function [target_bb,target_x] = get_target_bb(model, I, init_params)
    %Get the bounding box of the top detection

    m{1}.model = model;
    m{1}.model.hg_size = size(model.w);
    localizeparams.detect_keep_threshold = -100000.0;
    localizeparams.detect_max_windows_per_exemplar = 1;
    localizeparams.detect_levels_per_octave = 10;
    localizeparams.detect_save_features = 1;
    localizeparams.detect_add_flip = 0;
    localizeparams.detect_pyramid_padding = 5;
    localizeparams.dfun = 0;
    localizeparams.init_params = init_params;
    [rs,~] = detect(I,m,localizeparams);
    target_bb = rs.bbs{1}(1,:);
    target_x = rs.xs{1}{1};
end
