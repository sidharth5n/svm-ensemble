function [x, nbrids] = get_M_features(boxes, N, neighbor_thresh)
    % Get the contextual "box features" for a set of detection boxes
    % Turns a detection box into a contextual feature by stacking the
    % scores of nearby (highly overlapping) detections.
    %
    % Parameters
    % ----------
    % boxes: K detections from a single image "[K x 12] matrix"
    % N: raw number of exemplars (unflipped) "scalar"
    % neighbor_thresh: the OS threshold which two boxes must meet
    %   to be considered neighbors "scalar"
    %
    % Returns
    % -------
    % x: a [2*N x K]
    % nbrids: [1 x K] cell array indicating the raw box ids belonging
    %   to each context feature
    %
    % Copyright (C) 2011-12 by Tomasz Malisiewicz
    % All rights reserved.
    % 
    % This file is part of the Exemplar-SVM library and is made
    % available under the terms of the MIT license (see COPYING file).
    % Project homepage: https://github.com/quantombone/exemplarsvm
    
    K = size(boxes,1);
    x = sparse(N*2, K);
    nbrids = cell(1,K);

    if K == 0
        return;
    end

    %Get overlaps between all boxes in the set
    osmat = iou(boxes, boxes);
    
    % Get new exemplar id for flipped boxes
    exid = boxes(:,6)';
    isflip = boxes(:,7) == 1;
    exid(isflip) = exid(isflip) + N;

    % scores already calibrated
    box_scores = boxes(:,end);

    for j = 1:K
        friends = (osmat(:,j) >= neighbor_thresh);
        friend_scores = box_scores .* friends;
        nbrids{j} = find(friends);
        if sum(friends) == 0
            continue
        end
        x(exid(friends),j) = friend_scores(friends);
    end
end
