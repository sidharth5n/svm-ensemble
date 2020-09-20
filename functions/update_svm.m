function m = update_svm(m)
    % Perform SVM learning for a single exemplar model, we assume that
    % the exemplar has a set of detections loaded in m.model.svxs and m.model.svbbs
    % Durning Learning, we can apply some pre-processing such as PCA or
    % dominant gradient projection
    %
    % Copyright (C) 2011-12 by Tomasz Malisiewicz
    % All rights reserved.
    %
    % This file is part of the Exemplar-SVM library and is made
    % available under the terms of the MIT license (see COPYING file).
    % Project homepage: https://github.com/quantombone/exemplarsvm

    %if no inputs are specified, just return the suffix of current method
    if nargin == 0
        m = '-svm';
        return;
    end

    if ~isfield(m.model, 'mask') || isempty(m.model.mask)
        m.model.mask = true(numel(m.model.w),1);
    end

    if length(m.model.mask(:)) ~= numel(m.model.w)
        m.model.mask = repmat(m.model.mask,[1 1 m.model.hg_size(3)]);
        m.model.mask = logical(m.model.mask(:));
    end

    mining_params = m.mining_params;
    xs = m.model.svxs;
    bbs = m.model.svbbs;

    max_size = 3500;
    if size(xs,2) >= max_size
        half_size = max_size/2;
        %NOTE: random is better than top 5000
        r = m.model.w(:)'*xs;
        [~,r] = sort(r, 'descend');
        r1 = r(1:half_size);

        r = half_size + randperm(length(r((half_size+1):end)));
        r = r(1:half_size);
        r = [r1 r];
        xs = xs(:,r);
    end

    super_x = cat(2,m.model.x,xs);
    super_y = cat(1,ones(size(m.model.x,2),1),-1*ones(size(xs,2),1));

    num_pos = sum(super_y == 1);
    num_neg = sum(super_y == -1);

    wpos = mining_params.train_positives_constant;

    A = eye(size(super_x,1));
    mu = zeros(size(super_x,1),1);

    newx = bsxfun(@minus,super_x,mu);
    newx = newx(logical(m.model.mask),:);
    newx = A(m.model.mask,:)'*newx;

    fprintf(1,' -----\nStarting SVM: dim=%d... #pos=%d, #neg=%d ', size(newx,1),num_pos,num_neg);
    start_time = tic;

    svm_model = libsvmtrain(super_y, newx', sprintf(['-s 0 -t 0 -c %f -w1 %.9f -q'], mining_params.train_svm_c, wpos));

    if isempty(svm_model.sv_coef)
        % Learning had no negatives
        wex = m.model.w;
        b = m.model.b;
        fprintf(1,'reverting to old model...\n');
    else
        % convert support vectors to decision boundary
        svm_weights = full(sum(svm_model.SVs .* repmat(svm_model.sv_coef,1, size(svm_model.SVs,2)),1));
        wex = svm_weights';
        b = svm_model.rho;
        if super_y(1) == -1
            wex = wex*-1;
            b = b*-1;
        end

        % project back to original space
        b = b + wex'*A(m.model.mask,:)'*mu(m.model.mask);
        wex = A(m.model.mask,:)*wex;

        wex2 = zeros(size(super_x,1),1);
        wex2(m.model.mask) = wex;

        wex = wex2;

        % issue a warning if the norm is very small
        if norm(wex) < .00001
            fprintf(1,'learning broke down!\n');
        end
    end

    maxpos = max(wex'*m.model.x - b);
    fprintf(1, ' --- Max positive is %.3f\n', maxpos);
    fprintf(1, 'SVM iteration took %.3f sec, ', toc(start_time));

    m.model.w = reshape(wex, size(m.model.w));
    m.model.b = b;

    r = m.model.w(:)'*m.model.svxs - m.model.b;
    svs = find(r >= -1.0000);

    if isempty(svs)
        fprintf(1,' ERROR: number of negative support vectors is 0!\');
        error('Something went wrong');
    end

    %KEEP (nsv_multiplier * #SV) vectors, but at most max_negatives of them
    total_length = ceil(mining_params.train_keep_nsv_multiplier*length(svs));
    total_length = min(total_length,mining_params.train_max_negatives_in_cache);

    [~, beta] = sort(r,'descend');
    svs = beta(1:min(length(beta),total_length));
    m.model.svxs = m.model.svxs(:,svs);
    m.model.svbbs = m.model.svbbs(svs,:);
    fprintf(1,' kept %d negatives\n',total_length);
end
