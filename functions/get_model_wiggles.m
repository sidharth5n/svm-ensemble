function model = get_model_wiggles(I, model, num_wiggles)
    % Get wiggles of a model's bounding box and populate model with
    % them.  Wiggles are also known as "perturbations."
    %
    % Copyright (C) 2011-12 by Tomasz Malisiewicz
    % All rights reserved.
    % 
    % This file is part of the Exemplar-SVM library and is made
    % available under the terms of the MIT license (see COPYING file).
    % Project homepage: https://github.com/quantombone/exemplarsvm

    x = replica_hits(I, model.init_params.sbin, model.bb(1,:), model.hg_size, num_wiggles);

    % GET self feature + NWIGGLES wiggles "perturbed images"
    model.x = x;
    model.w = reshape(mean(model.x,2), model.hg_size);
    model.w = model.w - mean(model.w(:));
    model.b = 0;
end


function x = replica_hits(I, sbin, bb, hg_size, num_wiggles)
    % Given an image, an sbin, and the location of the detection window
    % encoded as bb, create N_WIGGLES wiggles by perturbing image content
    % inside the frame
    
    if ~exist('N_WIGGLES','var')
        num_wiggles = 100;
    end

    I = im2double(I);

    pad = 5;

    u = bb(9);
    v = bb(10);

    if bb(7) == 1
        I = flip_image(I);
    end

    lilI = resize(I,bb(8));
    fprintf(1,'Warning: using features directly\n');
    f = features(lilI, sbin);
    f = padarray(f,[pad pad 0]);
    x_base = f(u-2+(1:hg_size(1)),v-2+(1:hg_size(2)),:);

    % NOW DO WIGGLES WITHOUT A SLIDE!
    x = zeros(prod(hg_size),num_wiggles);
    x(:,1) = x_base(:);
    for i = 2:num_wiggles
        % Add random noise
        curI = lilI + .02*randn(size(lilI));

        %Perform random gaussian smoothing
        randsigma = .1+rand;
        curI = (imfilter(curI,fspecial('gaussian',[9 9],randsigma)));

        %Perform random shift
        cx = floor(3*rand)-1;
        cy = floor(3*rand)-1;
        curI = circshift2(curI,[cx cy]);

        f = features(curI, sbin);
        f = padarray(f,[pad pad 0]);
        x = f(u-2+(1:hg_size(1)),v-2+(1:hg_size(2)),:);
        x(:,i) = x(:);
    end
end