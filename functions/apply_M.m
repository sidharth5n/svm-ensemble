function r = apply_M(x, boxes, M)
    % Applies boosting "co-occurrence" matrix M to the boxes
    %
    % Copyright (C) 2011-12 by Tomasz Malisiewicz
    % All rights reserved.
    % 
    % This file is part of the Exemplar-SVM library and is made
    % available under the terms of the MIT license (see COPYING file).
    % Project homepage: https://github.com/quantombone/exemplarsvm

    if numel(x) == 0
        r = zeros(1,0);
        return;
    end
    
    exemplar_id = boxes(:,6);

    % Get new exemplar id for exemplars with flips
    exemplar_id(boxes(:,7) == 1) = exemplar_id(boxes(:,7)==1) + size(x,1)/2;
    r = zeros(1,size(boxes,1));

    for i = 1:size(boxes,1)
        r(i) = M.w{exemplar_id(i)}'*x(:,i) + sum(x(:,i)) - M.b{exemplar_id(i)};
    end
end