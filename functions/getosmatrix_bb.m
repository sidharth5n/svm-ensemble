function os = getosmatrix_bb(boxes, gts)
    % Given N1 boxes and N2 ground truth boxes, computes IoU (Intersection
    % over Union) score of each box wrt each ground truth box resulting in
    % an N1 x N2 score matrix.
    %
    % Parameters
    % ----------
    % boxes : N1 x 4 matrix of box coordinates
    % gts   : N2 x 4 matrix of ground truth coordinates
    % Returns
    % -------
    % os    : N1 x N2 matrix of IoU scores

    if ~exist('gts','var')
        gts = boxes;
    end

    % IF ANY OF BOXES, GT BOXES IS EMPTY RETURN EMPTY MATRIX/VECTOR
    if numel(boxes) == 0 || numel(gts) == 0
        os = zeros(size(boxes,1),size(gts,1));
        return;
    end

    % COORDINATES AND AREA OF BOXES
    x1 = boxes(:,1);
    y1 = boxes(:,2);
    x2 = boxes(:,3);
    y2 = boxes(:,4);
    area = (x2 - x1 + 1) .* (y2 - y1 + 1);

    % COORDINATES AND AREA OF GT BOXES
    xa1 = gts(:,1);
    ya1 = gts(:,2);
    xa2 = gts(:,3);
    ya2 = gts(:,4);
    area2 = (xa2 - xa1 + 1) .* (ya2 - ya1 + 1);

    os = zeros(size(boxes,1),size(gts,1));

    for i = 1:size(boxes,1)
        % COORDINATES OF INTERSECTION BOX
        xx1 = max(boxes(i,1),gts(:,1));
        yy1 = max(boxes(i,2),gts(:,2));
        xx2 = min(boxes(i,3),gts(:,3));
        yy2 = min(boxes(i,4),gts(:,4));

        % WIDTH AND HEIGHT
        w = xx2 - xx1 + 1;
        h = yy2 - yy1 + 1;

        % INTERSECTION AREA WRT ALL GT BOXES
        o = w.*h;
        % SET AREA TO 0 IF ANY OF W, H IS NEGATIVE
        o( (w < 0) | (h < 0) ) = 0;

        % IoU
        os(i,:) = o ./ (eps + (area(i) + area2 - o));
    end
end
