function transformed_c = apply_transform(c, transform)
    % Applies the given transform on the bounding box and generates a new
    % bounding box with all non coordinate fields left intact
    %
    % Parameters
    % ----------
    % c             : N x M set of bounding box coordinates and other 
    %                 properties (if any) where M is 2, 4 or any number 
    %                 greater than 4
    % xform         : 3 x 3 transformation matrix
    %
    % Returns
    % -------
    % transformed_c : N x M set of transformed bounding box coordinates and
    %                 other properties (if any) same as in c

    transformed_c = c;

    if size(c,1) == 0
        return;
    end

    if size(c,2) == 1 || size(c,2) == 3
        error('apply_xform: invalid size of input');
    end

    % Since xform is 3 x 3, each coordinate is padded with z = 1 and the third
    % coordinate is ignored after the transformation

    % APPLYING THE TRANSFORMATION ON (x1, y1)
    xs = c(:,1:2)';
    xs(3,:) = 1;
    d = transform * xs;
    transformed_c(:, 1:2) = d(1:2,:)';

    % APPLYING THE TRANSFORMATION ON (x2, y2) IF ANY
    if size(c, 2) >= 4
        xs = c(:,3:4)';
        xs(3,:) = 1;
        d = transform * xs;
        transformed_c(:, 3:4) = d(1:2,:)';
    end
end
