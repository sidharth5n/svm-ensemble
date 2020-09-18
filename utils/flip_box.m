function bbox2 = flip_box(bbox, sizeI)
    % Flip a matrix of boxes using a L-R reflection
    % Each row is a new BB with the first four columns as the BB location

    % x2 - x1 + 1
    W = bbox(:,3) - bbox(:,1) + 1;

    bbox2 = bbox;
    % x2 = image width - x1
    bbox2(:,3) = sizeI(2)-bbox2(:,1);
    % x1 = x2 - width + 1
    bbox2(:,1) = bbox2(:,3) - W + 1;
end
