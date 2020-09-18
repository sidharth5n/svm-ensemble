function transform = find_transform(c, d)
    % Finds the 3 x 3 transformation matrix that maps c to d whcih is a simple
    % translation and scaling. (Applied to homogeneous coordinates)

    % convert bounding box to corners
    xs(:,1) = c([1 2])';
    xs(:,2) = c([3 2])';
    xs(:,3) = c([1 4])';
    xs(:,4) = c([3 4])';
    
    % convert bounding box to corners
    ys(:,1) = d([1 2])';
    ys(:,2) = d([3 2])';
    ys(:,3) = d([1 4])';
    ys(:,4) = d([3 4])';

    xs(3,:) = 1;
    ys(3,:) = 1;

    A = ys * pinv(xs);
    A(abs(A)<.000001) = 0;

    transform = A;
end
