function I = load_image(path)
    % Loads the image in the given path and converts it into double.
    % Parameters
    % ----------
    % path : Path to an image file
    %
    % Returns
    % -------
    % I : M x N x 3 image matrix

    I = imread(path);
    I = im2double(I);
end
