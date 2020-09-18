function x = esvm_features(I, sbin)
    % Computes the HOG feature of a color image.
    %
    % Parameters
    % ----------
    % I - Color image of type double
    % sbin - Dimension of the feature
    %
    % Returns
    % -------
    % x - HOG features

    % Compute the HOG features
    x = features_pedro(I, sbin);

end
