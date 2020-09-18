function I = flip_image(I)
    % Flips the image in the LR direction
    %
    % Parameters
    % ----------
    % I - Colour image
    %
    % Returns
    % -------
    % I - LR flipped image

    for i = 1:size(I,3)
        I(:,:,i) = fliplr(I(:,:,i));
    end
end
