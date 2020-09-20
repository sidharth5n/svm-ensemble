function I = get_exemplar_icon(model, flip, mode)
    % Extract an exemplar visualization image from cb box with the default 
    % window (mode = 1) or from GT box.
    %
    % Parameters
    % ----------
    % model : cell
    % flip  : Whether to flip (optional)
    % mode  : If True, exemplar visualization image from cb box, otherwise
    %         from GT box.
    %
    % Returns
    % -------
    % I     : Exemplar visualization image
    
    if ~isfield(model.model,'bb') || numel(model.model.bb) == 0
        I = ones(10, 10, 3);
        return;
    end
    
    if isfield(model, 'I')
        I = load_image(model.I);
    else
        I = load_image(model.train_set(model.model.bb(1, 11)));
    end
    
    if exist('mode', 'var') && mode == 1
        % Coordinates from cb box
        coordinates = round(model.model.bb(1, 1:4));
    else
        % Coordinates from GT box
        coordinates = round(model.gt_box);
    end
    
    I = I(coordinates(2):coordinates(4), coordinates(1):coordinates(3), :);
    if exist('flip', 'var') && flip == 1
        I = flip_image(I);
    end 
end