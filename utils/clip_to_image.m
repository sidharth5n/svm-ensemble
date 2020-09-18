function boxes = clip_to_image(boxes,img_size)
    % Clips bounding boxes to image dimensions.
    %
    % Parameters
    % ----------
    % boxes : Bounding boxes
    % img_size : Image dimensions
    %
    % Returns
    % -------
    % boxes : Clipped bounding boxes
    
    if size(boxes,1) == 0
        return;
    end

    for i = 1:2
        boxes(:,i) = max(img_size(i),boxes(:,i));
    end

    for i = 3:4
        boxes(:,i) = min(img_size(i),boxes(:,i));
    end
end
