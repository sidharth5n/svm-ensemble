function image_directory = get_pascal_set(VOCopts, type, class, belonging)
    % Finds the absolute path w.r.t master directory of the images given 
    % the target directory, class and belongingness.
    %
    % Parameters
    % ----------
    % VOCopts   : Dataset parameters
    % type      : One of 'train', 'trainval', 'test' or 'both
    % cls       : Object class
    % belonging : 1 if belonging to the given class and -1 otherwise
    %
    % Returns
    % -------
    % image_directory : cell array of absolute simage paths
    
    if strcmp(type, 'both') && ~exist('classs', 'var')
        image_directory = cat(1, get_pascal_set(VOCopts, 'trainval'), get_pascal_set(VOCopts, 'test'));
        return;
    end
    
    if ~exist('class', 'var')
        file = sprintf(VOCopts.imgsetpath, type);
    else
        file = sprintf(VOCopts.clsimgsetpath, class, type);
    end
    
    try
        [neg_set, gt] = textread(file, '%s %d');
    catch
        fprintf(1,'Cannot load file %s\n',file);
        error('Cannot load file');
    end
    
    if exist('belonging', 'var')
        neg_set = neg_set(gt == belonging);
    end
    
    % Generate full path of the image
    image_directory = cellfun2(@(x)sprintf(VOCopts.imgpath, x),neg_set);
end
