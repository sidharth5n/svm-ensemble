function bool = mymkdir(path)
    % Creates a directory with the given path if it does not exist.
    %
    % Parameters
    % ----------
    % path : Path to a directory
    %
    % Returns
    % bool : 1 if directory did not exist and 0 if it already exists
    
    [~, ~, smessid] = mkdir(path);
    bool = ~strcmp(smessid,'MATLAB:MKDIR:DirectoryExists');
end