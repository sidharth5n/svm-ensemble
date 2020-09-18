function res = fileexists(filename)
    % Checks whether the given file exists. Faster than exist method.
    % Parameters
    % ----------
    % filename : Path to the file
    % 
    % Returns
    % -------
    % res : 0 if file does not exist, 1 if file exists
    
    fid = fopen(filename,'r');
    if fid == -1
        res = 0;
    else
        fclose(fid);
        res = 1;
    end