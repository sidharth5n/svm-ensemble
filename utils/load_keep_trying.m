function res = load_keep_trying(path, pause_interval)
    % Loads the given file and waits for given interval before retrying if
    % file does not exist.
    %
    % Parameters
    % ----------
    % path : Path to the file be loaded
    % pause_interval : Gap between two tries
    %
    % Returns
    % -------
    % res : Contents of the file
    
    if ~exist('pause_interval','var')
        pause_interval = 5.0;
    end

    while 1
        try
            res = load(path);
            break;
        catch
            fprintf(1,'cannot load %s\n ---sleeping for %.3fsec, trying again\n', path,pause_interval);
            pause(pause_interval);
        end
    end
end
