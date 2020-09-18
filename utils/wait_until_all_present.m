function wait_until_all_present(files, PAUSE_TIME, invert)
    % Waits until all the files are present (or absent), checks after every
    % PAUSE_TIME seconds.
    %
    % Parameters
    % files      : List of file names
    % PAUSE_TIME : Interval between two checks
    % invert     : If True check for absence of files, if False check for
    %              presence
    
    if isempty(files)
        return;
    end

    if ~exist('PAUSE_TIME','var')
        PAUSE_TIME = 5;
    end

    if ~exist('invert','var')
        TARGET = 0;
    elseif invert == 1
        TARGET = length(files);
    else
        error('invert must be absent or 1');
    end

    while 1
        missingfile = cellfun(@(x)~fileexists(x),files);
        if sum(missingfile) == TARGET
            break;
        else
            missings = find(missingfile);
            fprintf(1,['%03d File(s) missing [should be %d]', 'waiting %d sec until re-try\n'], sum(missingfile),TARGET, PAUSE_TIME);
            for q = 1:length(missings)
                fprintf(1,' --missing %s\n',files{missings(q)});
            end
            pause(PAUSE_TIME);
        end
    end
end