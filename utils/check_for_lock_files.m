function lockfiles = check_for_lock_files(target_directory)
    % Checks whether there are any lock files (directories with .lock
    % ending) inside the target directory.
    % 
    % Parameters
    % ----------
    % target_directory - Path to the target directory
    %
    % Returns
    % -------
    % lockfiles - Cell array of lock files.

    [a, b] = system(sprintf('find %s -name "*.lock"',target_directory));
    if a == 1 || isempty(b)
        lockfiles = {};
    else
        b = textscan(b,'%s');
        lockfiles = b{1};
    end
end