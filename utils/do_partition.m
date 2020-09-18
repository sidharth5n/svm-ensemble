function partition_inds = do_partition(sequence, chunk_size)
    % Partitions the sequence into blocks of chunk size.
    %
    % Parameters
    % ----------
    % sequence : The sequence to be partitioned
    % chunk_size : Size of each chunk or block
    %
    % Returns
    % -------
    % partition_inds : Sequence paritioned into blocks.
    
    len = length(sequence);
    starts = 1:chunk_size:(len+chunk_size);
    partition_inds{1} = [];
    for i = 1:length(starts)-1
        partition_inds{i} = sequence(starts(i):min(len,starts(i+1)-1));
    end
end

