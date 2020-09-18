function [res,ind] = max_or_this(x,value)
    % Returns the maximum value and index in x if x is not empty, else
    % returns value as the maximum value and index as -1.
    
    if ~isempty(x)
        [res,ind] = max(x);
        return;
    end
    
    res = value;
    ind = -1;
    
end
