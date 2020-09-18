function r = cap_range(r, low, high)
    % Clips the value between high and low
    
    r = min(r, high);
    r = max(r, low);
    
end