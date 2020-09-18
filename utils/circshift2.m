function I2 = circshift2(I, K)
    % Shifts the contents of the image by K(1) rows down and K(2) columns left
    % and replaces the shifted rows and columns with zeros.
    %
    % Parameters
    % ----------
    % I : Image to be shifted
    % K : [K(1), K(2)]
    %
    % Returns
    % -------
    % I2 : The shifted image

    % Circularly shift I by K(1) rows down and K(2) columns left
    I2 = circshift(I, K);

    % If K(1) is positive replace rows 1 to min of rows of I and K(1) with zero
    if K(1) > 0
        I2(1:min(size(I,1),K(1)),:) = 0;
    % If K(1) is negative replace rows 1 + max of rows of I and K(1) to rows of I with zero
    elseif K(1) < 0
        laster = size(I2,1);
        I2(1+max(1,laster+K(1)):laster,:) = 0;
    else  
    end
    
    % If K(2) is positive replace columns 1 to min of columns of I and K(1) with zero
    if K(2) > 0
        I2(:,1:min(size(I,2),K(2)),:)=0;
    % If K(2) is negative replace columns 1 + max of columns of I and K(1) to columns of I with zero
    elseif K(2) < 0
        laster = size(I2,2);
        I2(:,1+max(1,laster+K(2)):(laster),:) = 0;
    else
    end
end


