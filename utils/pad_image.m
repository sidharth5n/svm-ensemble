function I2 = pad_image(I, pad_size, pad_value)
    % Pads an image by the given size with given values.
    %
    % Parameters
    % ----------
    % I         : Image to be padded
    % pad_size  : Size of padding
    % pad_value : The value with which to be padded. Can be different for
    %             each channel.
    %
    % Returns
    % -------
    % I2        : Padded image
    

    % DEFAULT VALUE OF PADDING IS 0
    if ~exist('pad_value','var')
        pad_value = 0;
    end

    % PAD VALUE IS DIFFERENT FOR EACH CHANNEL
    if length(pad_value) == 3
        I2 = ones(size(I,1) + pad_size*2, size(I,2) + pad_size*2, size(I,3));
        for q = 1:3
            I2(:,:,q) = pad_value(q);
        end
    % PAD VALUE IS SAME FOR ALL THE CHANNELS
    elseif length(pad_value) == 1
        I2 = ones(size(I,1) + pad_size*2, size(I,2) + pad_size*2, size(I,3))*pad_value;
    end

    % POSITIVE PADDING
    if pad_size > 0
        I2(pad_size+(1:size(I,1)), pad_size+(1:size(I,2)), :) = I;
    % NEGATIVE PADDING
    else
        I2 = I(-pad_size+1:(size(I,1)+pad_size),-pad_size+1:(size(I,2)+pad_size),:);
    end
end
