function im = HOGpicture(w, bs)
    % Creates a picture of positive HOG weights by scanning over 18
    % orientations when creating the histogram.
    
    if ~exist('bs','var')
        bs = 20;
    end
    
    bim1 = zeros(bs, bs);
    bim1(:,round(bs/2):round(bs/2)+1) = 1;
    bim = zeros([size(bim1) 9]);
    bim(:,:,1) = bim1;
    for i = 2:9
        bim(:,:,i) = imrotate(bim1, -(i-1)*20, 'crop');
    end

    bim = cat(3,bim,bim);

    % make pictures of positive weights bs adding up weighted glyphs
    s = size(w);    
    w(w < 0) = 0;    
    im = zeros(bs*s(1), bs*s(2));
    for i = 1:s(1)
        iis = (i-1)*bs+1:i*bs;
        for j = 1:s(2)
            jjs = (j-1)*bs+1:j*bs;          
            for k = 1:18
                im(iis,jjs) = im(iis,jjs) + bim(:,:,k) * w(i,j,k);
            end
        end
    end
end