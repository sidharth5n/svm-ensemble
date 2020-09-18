function myRandomize

    if exist('RandStream', 'builtin')
        RandStream.setDefaultStream(RandStream('mt19937ar','seed',sum(100*clock)))
    else
        try
            rng('default');
        catch
            rng('shuffle');
        end
    end
end
