function dataset_params = get_voc_dataset(VOCYEAR, data_dir, result_dir)
    % Get the dataset structure for a VOC dataset, given the VOCYEAR
    % string which is something like: VOC2007, VOC2010, etc.  This assumes
    % that VOC is locally installed, see Exemplar-SVM project page for
    % instructions if you need to do this.

    % Create a root directory
    dataset_params.devkitroot = [result_dir];

    % change this path to a writable local directory for the example code
    dataset_params.localdir = [dataset_params.devkitroot];

    % change this path to a writable directory for your results
    dataset_params.resdir = sprintf('%s%s', dataset_params.devkitroot, '/results');

    % This is location of the installed VOC datasets
    dataset_params.datadir = data_dir;
    
    % If enabled, shows outputs throughout the training/testing procedure
    dataset_params.display = 0;

    % Some VOC-specific dataset stats
    dataset_params.dataset = VOCYEAR;
    dataset_params.testset = 'test';

    %Do not skip evaluation
    dataset_params.SKIP_EVAL = 0;

    %Fill in the params structure with VOC-specific stuff
    dataset_params = VOCinit(dataset_params);
    
end
