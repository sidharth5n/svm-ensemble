function [models] = load_models(dataset_params, models_name, files, ...
                                    CACHE_FILE, STRIP_FILE)
    % Loads all the trained models of a specified class and type from the
    % models directory.
    %
    % Copyright (C) 2011-12 by Tomasz Malisiewicz
    % All rights reserved.
    %
    % This file is part of the Exemplar-SVM library and is made
    % available under the terms of the MIT license (see COPYING file).

    if ~exist('CACHE_FILE','var')
        CACHE_FILE = 0;
    end

    if ~exist('STRIP_FILE','var')
        STRIP_FILE = 0;
    end

    if CACHE_FILE == 1
        cache_dir = sprintf('%s/models/',dataset_params.localdir);
        if ~exist(cache_dir,'dir')
            mkdir(cache_dir);
        end

        cache_file = sprintf('%s/%s.mat',cache_dir,models_name);
        cache_file_stripped = sprintf('%s/%s-stripped.mat',cache_dir,models_name);

        if fileexists(cache_file)
            % Strip_file can only be present if the cache_file is present
            if STRIP_FILE == 1
                cache_file = cache_file_stripped;
            end
            fprintf(1,'Loading CACHED file: %s\n', cache_file);
            load(cache_file);
            return;
        end
    end

    if ~exist('files', 'var') || isempty(files)
        results_directory = sprintf('%s/models/%s/', dataset_params.localdir, models_name);
        dirstring = [results_directory '*.mat'];
        files = dir(dirstring);
        fprintf(1, 'Pattern of files to load: %s\n',dirstring);
        fprintf(1, 'Length of files to load: %d\n',length(files));
        newfiles = cell(length(files), 1);
        for i = 1:length(files)
            newfiles{i} = [results_directory files(i).name];
        end
        files = newfiles;
    else
        fprintf(1, 'Load from a total of %d files:\n',length(files));
    end

    models = cell(1,length(files));
    for i = 1:length(files)
        fprintf(1,'.');
        m = load(files{i});
        models{i} = m.m;
    end

    if isempty(files)
        fprintf(1,'WARNING: no models loaded for %s\n', models_name);
        models = cell(0,1);
        return;
    end

    fprintf(1,'\n');

    if CACHE_FILE == 1
        if file_exists(cache_file)
            return;
        end

        fprintf(1,'Loaded models, saving to %s\n', cache_file);
        save(cache_file, 'models');

        if STRIP_FILE == 1
            stripped_models = strip_models(models);
            fprintf(1,'Saving stripped to %s\n', cache_file_stripped);
            save(cache_file_stripped,'stripped_models');
        end
    end
end

function models = strip_models(models)
    % Strips the models of residual training data and keeps only the
    % information relevant for detection. Helps in faster detection.
    
    if ~iscell(models)
        models = strip_models({models});
        models = models{1};
        return;
    end

    for i = 1:length(models)
        cur = models{i};
        clear m;
        m.model.init_params = cur.model.init_params;
        m.model.hg_size = cur.model.hg_size;
        m.model.mask = cur.model.mask;
        m.model.w = cur.model.w;
        m.model.x = cur.model.x(:,1);
        m.model.b = cur.model.b;
        m.model.bb = cur.model.bb;
        if isfield(cur,'I')
            m.I = cur.I;
        end
        if isfield(cur,'curid')
            m.curid = cur.curid;
            m.objectid = cur.objectid;
        end
        if isfield(cur,'cls')
            m.cls = cur.cls;
        end
        m.gt_box = cur.gt_box;
        m.sizeI = cur.sizeI;
        m.models_name = cur.models_name;
        models{i} = m;
    end
end