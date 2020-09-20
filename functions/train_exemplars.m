function [newmodels,new_models_name] = train_exemplars(models, train_set, params)
    % Train models with hard negatives mined from train_set.
    %
    % Parameters
    % ----------
    % models      : a cell array of initialized exemplar models
    % train_set   : a virtual set of images to mine from
    % params      : localization and training parameters

    % Copyright (C) 2011-12 by Tomasz Malisiewicz
    % All rights reserved.
    %
    % This file is part of the Exemplar-SVM library and is made
    % available under the terms of the MIT license (see COPYING file).
    % Project homepage: https://github.com/quantombone/exemplarsvm

    if isempty(models)
        newmodels = models;
        new_models_name = '';
        return;
    end

    models_name = models{1}.models_name;
    new_models_name = [models_name params.training_function()];

    dir_path =  sprintf('%s/models/', params.dataset_params.localdir);
    file_path = sprintf('%s/%s.mat', dir_path, new_models_name);
    stripped_file_path = sprintf('%s/%s-stripped.mat', dir_path, new_models_name);

    if file_exists(stripped_file_path)
        newmodels = load(stripped_file_path);
        newmodels = newmodels.models;
        return;
    end

    if file_exists(file_path)
        newmodels = load(file_path);
        newmodels = newmodels.models;
        return;
    end

    final_directory = sprintf('%s/models/%s/',params.dataset_params.localdir, new_models_name);

    %make results directory if needed
    if ~exist(final_directory,'dir')
        mkdir(final_directory);
    end

    % randomize chunk orderings
    myRandomize;
    ordering = randperm(length(models));

    models = models(ordering);
    allfiles = cell(length(models), 1);
    
    for i = 1:length(models)
        m = models{i};
        complete_file = sprintf('%s/%s.mat',final_directory,m.name);
        [basedir, basename, ~] = fileparts(complete_file);
        filer2fill = sprintf('%s/%%s.%s.mat', basedir, basename);
        filer2final = sprintf('%s/%s.mat', basedir, basename);
        allfiles{i} = filer2final;
        
        if file_exists(filer2final)
            continue
        end
        
        % Add training set and training set's mining queue
        m.train_set = train_set;
        m.mining_queue = initialize_mining_queue(m.train_set);

        % Add mining_params, and params.dataset_params to this exemplar
        m.mining_params = params;
        m.dataset_params = params.dataset_params;

        % Append '-svm' to the mode to create the models name
        m.models_name = new_models_name;
        m.iteration = 1;

        % If we are a distance function, initialize to uniform weights
        if isfield(params, 'wtype') && strcmp(params.wtype,'dfun')==1
            m.model.w = m.model.w*0-1;
            m.model.b = -1000;
        end

        % The mining queue is the ordering in which we process new images
        keep_going = 1;
        while keep_going == 1
            %Get the name of the next chunk file to write
            filer2 = sprintf(filer2fill,num2str(m.iteration));
            if ~isfield(m,'mining_stats')
                total_mines = 0;
            else
                total_mines = sum(cellfun(@(x)x.total_mines,m.mining_stats));
            end
            m.total_mines = total_mines;
            m = mine_train_iteration(m, params.training_function);

            if total_mines >= params.train_max_mined_images || isempty(m.mining_queue) || m.iteration == params.train_max_mine_iterations
                keep_going = 0;
                %bump up filename to final file
                filer2 = filer2final;
            end

            msave = m;
            m = rmfield(m,'train_set');

            % Save the current result
            save(filer2, 'm');
            
            m = msave;

            %delete old files
            if m.iteration > 1
                for q = 1:m.iteration-1
                    filer2old = sprintf(filer2fill,num2str(q));
                    if file_exists(filer2old)
                        delete(filer2old);
                    end
                end
            end

            if keep_going == 0
                fprintf(1,' ### End of training... \n');
                break;
            end

            m.iteration = m.iteration + 1;
        end
    end

    [allfiles] = sort(allfiles);

    % Load all of the initialized exemplars
    CACHE_FILE = 1;
    STRIP_FILE = 1;

    if new_models_name(1) == '-'
        CACHE_FILE = 0;
        STRIP_FILE = 0;
    end

    newmodels = load_models(params.dataset_params, new_models_name, allfiles, CACHE_FILE, STRIP_FILE);
end

function mining_queue = initialize_mining_queue(imageset, ordering)
    % Initializes mining queue with the given ordering (if given) or 
    % randomly.
    
    if ~exist('ordering','var')
        fprintf(1,'Randomizing mining queue\n');
        myRandomize;
        ordering = randperm(length(imageset));
    end

    mining_queue = cell(0,1);
    for i = 1:length(ordering)
        mining_queue{i}.index = ordering(i);
        mining_queue{i}.num_visited = 0;
    end
end
