function models = initialize_exemplars(e_set, params, models_name)
    % Writes initial model files for all exemplars in the exemplar stream.
    %
    % Parameters
    % ----------
    % e_set : Exemplar stream set
    % params : Configuration parameters
    % models_name : Name of the model to indicate type of training
    % 
    % Returns
    % -------
    % models : Cell array of models
    %
    % Copyright (C) 2011-12 by Tomasz Malisiewicz
    % All rights reserved.
    %
    % This file is part of the Exemplar-SVM library and is made
    % available under the terms of the MIT license (see COPYING file).
    % Project homepage: https://github.com/quantombone/exemplarsvm

    if ~exist('models_name','var')
        models_name = '';
    end

    dir_path =  sprintf('%s/models/', params.dataset_params.localdir);
    file_path = sprintf('%s/%s.mat', dir_path, models_name);
    if file_exists(file_path)
        models = load(file_path);
        models = models.models;
        return;
    end

    results_directory = sprintf('%s/models/%s/',params.dataset_params.localdir, models_name);
    if ~exist(results_directory,'dir')
        fprintf(1,'Making directory %s\n',results_directory);
        mkdir(results_directory);
    end

    if params.dataset_params.display == 1
        r = 1:length(length(e_set));
    else
        % Randomize creation order
        myRandomize;
        r = randperm(length(e_set));
    end
    e_set = e_set(r);

    % Create an array of all final file names
    allfiles = cell(length(e_set), 1);

    for i = 1:length(e_set)
        cls = e_set{i}.cls;
        objectid = e_set{i}.objectid;
        bbox = e_set{i}.bbox;
        curid = e_set{i}.curid;

        path = sprintf('%s/%s',results_directory, e_set{i}.filer);

        allfiles{i} = path;
        if ~isfield(params,'init_params')
            error('Warning, cannot initialize without params.init_params\n');
        end
        
        if file_exists(path)
            continue
        end
        gt_box = bbox;
        fprintf(1,'.');

        I = load_image(e_set{i}.I);

        % Call the init function which is a mapping from (I, bbox) to (model)
        [model] = params.init_params.init_function(I, bbox, params.init_params);

        clear m
        m.model = model;

        % Save filename (or original image)
        m.I = e_set{i}.I;
        m.curid = curid;
        m.objectid = objectid;
        m.cls = cls;
        m.gt_box = gt_box;

        m.sizeI = size(I);
        m.models_name = models_name;
        m.name = sprintf('%s.%d.%s', m.curid, m.objectid, m.cls);

        save(path, 'm');

        % Show the initialized exemplars
        if params.dataset_params.display == 1
            show_exemplar_frames({m}, 1);
            drawnow
            snapnow;
        end
    end

    %sort files so they are in alphabetical order
    [allfiles, ~] = sort(allfiles);

    %Load all of the initialized exemplars
    CACHE_FILE = 1;

    if isempty(models_name)
        CACHE_FILE = 0;
    end

    STRIP_FILE = 0;

    models = load_models(params.dataset_params, models_name, allfiles, CACHE_FILE, STRIP_FILE);

    fprintf(1,'\n   --- Done initializing %d exemplars\n',length(e_set));
end

function show_exemplar_frames(allmodels, N_PER_PAGE)
    % Plots the initialized exemplar frame of 3 images :
    % input + GT bounding box + template
    % template mask + GT bounding box
    % HOG descriptor

    chunked_indices = do_partition(1:length(allmodels), N_PER_PAGE);
    for i = 1:length(chunked_indices)
        models = allmodels(chunked_indices{i});
        figure(i)
        clf
        N = length(models);
        for j = 1:N
            o = (j-1)*3;
            m = models{j};
            I = load_image(m.I);
            
            subplot(N, 3, o+1)
            imagesc(I)
            plot_bbox(m.model.bb(1,:),'', [1 0 0], [0 1 0], 0 ,[1 3], m.model.hg_size)
            plot_bbox(m.gt_box,'',[0 0 1])
            title(sprintf('Ex %s.%d %s',m.curid,m.objectid,m.cls))
            axis image
            axis off            

            onimage = m.model.mask*1;
            onimage(onimage == 0) = 2;
            colors = [1 0 0; 0 0 1];
            cim = colors(onimage(:),:);
            cim = reshape(cim,[size(m.model.mask,1) size(m.model.mask,2) 3]);
            fullimbox = [0 0 size(cim,2) size(cim,1)]+.5;
            transform = find_transform(m.model.bb(1,1:4), fullimbox);
            gtprime = apply_transform(m.gt_box, transform);
            [u,v] = find(m.model.mask);
            curselection = [min(v) min(u) max(v) max(u)];
            curos = getosmatrix_bb(curselection, gtprime);
            
            subplot(N, 3, o+2)
            imagesc(cim);
            plot_bbox(fullimbox,'', [1 0 0], [0 1 0], 0 ,[1 3], m.model.hg_size)
            plot_bbox(gtprime,'',[0 0 1])
            title(sprintf('%s: Template:[%d x %d] \ncuros=%.2f Mask: [%d x %d]', m.model.init_params.init_type, m.model.hg_size(1),m.model.hg_size(2), curos,range(u)+1,range(v)+1));          
            axis image
            axis off
            grid on

            hogim = HOGpicture(repmat(m.model.mask,[1 1 size(m.model.w,3)]).* m.model.w);
            
            subplot(N, 3, o+3)
            imagesc(hogim)
            axis image
            axis off
            grid on
            title('HOG features')
            drawnow
        end
    end
end