function grid = detect_imageset(imageset, models, params, setname)
    % Apply a set of models (raw exemplars, trained exemplars, dalals,
    % poselets, components, etc) to a set of images.  
    %
    % imageset: a (virtual) set of images, such that
    %   convert_to_I(imageset{i}) returns an image
    % models: Cell array of models
    % params(optional): detection parameters
    % setname(optional): a name of the set, which lets us cache results
    %   on disk
    %
    % Copyright (C) 2011-12 by Tomasz Malisiewicz
    % All rights reserved.
    % 
    % This file is part of the Exemplar-SVM library and is made
    % available under the terms of the MIT license (see COPYING file).
    % Project homepage: https://github.com/quantombone/exemplarsvm
    
    if ~exist('params','var')
        params = get_default_params;
    end

    if ~exist('setname','var')
        params.detect_images_per_chunk = 1;
        setname = '';
    end

    if isfield(params,'dataset_params') && isfield(params.dataset_params,'localdir') && ~isempty(params.dataset_params.localdir) && ~isempty(setname)
        save_files = 1;
    else
        save_files = 0;
    end

    if isempty(imageset)
        grid = {};
        return;
    end

    if save_files == 1
        models_name = '';
        if length(models)>=1 && isfield(models{1},'models_name') && isstr(models{1}.models_name)
            models_name = models{1}.models_name;
        end
        final_file = sprintf('%s/detections/%s-%s.mat', params.dataset_params.localdir,setname, models_name);

        if fileexists(final_file)
            res = load(final_file);
            grid = res.grid;
            return;
        end
    end

    if save_files == 1
        baser = sprintf('%s/detections/%s-%s/',params.dataset_params.localdir,setname, models_name);
    else
        baser = '';
    end

    if (save_files == 1) && ~exist(baser,'dir')
        fprintf(1,'Making directory %s\n',baser);
        mkdir(baser);
    end

    % Chunk the data into detect_images_per_chunk images per chunk so that we
    % process several images, then write results for entire chunk
    inds = do_partition(1:length(imageset),params.detect_images_per_chunk);

    % randomize chunk orderings
    myRandomize;
    ordering = randperm(length(inds));

    allfiles = cell(length(ordering), 1);
    counter = 0;
    for i = 1:length(ordering)
        ind1 = inds{ordering(i)}(1);
        ind2 = inds{ordering(i)}(end);
        filer = sprintf('%s/result_%05d-%05d.mat',baser,ind1,ind2);
        allfiles{i} = filer;
        filerlock = [filer '.lock'];

        if save_files == 1
            if fileexists(filer) || mymkdir_dist(filerlock) == 0
                continue
            end
        end
        res = cell(0,1);

        % pre-load all images in a chunk
        clear Is;
        Is = imageset(inds{ordering(i)});
        L = length(inds{ordering(i)});
        
        for j = 1:L
            index = inds{ordering(i)}(j);
            fprintf(1,' --image %05d/%05d:',counter+j,length(imageset));
            Iname = imageset{index};
            try
                hit = strfind(Iname,'JPEGImages/');
                curid = Iname((hit+11):end);
                hit = strfind(curid,'.');
                curid = curid(1:(hit(end)-1));      
            catch
                curid = '';
            end
            I = load_image(Is{j});
            starter = tic;
            rs = detect(I, models, params);

            coarse_boxes = cat(1,rs.bbs{:});
            if ~isempty(coarse_boxes)
                coarse_boxes(:,11) = index;
                scores = coarse_boxes(:,end);
            else
                scores = [];
            end
            [aa,~] = max(scores);
            fprintf(1,' %d exemplars took %.3fsec, #windows=%05d, max=%.3f \n', length(models),toc(starter),length(scores),aa);

            % Transfer GT boxes from models onto the detection windows
            boxes = adjust_boxes(coarse_boxes,models);

            if params.detect_min_scene_os > 0.0
                os = iou(boxes,[1 1 size(I,2) size(I,1)]);
                goods = find(os >= params.detect_min_scene_os);
                boxes = boxes(goods,:);
                coarse_boxes = coarse_boxes(goods,:);
            end

            extras = [];
            res{j}.coarse_boxes = coarse_boxes;
            res{j}.bboxes = boxes;

            res{j}.index = index;
            res{j}.extras = extras;
            res{j}.imbb = [1 1 size(I,2) size(I,1)];
            res{j}.curid = curid;

            % NOTE: the gt-function is well-defined for VOC-exemplars
            if isfield(params,'gt_function') && ~isempty(params.gt_function)
                res{j}.extras = params.gt_function(params.dataset_params, Iname, res{j}.bboxes);
            end
        end

        counter = counter + L;

        % save results into file and remove lock file

        if save_files == 1
            save(filer,'res');
            try
                rmdir(filerlock);
            catch
                fprintf(1,'Directory %s already gone\n',filerlock);
            end
        else
            allfiles{i} = res;
        end
    end

    if save_files == 0
        grid = cellfun2(@(x)x{1},allfiles);
        return;
    end

    [allfiles] = sort(allfiles);
    grid = load_result_grid(params.dataset_params, models, setname, allfiles);
end

function grid = load_result_grid(dataset_params, models,setname,files,curthresh)
    % Given a set of models, return a grid of results from those models' firings
    % on the subset of images (target_directory is 'trainval' or 'test')
    % [curthresh]: only keep detections above this number (-1.1 for
    % 
    %
    % Copyright (C) 2011-12 by Tomasz Malisiewicz
    % All rights reserved.
    % 
    % This file is part of the Exemplar-SVM library and is made
    % available under the terms of the MIT license (see COPYING file).
    % Project homepage: https://github.com/quantombone/exemplarsvm
    
    fullsetname = [setname];
    models_name = '';
    if length(models)>=1 && isfield(models{1},'models_name') && isstr(models{1}.models_name)
        models_name = models{1}.models_name;
    end

    final_file = sprintf('%s/detections/%s-%s.mat', dataset_params.localdir,fullsetname, models_name);

    if fileexists(final_file)
        res = load(final_file);
        grid = res.grid;
        return;
    end

    wait_until_all_present(files,5);

    if ~exist('curthresh','var')
        curthresh = -1.1;
    end

    lockfile = [final_file '.lock'];

    if fileexists(final_file) || mymkdir_dist(lockfile) == 0
        %wait until lockfile is gone
        wait_until_all_present({lockfile},5,1);
        fprintf(1,'Loading final file %s\n',final_file);
        res = load_keep_trying(final_file,5);
        grid = res.grid;
        return;
    end

    %with the dir command partial results could be loaded 
    %files = dir([baser 'result*mat']);
    grid = cell(1,length(files));

    for i = 1:length(files)
        if mod(i,100) == 0
            fprintf(1,'%d/%d\n',i,length(files));
        end
        filer = files{i};
        stuff = load(filer);
        grid{i} = stuff;

        for j = 1:length(grid{i}.res)
            index = grid{i}.res{j}.index;
            if size(grid{i}.res{j}.bboxes,1) > 0
                grid{i}.res{j}.bboxes(:,11) = index;
                grid{i}.res{j}.coarse_boxes(:,11) = index;
                goods = find(grid{i}.res{j}.bboxes(:,end) >= curthresh);
                grid{i}.res{j}.bboxes = grid{i}.res{j}.bboxes(goods,:);
                grid{i}.res{j}.coarse_boxes = grid{i}.res{j}.coarse_boxes(goods,:);

                if ~isempty(grid{i}.res{j}.extras)
                    grid{i}.res{j}.extras.maxos = grid{i}.res{j}.extras.maxos(goods);
                    grid{i}.res{j}.extras.maxind = grid{i}.res{j}.extras.maxind(goods);
                    grid{i}.res{j}.extras.maxclass = grid{i}.res{j}.extras.maxclass(goods);
                end
            end
        end
    end

    %Prune away files which didn't load
    lens = cellfun(@(x)length(x),grid);
    grid = grid(lens>0);
    grid = cellfun2(@(x)x.res,grid);
    grid2 = grid;
    grid = [grid2{:}];

    if ~isempty(grid)
        %sort grids by image index
        [~,bb] = sort(cellfun(@(x)x.index,grid));
        grid = grid(bb);
    end

    save(final_file,'grid');

    if exist(lockfile,'dir')
        rmdir(lockfile);
    end

    f = dir(final_file);
    if (f.bytes < 1000)
        fprintf(1,'warning file too small, not saved\n');
        delete(final_file);
    end

    if fileexists(final_file)
        % Clean up individual files which were written to disk
        for i = 1:length(files)
            delete(files{i});
        end
        % Delete directory too
        [basedir,~,~] = fileparts(files{1});
        rmdir(basedir);
    end
end

function top = adjust_boxes(boxes, models)
    % Adjusts coarse-frame detections into ground-truth frames.

    top = boxes;

    if length(models) >= 1 && (strcmp(models{1}.models_name,'dalal') || ~isfield(models{1},'gt_box'))
        return;
    end

    if numel(boxes) == 0
        return;
    end

    top(:,1:4) = 0;

    for i = 1:size(boxes,1)
        d = boxes(i,:);
        c = models{boxes(i,6)}.model.bb(1,1:4);
        gt = models{boxes(i,6)}.gt_box;

        % find the transform from c to d
        xform = find_transform(c, d(1:4));
        top(i,1:4) = apply_transform(gt, xform);
    end
end