function fg = get_pascal_stream(stream_params, dataset_params)
    % Create an exemplar stream, such that each element fg{i} contains
    % these fields: (I, bbox, cls, curid, filer, [objectid], [anno])
    % Make sure the exemplar has a segmentation associated with it if
    % must_have_seg is provided

    basedir = fullfile(dataset_params.localdir, 'models', 'streams');
    if ~exist(basedir,'dir')
        mkdir(basedir);
    end
    
    streamname = fullfile(basedir, sprintf('%s-%s-%d-%s%s.mat', stream_params.stream_set_name, stream_params.cls, stream_params.stream_max_ex, stream_params.model_type, stream_params.must_have_seg_string));
    
    if file_exists(streamname)
        fprintf(1,'Loading %s\n',streamname);
        fg = load(streamname);
        fprintf(1,'get_pascal_stream: length of stream=%05d\n', length(fg));
        return;
    end

    if ~isempty(stream_params.cls)
        % Load ids of all images in trainval that contain cls
        [ids,gt] = textread(sprintf(dataset_params.clsimgsetpath,stream_params.cls, stream_params.stream_set_name), '%s %d');
        ids = ids(gt == 1);
        all_recs = cellfun2(@(x)sprintf(dataset_params.annopath,x),ids);
    end

    fg = cell(0,1);
    for i = 1:length(all_recs)
        curid = ids{i};
        if ischar(all_recs{i})
            recs = PASreadrecord(all_recs{i});
        else
            recs = all_recs{i};
        end

        if stream_params.must_have_seg && (recs.segmented == 0)
            %skip over unsegmented images
            continue
        end

        if ischar(all_recs{i})
            filename = sprintf(dataset_params.imgpath,curid);
        end

        if strcmp(stream_params.model_type,'exemplar')
            for objectid = 1:length(recs.objects)
                isinclass = ismember({recs.objects(objectid).class},{stream_params.cls});
                if isempty(stream_params.cls)
                    isinclass = 1;
                end
                %skip difficult objects, and objects not of target class
                if (recs.objects(objectid).difficult==1) || ~isinclass
                    continue
                end
                fprintf(1,'.');

                res.I = filename;
                res.bbox = recs.objects(objectid).bbox;
                res.cls = recs.objects(objectid).class;%stream_params.cls;
                res.objectid = objectid;
                res.curid = curid;

                %anno is the data-set-specific version
                res.anno = recs.objects(objectid);

                res.filer = sprintf('%s.%d.%s.mat', curid, objectid, stream_params.cls);

                fg{end+1} = res;

                if length(fg) == stream_params.stream_max_ex
                    save(streamname,'fg');
                    fprintf(1, 'get_pascal_stream: length of stream = %05d\n',length(fg));
                    return;
                end
            end
        elseif strcmp(stream_params.model_type,'scene')
            fprintf(1,'.');
            res.I = filename;

            %Use the entire scene (remember VOC stores imgsize in a strange order)
            res.bbox = [1 1 recs.imgsize(1) recs.imgsize(2)];
            res.cls = stream_params.cls;

            %for scenes use a 1 for objectid
            res.objectid = 1;

            %anno is the data-set-specific version
            res.anno = recs.objects;

            res.filer = sprintf('%s.%d.%s.mat', curid, res.objectid, stream_params.cls);

            fg{end+1} = res;

            if length(fg) == stream_params.stream_max_ex
                save(streamname,'fg');
                fprintf(1,'get_pascal_stream: length of stream = %05d\n', length(fg));
                return;
            end
        else
            error('Invalid model_type %s\n',stream_params.model_type);
        end
    end

    save(streamname,'fg');
    fprintf(1,'get_pascal_stream: length of stream = %05d\n', length(fg));
end
