results = batchs('Input Image Path', 'Mask Image Path','Output Directory');
function [  segmented_image, comprehensive_metrics] = process_single_image_comprehensive(image_path, gt_path)
% Process single image and calculate comprehensive metrics
    
    % Load images
    I_original = imread(image_path);
    I_GroundTruth = imread(gt_path);

    % Preprocess
    target_size = [512 512];
    I_resized = imresize(I_original, target_size);
    I_no_vessels = removal_vessel(I_original);
%     I_no_vessels_resized = imresize(I_no_vessels, target_size);

    % Get optic disc point
    [~, optic_disc_point] = process_retinal_patch(I_resized);
    scaled_point = scale_point_to_original(optic_disc_point, [600 600], size(I_original));

    % Create circular mask for segmentation initialization
    radius = 5;
    [x, y] = meshgrid(1:size(I_original, 2), 1:size(I_original, 1));
    mask = (x - scaled_point(1)).^2 + (y - scaled_point(2)).^2 < radius^2;

    % Perform active contour segmentation
     K=(0.7*I_no_vessels(:,:,1)+0.3*I_no_vessels(:,:,3))/2;
    K=imresize(K,[size(I_original,1),size(I_original,2)]);
    img = double(K) / 255;
% 
% % Set parameters (For Origa Dataset)
% alpha = 3.5;  % Increase high intensity pixels
% beta = 3.5;   % Decrease low intensity pixels
% 
% % Initialize transformed image
% transformed = zeros(size(img));
% 
% % Apply transformation
% transformed(img > 0.5) = img(img > 0.5) .^ alpha;
% transformed(img <= 0.5) = img(img <= 0.5) .^ beta;
% 
% % Scale back to 0-255 and convert to uint8
% result = uint8(255 * mat2gray(transformed));
    BW1=chenvese(K,mask,300);
    close all;
    % Extract and process boundary
    binary_mask = imbinarize(double(BW1));
    binary_mask = extract_largest_component(binary_mask);
    segmented_image = imresize(binary_mask, size(I_original, 1:2));
    
    % Calculate comprehensive metrics
    comprehensive_metrics = calculate_comprehensive_metrics( segmented_image, imbinarize(I_GroundTruth));
end
function results = batchs(input_directory, ground_truth_directory, output_directory)
% BATCHS_WITH_ELLIPSE - Enhanced batch processing with ellipse fitting and visualization
% 
% Usage: results = batchs_with_ellipse('input_folder', 'gt_folder', 'output_folder')
% 
% Inputs:
%   - input_directory: Path to original images
%   - ground_truth_directory: Path to ground truth segmentations
%   - output_directory: Path where segmented images and ellipses will be saved
% 
% Returns: Structure array with comprehensive metrics, ellipse parameters, and results
    
    % Create output directory if it doesn't exist
    if ~exist(output_directory, 'dir')
        mkdir(output_directory);
        fprintf('Created output directory: %s\n', output_directory);
    end
    
    % Create subdirectories for different outputs
    seg_output_dir = fullfile(output_directory, 'segmented_images');
    ellipse_output_dir = fullfile(output_directory, 'ellipse_visualizations');
    overlay_output_dir = fullfile(output_directory, 'overlay_images');
    metrics_output_dir = fullfile(output_directory, 'metrics');
    
    if ~exist(seg_output_dir, 'dir'), mkdir(seg_output_dir); end
    if ~exist(ellipse_output_dir, 'dir'), mkdir(ellipse_output_dir); end
    if ~exist(overlay_output_dir, 'dir'), mkdir(overlay_output_dir); end
    if ~exist(metrics_output_dir, 'dir'), mkdir(metrics_output_dir); end
    
    % Get list of image files
    image_extensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif'};
    image_files = [];
    
    for ext = image_extensions
        files = dir(fullfile(input_directory, ext{1}));
        image_files = [image_files; files];
    end
    
    % Initialize enhanced results structure with ellipse data
    results = struct('filename', {}, 'segmented_image', {}, 'metrics', {}, ...
                    'processing_time', {}, 'output_paths', {}, 'ellipse_params', {}, ...
                    'ellipse_center', {}, 'ellipse_metrics', {});
    
    % Initialize counters and arrays for summary statistics
    total_images = length(image_files);
    processed_count = 0;
    error_count = 0;
    error_files = {};
    all_metrics = [];
    all_ellipse_metrics = [];
    
    fprintf('Found %d images to process...\n', total_images);
    fprintf('Output directory: %s\n', output_directory);
    
    % Process each image
    for i = 1:total_images
        current_file = image_files(i).name;
        full_path = fullfile(input_directory, current_file);
        
        % Find corresponding ground truth file
        [~, name, ~] = fileparts(current_file);
        gt_file = find_ground_truth_file(ground_truth_directory, name);
        
        fprintf('Processing image %d/%d: %s... ', i, total_images, current_file);
        
        try
            % Start timing
            tic;
            
            % Process the image with comprehensive metrics
            [segmented_image, comprehensive_metrics] = process_single_image_comprehensive(full_path, gt_file);
            
            % Fit ellipse to segmented region
            [ellipse_params, ellipse_center] = fit_ellipse_to_boundary(segmented_image);
            
            % Calculate ellipse-specific metrics
            ellipse_metrics = calculate_ellipse_metrics(segmented_image, ellipse_params, ellipse_center, gt_file);
            
            % Record processing time
            processing_time = toc;
            
            % Save outputs including ellipse visualizations
            output_paths = save_segmentation_outputs_with_ellipse(current_file, segmented_image, ...
                full_path, gt_file, seg_output_dir, ellipse_output_dir, overlay_output_dir, ...
                comprehensive_metrics, ellipse_params, ellipse_center);
            
            % Store results in structure
            processed_count = processed_count + 1;
            results(processed_count).filename = current_file;
            results(processed_count).segmented_image = segmented_image;
            results(processed_count).metrics = comprehensive_metrics;
            results(processed_count).processing_time = processing_time;
            results(processed_count).output_paths = output_paths;
            results(processed_count).ellipse_params = ellipse_params;
            results(processed_count).ellipse_center = ellipse_center;
            results(processed_count).ellipse_metrics = ellipse_metrics;
            
            % Store metrics for summary
            all_metrics = [all_metrics; comprehensive_metrics];
            all_ellipse_metrics = [all_ellipse_metrics; ellipse_metrics];
            
            fprintf('SUCCESS (Dice: %.3f, Ellipse Fit: %.3f, Time: %.2fs)\n', ...
                comprehensive_metrics.dice_score, ellipse_metrics.ellipse_overlap_score, processing_time);
            
        catch ME
            error_count = error_count + 1;
            error_files{end+1} = current_file;
            fprintf('ERROR: %s\n', ME.message);
        end
    end
    
    % Generate comprehensive summary report with ellipse analysis
    generate_summary_report_with_ellipse(results, all_metrics, all_ellipse_metrics, ...
        metrics_output_dir, total_images, processed_count, error_count, error_files);
    
    % Save results to MAT file
    save(fullfile(metrics_output_dir, 'batch_processing_results_with_ellipse.mat'), 'results');
    
    fprintf('\nProcessing complete! Results saved to: %s\n', output_directory);
end

function [ellipse_params, center] = fit_ellipse_to_boundary(binary_mask)
    % Enhanced ellipse fitting with error handling and validation
 
        % Clean the binary mask first
        binary_mask = extract_largest_component(binary_mask);
        
        % Get region properties
        props = regionprops(binary_mask, 'Centroid', 'MajorAxisLength', 'MinorAxisLength', 'Orientation', 'Area');
        
        if isempty(props)
            % Return default ellipse if no regions found
            center = [size(binary_mask, 2)/2, size(binary_mask, 1)/2];
            ellipse_params = struct('major_axis', 10, 'minor_axis', 8, 'orientation', 0, 'area', 0);
            return;
        end
        
        center = props(1).Centroid;
        
        % Handle multiple regions if present
        if length(props) > 1
            % Select the largest region
            areas = [props.Area];
            [~, max_idx] = max(areas);
            selected_props = props(max_idx);
        else
            selected_props = props(1);
        end
        
        % Extract ellipse parameters
        major_axis = selected_props.MajorAxisLength / 2;
        minor_axis = selected_props.MinorAxisLength / 2;
        orientation = deg2rad(-selected_props.Orientation);
        area = selected_props.Area;
        
        % Validate parameters
        if major_axis == 0 || minor_axis == 0
            major_axis = max(major_axis, 1);
            minor_axis = max(minor_axis, 1);
        end
        
        % Return ellipse parameters
        ellipse_params = struct('major_axis', major_axis, 'minor_axis', minor_axis, ...
                               'orientation', orientation, 'area', area);
        

end

function [ellipse_x, ellipse_y] = generate_ellipse_points(ellipse_params, center, num_points)
    % Generate points along ellipse boundary with configurable resolution
    if nargin < 3
        num_points = 100;
    end
    
    angle = linspace(0, 2 * pi, num_points);
    
    % Extract parameters
    major_axis = ellipse_params.major_axis;
    minor_axis = ellipse_params.minor_axis;
    orientation = ellipse_params.orientation;
    
    % Generate ellipse coordinates
    ellipse_x = center(1) + major_axis * cos(angle) * cos(orientation) - minor_axis * sin(angle) * sin(orientation);
    ellipse_y = center(2) + major_axis * cos(angle) * sin(orientation) + minor_axis * sin(angle) * cos(orientation);
end

function ellipse_mask = create_ellipse_mask(ellipse_params, center, image_size)
    % Create binary mask from ellipse parameters
    [X, Y] = meshgrid(1:image_size(2), 1:image_size(1));
    
    % Transform coordinates to ellipse coordinate system
    X_centered = X - center(1);
    Y_centered = Y - center(2);
    
    % Rotate coordinates
    cos_theta = cos(ellipse_params.orientation);
    sin_theta = sin(ellipse_params.orientation);
    
    X_rot = X_centered * cos_theta + Y_centered * sin_theta;
    Y_rot = -X_centered * sin_theta + Y_centered * cos_theta;
    
    % Create ellipse mask
    ellipse_mask = (X_rot.^2 / ellipse_params.major_axis^2) + (Y_rot.^2 / ellipse_params.minor_axis^2) <= 1;
end

function ellipse_metrics = calculate_ellipse_metrics(segmented_mask, ellipse_params, ellipse_center, gt_path)
    % Calculate metrics specific to ellipse fitting quality
    try
        % Load ground truth
        gt_mask = imread(gt_path);
        gt_mask = logical(gt_mask);
        
        % Create ellipse mask
        ellipse_mask = create_ellipse_mask(ellipse_params, ellipse_center, size(segmented_mask));
        
        % Calculate overlap between segmentation and fitted ellipse
        seg_ellipse_overlap = sum(segmented_mask(:) & ellipse_mask(:)) / sum(segmented_mask(:) | ellipse_mask(:));
        
        % Calculate how well ellipse represents the ground truth
        gt_ellipse_overlap = sum(gt_mask(:) & ellipse_mask(:)) / sum(gt_mask(:) | ellipse_mask(:));
        
        % Calculate ellipse-based Dice score with ground truth
        ellipse_gt_dice = 2 * sum(gt_mask(:) & ellipse_mask(:)) / (sum(gt_mask(:)) + sum(ellipse_mask(:)));
        
        % Calculate eccentricity
        eccentricity = sqrt(1 - (ellipse_params.minor_axis^2 / ellipse_params.major_axis^2));
        
        % Calculate aspect ratio
        aspect_ratio = ellipse_params.major_axis / ellipse_params.minor_axis;
        
        % Package metrics
        ellipse_metrics = struct();
        ellipse_metrics.ellipse_overlap_score = seg_ellipse_overlap;
        ellipse_metrics.gt_ellipse_overlap = gt_ellipse_overlap;
        ellipse_metrics.ellipse_gt_dice = ellipse_gt_dice;
        ellipse_metrics.eccentricity = eccentricity;
        ellipse_metrics.aspect_ratio = aspect_ratio;
        ellipse_metrics.ellipse_area = pi * ellipse_params.major_axis * ellipse_params.minor_axis;
        
    catch ME
        % Handle errors by returning default values
        fprintf('Warning: Error calculating ellipse metrics: %s\n', ME.message);
        ellipse_metrics = struct();
        ellipse_metrics.ellipse_overlap_score = 0;
        ellipse_metrics.gt_ellipse_overlap = 0;
        ellipse_metrics.ellipse_gt_dice = 0;
        ellipse_metrics.eccentricity = 0;
        ellipse_metrics.aspect_ratio = 1;
        ellipse_metrics.ellipse_area = 0;
    end
end

function output_paths = save_segmentation_outputs_with_ellipse(filename, segmented_image, original_path, gt_path, ...
    seg_dir, ellipse_dir, overlay_dir, ~, ellipse_params, ellipse_center)
    % Save various outputs including ellipse visualizations
    
    [~, name, ~] = fileparts(filename);
    
    % Initialize output paths structure
    output_paths = struct();
    
    % Save segmented binary mask
    seg_filename = [name, '_segmented.png'];
    seg_path = fullfile(seg_dir, seg_filename);
    imwrite(segmented_image, seg_path);
    output_paths.segmented = seg_path;
    
    % Create and save ellipse visualization
    try
        original_img = imread(original_path);
        ellipse_vis = create_ellipse_visualization(original_img, segmented_image, ellipse_params, ellipse_center);
        
        ellipse_filename = [name, '_ellipse.png'];
        ellipse_path = fullfile(ellipse_dir, ellipse_filename);
        imwrite(ellipse_vis, ellipse_path);
        output_paths.ellipse = ellipse_path;
        
    catch ME
        fprintf('Warning: Could not create ellipse visualization for %s: %s\n', filename, ME.message);
        output_paths.ellipse = '';
    end
    
    % Create and save overlay image with ellipse
    try
        original_img = imread(original_path);
        gt_img = imread(gt_path);
        
        overlay_img = create_comparison_overlay_with_ellipse(original_img, segmented_image, gt_img, ellipse_params, ellipse_center);
        
        overlay_filename = [name, '_overlay_ellipse.png'];
        overlay_path = fullfile(overlay_dir, overlay_filename);
        imwrite(overlay_img, overlay_path);
        output_paths.overlay = overlay_path;
        
    catch ME
        fprintf('Warning: Could not create overlay for %s: %s\n', filename, ME.message);
        output_paths.overlay = '';
    end
end

function ellipse_vis = create_ellipse_visualization(original_img, segmented_mask, ellipse_params, ellipse_center)
    % Create visualization showing original image, segmentation, and fitted ellipse
    
    % Convert to RGB if grayscale
    if size(original_img, 3) == 1
        vis_img = repmat(original_img, [1, 1, 3]);
    else
        vis_img = original_img;
    end
    
    % Generate ellipse points
    [ellipse_x, ellipse_y] = generate_ellipse_points(ellipse_params, ellipse_center, 200);
    
    % Create figure (invisible for batch processing)
    fig = figure('Visible', 'off');
    
    % Display original image
    imshow(vis_img);
    hold on;
    
    % Overlay segmentation boundary
    seg_boundary = bwperim(segmented_mask);
    [seg_y, seg_x] = find(seg_boundary);
    plot(seg_x, seg_y, 'r.', 'MarkerSize', 2);
    
    % Plot fitted ellipse
    plot(ellipse_x, ellipse_y, 'b-', 'LineWidth', 2);
    
    % Mark center
    plot(ellipse_center(1), ellipse_center(2), 'go', 'MarkerSize', 8, 'MarkerFaceColor', 'g');
    
    % Add legend
    legend('Segmentation Boundary', 'Fitted Ellipse', 'Center', 'Location', 'best');
    title('Ellipse Fitting Results');
    
    % Capture the figure
    frame = getframe(fig);
    ellipse_vis = frame.cdata;
    
    % Close figure
    close(fig);
end

function overlay_img = create_comparison_overlay_with_ellipse(original_img, predicted_mask, gt_mask, ellipse_params, ellipse_center)
    % Create overlay with segmentation, ground truth, and fitted ellipse
    
    % Convert to RGB if grayscale
    if size(original_img, 3) == 1
        original_img = repmat(original_img, [1, 1, 3]);
    end
    
    % Ensure masks are binary
    predicted_mask = logical(predicted_mask);
    gt_mask = logical(gt_mask);
    
    % Create ellipse mask
    ellipse_mask = create_ellipse_mask(ellipse_params, ellipse_center, size(predicted_mask));
    
    % Create overlay
    overlay_img = double(original_img);
    
    % Define colors
    gt_color = [0, 1, 0];      % Green for ground truth
    pred_color = [1, 0, 0];    % Red for prediction
    ellipse_color = [0, 0, 1]; % Blue for ellipse
    overlap_color = [1, 1, 0]; % Yellow for overlap
    
    % Apply colors with transparency
    alpha = 0.3;
    
    % Ground truth overlay
    for c = 1:3
        overlay_img(:,:,c) = overlay_img(:,:,c) .* ~gt_mask + ...
                            (overlay_img(:,:,c) * (1-alpha) + gt_color(c) * alpha * 255) .* gt_mask;
    end
    
    % Prediction overlay
    for c = 1:3
        overlay_img(:,:,c) = overlay_img(:,:,c) .* ~predicted_mask + ...
                            (overlay_img(:,:,c) * (1-alpha) + pred_color(c) * alpha * 255) .* predicted_mask;
    end
    
    % Ellipse boundary overlay
    ellipse_boundary = bwperim(ellipse_mask);
    for c = 1:3
        overlay_img(:,:,c) = overlay_img(:,:,c) .* ~ellipse_boundary + ...
                            ellipse_color(c) * 255 .* ellipse_boundary;
    end
    
    overlay_img = uint8(overlay_img);
end

function generate_summary_report_with_ellipse(results, all_metrics, all_ellipse_metrics, output_dir, total_images, processed_count, error_count, error_files)
    % Generate comprehensive summary report including ellipse analysis
    
    if processed_count == 0
        fprintf('No images were successfully processed.\n');
        return;
    end
    
    % Calculate summary statistics for segmentation metrics
    metric_fields = fieldnames(all_metrics(1));
    summary_stats = struct();
    
    for i = 1:length(metric_fields)
        field = metric_fields{i};
        if isnumeric(all_metrics(1).(field))
            values = [all_metrics.(field)];
            summary_stats.(field) = struct();
            summary_stats.(field).mean = mean(values);
            summary_stats.(field).std = std(values);
            summary_stats.(field).min = min(values);
            summary_stats.(field).max = max(values);
            summary_stats.(field).median = median(values);
        end
    end
    
    % Calculate summary statistics for ellipse metrics
    ellipse_fields = fieldnames(all_ellipse_metrics(1));
    ellipse_summary_stats = struct();
    
    for i = 1:length(ellipse_fields)
        field = ellipse_fields{i};
        if isnumeric(all_ellipse_metrics(1).(field))
            values = [all_ellipse_metrics.(field)];
            ellipse_summary_stats.(field) = struct();
            ellipse_summary_stats.(field).mean = mean(values);
            ellipse_summary_stats.(field).std = std(values);
            ellipse_summary_stats.(field).min = min(values);
            ellipse_summary_stats.(field).max = max(values);
            ellipse_summary_stats.(field).median = median(values);
        end
    end
    
    % Create detailed report
    report_file = fullfile(output_dir, 'segmentation_ellipse_report.txt');
    fid = fopen(report_file, 'w');
    
    fprintf(fid, '=== OPTIC DISC SEGMENTATION WITH ELLIPSE FITTING REPORT ===\n\n');
    fprintf(fid, 'Processing Date: %s\n', datestr(now));
    fprintf(fid, 'Total Images: %d\n', total_images);
    fprintf(fid, 'Successfully Processed: %d\n', processed_count);
    fprintf(fid, 'Errors: %d\n\n', error_count);
    
    fprintf(fid, '=== SEGMENTATION METRICS SUMMARY ===\n');
    fprintf(fid, 'Metric\t\t\tMean ± Std\t\tMin\t\tMax\t\tMedian\n');
    fprintf(fid, '----------------------------------------------------------------\n');
    
    key_metrics = {'dice_score', 'jaccard_score', 'sensitivity', 'specificity', 'precision', 'accuracy'};
    for i = 1:length(key_metrics)
        if isfield(summary_stats, key_metrics{i})
            stats = summary_stats.(key_metrics{i});
            fprintf(fid, '%-15s\t%.3f ± %.3f\t\t%.3f\t\t%.3f\t\t%.3f\n', ...
                key_metrics{i}, stats.mean, stats.std, stats.min, stats.max, stats.median);
        end
    end
    
    fprintf(fid, '\n=== ELLIPSE FITTING METRICS SUMMARY ===\n');
    fprintf(fid, 'Metric\t\t\tMean ± Std\t\tMin\t\tMax\t\tMedian\n');
    fprintf(fid, '----------------------------------------------------------------\n');
    
    ellipse_key_metrics = {'ellipse_overlap_score', 'gt_ellipse_overlap', 'ellipse_gt_dice', 'eccentricity', 'aspect_ratio'};
    for i = 1:length(ellipse_key_metrics)
        if isfield(ellipse_summary_stats, ellipse_key_metrics{i})
            stats = ellipse_summary_stats.(ellipse_key_metrics{i});
            fprintf(fid, '%-20s\t%.3f ± %.3f\t\t%.3f\t\t%.3f\t\t%.3f\n', ...
                ellipse_key_metrics{i}, stats.mean, stats.std, stats.min, stats.max, stats.median);
        end
    end
    
    if error_count > 0
        fprintf(fid, '\n=== ERROR FILES ===\n');
        for i = 1:length(error_files)
            fprintf(fid, '%s\n', error_files{i});
        end
    end
    
    fclose(fid);
    
    % Save detailed metrics to CSV including ellipse data
    csv_file = fullfile(output_dir, 'detailed_metrics_with_ellipse.csv');
    write_metrics_to_csv_with_ellipse(results, csv_file);
    
    % Display summary to console
    fprintf('\n=== PROCESSING SUMMARY ===\n');
    fprintf('Total images: %d\n', total_images);
    fprintf('Successfully processed: %d\n', processed_count);
    fprintf('Errors encountered: %d\n', error_count);
    fprintf('Average Dice Score: %.3f ± %.3f\n', summary_stats.dice_score.mean, summary_stats.dice_score.std);
    fprintf('Average Ellipse Overlap: %.3f ± %.3f\n', ellipse_summary_stats.ellipse_overlap_score.mean, ellipse_summary_stats.ellipse_overlap_score.std);
    fprintf('Average Aspect Ratio: %.3f ± %.3f\n', ellipse_summary_stats.aspect_ratio.mean, ellipse_summary_stats.aspect_ratio.std);
    fprintf('\nDetailed report saved to: %s\n', report_file);
end

function write_metrics_to_csv_with_ellipse(results, csv_file)
    % Write detailed metrics including ellipse data to CSV file
    
    fid = fopen(csv_file, 'w');
    
    % Write header
    fprintf(fid, 'Filename,Dice_Score,Jaccard_Score,Sensitivity,Specificity,Precision,Accuracy,F1_Score,');
    fprintf(fid, 'Hausdorff_Distance,Volume_Similarity,Boundary_Dice,Avg_Surface_Distance,');
    fprintf(fid, 'Ellipse_Overlap,GT_Ellipse_Overlap,Ellipse_GT_Dice,Eccentricity,Aspect_Ratio,Ellipse_Area,Processing_Time\n');
    
    % Write data
    for i = 1:length(results)
        m = results(i).metrics;
        e = results(i).ellipse_metrics;
        fprintf(fid, '%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f\n', ...
            results(i).filename, m.dice_score, m.jaccard_score, m.sensitivity, m.specificity, ...
            m.precision, m.accuracy, m.f1_score, m.hausdorff_distance, m.volume_similarity, ...
            m.boundary_dice, m.average_surface_distance, e.ellipse_overlap_score, e.gt_ellipse_overlap, ...
            e.ellipse_gt_dice, e.eccentricity, e.aspect_ratio, e.ellipse_area, results(i).processing_time);
    end
    
    fclose(fid);
end

% Keep all the helper functions from your original code
function gt_file = find_ground_truth_file(gt_directory, base_name)
    image_extensions = {'*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif'};
    gt_file = '';
    
    for ext = image_extensions
        potential_files = dir(fullfile(gt_directory, [base_name, ext{1}(2:end)]));
        if ~isempty(potential_files)
            gt_file = fullfile(gt_directory, potential_files(1).name);
            break;
        end
        
        potential_files = dir(fullfile(gt_directory, [base_name, '_gt', ext{1}(2:end)]));
        if ~isempty(potential_files)
            gt_file = fullfile(gt_directory, potential_files(1).name);
            break;
        end
    end
    
    if isempty(gt_file)
        error('Ground truth file not found for: %s', base_name);
    end
end

function scaled_point = scale_point_to_original(point, target_size, original_size)
    scaled_point = zeros(1, 2);
    scaled_point(1) = floor(((point(2) / target_size(2)) * original_size(2)));
    scaled_point(2) = floor(((point(1) / target_size(1)) * original_size(1)));
end

function largest_component = extract_largest_component(binary_image)
    CC = bwconncomp(binary_image);
    
    if CC.NumObjects == 0
        largest_component = binary_image;
        return;
    end
    
    stats = regionprops(CC, 'Area');
    area_values = [stats.Area];
    [~, max_index] = max(area_values);
    
    largest_component = false(size(binary_image));
    largest_component(CC.PixelIdxList{max_index}) = true;
end

% You'll need to include your other helper functions here:
% - process_single_image_comprehensive
% - calculate_comprehensive_metrics
% - calculate_hausdorff_distance
% - calculate_boundary_metrics
% - removal_vessel
% - process_retinal_patch
% - chenvese
% etc.

% Example usage:
% results = batchs_with_ellipse('input_folder', 'gt_folder', 'output_folder')
function comprehensive_metrics = calculate_comprehensive_metrics(segmented_mask, ground_truth_mask)
% CALCULATE_COMPREHENSIVE_METRICS - Calculate detailed segmentation metrics
% 
% Inputs:
%   - segmented_mask: Binary segmentation result
%   - ground_truth_mask: Binary ground truth
% 
% Returns: Structure with comprehensive metrics

    % Ensure both masks are binary
    segmented_binary = logical(segmented_mask);
    gt_binary = logical(ground_truth_mask);
    
    % Basic set operations
    intersection = segmented_binary & gt_binary;
    union_area = segmented_binary | gt_binary;
    
    % Count pixels
    TP = sum(intersection(:));           % True Positives
    FP = sum(segmented_binary(:)) - TP;  % False Positives
    FN = sum(gt_binary(:)) - TP;         % False Negatives
    TN = sum(~segmented_binary(:) & ~gt_binary(:)); % True Negatives
    
    total_pixels = numel(segmented_binary);
    
    % Calculate primary metrics
    dice_score = 2 * TP / (2 * TP + FP + FN);
    jaccard_score = TP / (TP + FP + FN);
    
    % Additional metrics
    sensitivity = TP / (TP + FN);        % Recall/True Positive Rate
    specificity = TN / (TN + FP);        % True Negative Rate
    precision = TP / (TP + FP);          % Positive Predictive Value
    accuracy = (TP + TN) / total_pixels; % Overall accuracy
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity);
    
    % Geometric metrics
    hausdorff_dist = calculate_hausdorff_distance(segmented_binary, gt_binary);
    
    % Boundary-based metrics
    boundary_metrics = calculate_boundary_metrics(segmented_binary, gt_binary);
    
    % Volume metrics
    volume_similarity = 1 - abs(sum(segmented_binary(:)) - sum(gt_binary(:))) / ...
                           (sum(segmented_binary(:)) + sum(gt_binary(:)));
    
    % Package all metrics
    comprehensive_metrics = struct();
    comprehensive_metrics.dice_score = dice_score;
    comprehensive_metrics.jaccard_score = jaccard_score;
    comprehensive_metrics.sensitivity = sensitivity;
    comprehensive_metrics.specificity = specificity;
    comprehensive_metrics.precision = precision;
    comprehensive_metrics.accuracy = accuracy;
    comprehensive_metrics.f1_score = f1_score;
    comprehensive_metrics.hausdorff_distance = hausdorff_dist;
    comprehensive_metrics.volume_similarity = volume_similarity;
    comprehensive_metrics.boundary_dice = boundary_metrics.boundary_dice;
    comprehensive_metrics.average_surface_distance = boundary_metrics.avg_surface_distance;
    
    % Confusion matrix elements
    comprehensive_metrics.true_positives = TP;
    comprehensive_metrics.false_positives = FP;
    comprehensive_metrics.false_negatives = FN;
    comprehensive_metrics.true_negatives = TN;
end

function hausdorff_dist = calculate_hausdorff_distance(mask1, mask2)
% Calculate Hausdorff distance between two binary masks
    try
        % Extract boundaries
        boundary1 = bwperim(mask1);
        boundary2 = bwperim(mask2);
        
        % Get boundary coordinates
        [y1, x1] = find(boundary1);
        [y2, x2] = find(boundary2);
        
        if isempty(x1) || isempty(x2)
            hausdorff_dist = Inf;
            return;
        end
        
        % Calculate distances
        coords1 = [x1, y1];
        coords2 = [x2, y2];
        
        % Distance from each point in set1 to closest point in set2
        dist1to2 = zeros(size(coords1, 1), 1);
        for i = 1:size(coords1, 1)
            distances = sqrt(sum((coords2 - coords1(i, :)).^2, 2));
            dist1to2(i) = min(distances);
        end
        
        % Distance from each point in set2 to closest point in set1
        dist2to1 = zeros(size(coords2, 1), 1);
        for i = 1:size(coords2, 1)
            distances = sqrt(sum((coords1 - coords2(i, :)).^2, 2));
            dist2to1(i) = min(distances);
        end
        
        % Hausdorff distance is the maximum of the minimum distances
        hausdorff_dist = max(max(dist1to2), max(dist2to1));
        
    catch
        hausdorff_dist = Inf;
    end
end

function boundary_metrics = calculate_boundary_metrics(mask1, mask2)
% Calculate boundary-based metrics
    try
        % Extract boundaries
        boundary1 = bwperim(mask1);
        boundary2 = bwperim(mask2);
        
        % Boundary Dice coefficient
        boundary_intersection = boundary1 & boundary2;
        boundary_dice = 2 * sum(boundary_intersection(:)) / ...
                       (sum(boundary1(:)) + sum(boundary2(:)));
        
        % Average surface distance
        [y1, x1] = find(boundary1);
        [y2, x2] = find(boundary2);
        
        if isempty(x1) || isempty(x2)
            avg_surface_distance = Inf;
        else
            coords1 = [x1, y1];
            coords2 = [x2, y2];
            
            % Average distance from boundary1 to boundary2
            total_dist = 0;
            for i = 1:size(coords1, 1)
                distances = sqrt(sum((coords2 - coords1(i, :)).^2, 2));
                total_dist = total_dist + min(distances);
            end
            avg_surface_distance = total_dist / size(coords1, 1);
        end
        
        boundary_metrics.boundary_dice = boundary_dice;
        boundary_metrics.avg_surface_distance = avg_surface_distance;
        
    catch
        boundary_metrics.boundary_dice = 0;
        boundary_metrics.avg_surface_distance = Inf;
    end
end





function output_paths = save_segmentation_outputs(filename, segmented_image, original_path, gt_path, seg_dir, overlay_dir, metrics)
% Save various outputs for each processed image
    
    [~, name, ~] = fileparts(filename);
    
    % Initialize output paths structure
    output_paths = struct();
    
    % Save segmented binary mask
    seg_filename = [name, '_segmented.png'];
    seg_path = fullfile(seg_dir, seg_filename);
    imwrite(segmented_image, seg_path);
    output_paths.segmented = seg_path;
    
    % Create and save overlay image
    try
        original_img = imread(original_path);
        gt_img = imread(gt_path);
        
        % Create overlay with different colors for GT, prediction, and overlap
        overlay_img = create_comparison_overlay(original_img, segmented_image, gt_img);
        
        overlay_filename = [name, '_overlay.png'];
        overlay_path = fullfile(overlay_dir, overlay_filename);
        imwrite(overlay_img, overlay_path);
        output_paths.overlay = overlay_path;
        
    catch ME
        fprintf('Warning: Could not create overlay for %s: %s\n', filename, ME.message);
        output_paths.overlay = '';
    end
end

function overlay_img = create_comparison_overlay(original_img, predicted_mask, gt_mask)
% Create a visual comparison overlay
    
    % Convert to RGB if grayscale
    if size(original_img, 3) == 1
        original_img = repmat(original_img, [1, 1, 3]);
    end
    
    % Ensure masks are binary
    predicted_mask = logical(predicted_mask);
    gt_mask = logical(gt_mask);
    
    % Create overlay
    overlay_img = double(original_img);
    
    % Define colors
    gt_color = [0, 1, 0];      % Green for ground truth
    pred_color = [1, 0, 0];    % Red for prediction
    overlap_color = [1, 1, 0]; % Yellow for overlap
    
    % Apply colors
    overlap = predicted_mask & gt_mask;
    gt_only = gt_mask & ~predicted_mask;
    pred_only = predicted_mask & ~gt_mask;
    
    % Apply overlays with transparency
    alpha = 0.4;
    for c = 1:3
        overlay_img(:,:,c) = overlay_img(:,:,c) .* ~(gt_only | pred_only | overlap) + ...
                            (overlay_img(:,:,c) * (1-alpha) + gt_color(c) * alpha * 255) .* gt_only + ...
                            (overlay_img(:,:,c) * (1-alpha) + pred_color(c) * alpha * 255) .* pred_only + ...
                            (overlay_img(:,:,c) * (1-alpha) + overlap_color(c) * alpha * 255) .* overlap;
    end
    
    overlay_img = uint8(overlay_img);
end

function generate_summary_report(results, all_metrics, output_dir, total_images, processed_count, error_count, error_files)
% Generate comprehensive summary report
    
    if processed_count == 0
        fprintf('No images were successfully processed.\n');
        return;
    end
    
    % Calculate summary statistics
    metric_fields = fieldnames(all_metrics(1));
    summary_stats = struct();
    
    for i = 1:length(metric_fields)
        field = metric_fields{i};
        if isnumeric(all_metrics(1).(field))
            values = [all_metrics.(field)];
            summary_stats.(field) = struct();
            summary_stats.(field).mean = mean(values);
            summary_stats.(field).std = std(values);
            summary_stats.(field).min = min(values);
            summary_stats.(field).max = max(values);
            summary_stats.(field).median = median(values);
        end
    end
    
    % Create detailed report
    report_file = fullfile(output_dir, 'segmentation_report.txt');
    fid = fopen(report_file, 'w');
    
    fprintf(fid, '=== OPTIC DISC SEGMENTATION BATCH PROCESSING REPORT ===\n\n');
    fprintf(fid, 'Processing Date: %s\n', datestr(now));
    fprintf(fid, 'Total Images: %d\n', total_images);
    fprintf(fid, 'Successfully Processed: %d\n', processed_count);
    fprintf(fid, 'Errors: %d\n\n', error_count);
    
    fprintf(fid, '=== SUMMARY STATISTICS ===\n');
    fprintf(fid, 'Metric\t\t\tMean ± Std\t\tMin\t\tMax\t\tMedian\n');
    fprintf(fid, '----------------------------------------------------------------\n');
    
    key_metrics = {'dice_score', 'jaccard_score', 'sensitivity', 'specificity', 'precision', 'accuracy'};
    for i = 1:length(key_metrics)
        if isfield(summary_stats, key_metrics{i})
            stats = summary_stats.(key_metrics{i});
            fprintf(fid, '%-15s\t%.3f ± %.3f\t\t%.3f\t\t%.3f\t\t%.3f\n', ...
                key_metrics{i}, stats.mean, stats.std, stats.min, stats.max, stats.median);
        end
    end
    
    if error_count > 0
        fprintf(fid, '\n=== ERROR FILES ===\n');
        for i = 1:length(error_files)
            fprintf(fid, '%s\n', error_files{i});
        end
    end
    
    fclose(fid);
    
    % Save detailed metrics to CSV
    csv_file = fullfile(output_dir, 'detailed_metrics.csv');
    write_metrics_to_csv(results, csv_file);
    
    % Display summary to console
    fprintf('\n=== PROCESSING SUMMARY ===\n');
    fprintf('Total images: %d\n', total_images);
    fprintf('Successfully processed: %d\n', processed_count);
    fprintf('Errors encountered: %d\n', error_count);
    fprintf('Average Dice Score: %.3f ± %.3f\n', summary_stats.dice_score.mean, summary_stats.dice_score.std);
    fprintf('Average Jaccard Score: %.3f ± %.3f\n', summary_stats.jaccard_score.mean, summary_stats.jaccard_score.std);
    fprintf('\nDetailed report saved to: %s\n', report_file);
end

function write_metrics_to_csv(results, csv_file)
% Write detailed metrics to CSV file
    
    fid = fopen(csv_file, 'w');
    
    % Write header
    fprintf(fid, 'Filename,Dice_Score,Jaccard_Score,Sensitivity,Specificity,Precision,Accuracy,F1_Score,');
    fprintf(fid, 'Hausdorff_Distance,Volume_Similarity,Boundary_Dice,Avg_Surface_Distance,Processing_Time\n');
    
    % Write data
    for i = 1:length(results)
        m = results(i).metrics;
        fprintf(fid, '%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f\n', ...
            results(i).filename, m.dice_score, m.jaccard_score, m.sensitivity, m.specificity, ...
            m.precision, m.accuracy, m.f1_score, m.hausdorff_distance, m.volume_similarity, ...
            m.boundary_dice, m.average_surface_distance, results(i).processing_time);
    end
    
    fclose(fid);
end




% Example usage:
% results = batch_process_images_enhanced('input_folder', 'gt_folder', 'output_folder');