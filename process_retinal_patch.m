function [K31, K41] = process_retinal_patch(I_4)
    % Resize and preprocess the image
    I = imresize(I_4, [600 600]);
    I_3 = preprocess(I);  % Custom preprocessing function
    G = im2gray(I_3);
    f = im2double(G);
    
    % FDCT Parameters
    N = 25;
    K2 = 1;  
    k1 = 2;  
    r0 = 10:10:20;

    % Apply FDCT
    [Cl1, ~] = fdct2_mod(f, k1, N, r0, 'abs');
    [Cl2, ~] = fdct2_mod(f, K2, N, r0, 'abs');
    Cl = [Cl1, Cl2];

    % Patch parameters
    patch_size = [60, 60];
    max_1 = [];
    
    % Loop over FDCT coefficient images
    for i = 1:2
        G1 = uint8(255 * Cl{2, i} / max(max(Cl{2, i})));
        image = G1;
        [height, width, ~] = size(image);
        patch_counter = 1;

        for row = 1:patch_size(1):height
            for col = 1:patch_size(2):width
                row_end = min(row + patch_size(1) - 1, height);
                col_end = min(col + patch_size(2) - 1, width);
                patch = image(row:row_end, col:col_end, :);
                max_1(i, patch_counter) = max(max(patch));
                patch_counter = patch_counter + 1;
            end
        end
    end

    % Vessel segmentation
    Clean_Image = vessel_seg(I_3);
    Clean_Image = imresize(Clean_Image, [600, 600]);
    
    % Entropy-based feature extraction
    patch_counter = 1;
    for row = 1:patch_size(1):height
        for col = 1:patch_size(2):width
            row_end = min(row + patch_size(1) - 1, height);
            col_end = min(col + patch_size(2) - 1, width);
            patch = Clean_Image(row:row_end, col:col_end, :);
            var_1(1, patch_counter) = entropy(patch);
            patch_counter = patch_counter + 1;
        end
    end

    % Mean intensity
    image = im2gray(I);
    patch_counter = 1;
    for row = 1:patch_size(1):height
        for col = 1:patch_size(2):width
            row_end = min(row + patch_size(1) - 1, height);
            col_end = min(col + patch_size(2) - 1, width);
            patch = image(row:row_end, col:col_end, :);
            mean_1(1, patch_counter) = mean(patch(:));
            patch_counter = patch_counter + 1;
        end
    end

    % Feature matrix
    features = [double(max_1); double(var_1)];
    X = features';

    % Normalize features
    X_scaled = (X - min(X)) ./ (max(X) - min(X));
    thres = mean(X_scaled(:, 3)) / 2;
    rows_to_zero = X_scaled(:, 3) <= thres;
    X_scaled(rows_to_zero, :) = 0;

    patches_features = X_scaled;

    % Weighted KMeans Clustering
    numClusters = 3; 
    b = compute_MCI(X_scaled);  % Custom function
    [~, ~, weights2] = weighted_Kmeans(patches_features, numClusters, b, 200);
    linear_combination = patches_features * weights2';
    [~, y] = max(linear_combination);
    
    % Patch refinement
    patch_check = [y, y+1, y-1, y-10, y+10];
    [~, y1] = max(mean_1(patch_check));
    patch_final = patch_check(y1);

    % Get high intensity pixels
    pts = find_high_intensity_pixel(I_3, patch_final);

    % Repeat with equal weights
    weights = [1/3, 1/3, 1/3];
    linear_combination = patches_features * weights';
    [~, y] = max(linear_combination);
    patch_check = [y, y+1, y-1, y-10, y+10];
    [~, y1] = max(mean_1(patch_check));
    patch_final = patch_check(y1);

    pts2 = find_high_intensity_pixel(I_3, patch_final);
%     hFig = figure;
% imshow(I);
% hold on;
% plot(pts2(:,2), pts2(:,1), 'go', 'MarkerSize', 5, 'LineWidth', 2);
% plot(pts(:,2), pts(:,1), 'r*', 'MarkerSize', 5, 'LineWidth', 2);
% hold off;
% pause;
    % Output
    K31 = pts2;
    K41 = pts;
end
