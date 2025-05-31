function [max_position] = find_high_intensity_pixel(I_3,patch_final)
% Initialize patch counter
patch_counter = 1;

% Define patch size
patch_size = [60, 60];

% Initialize variables for maximum intensity and position
max_intensity = -Inf;
max_position = [0, 0];
[height,width]=size(I_3,1:2);
% Loop through the image to extract patches
for row = 1:patch_size(1):height
    for col = 1:patch_size(2):width
        % Define the patch region
        row_end = min(row + patch_size(1) - 1, height);
        col_end = min(col + patch_size(2) - 1, width);
        
        % Extract the patch
        patch = I_3(row:row_end, col:col_end);
        
        % Check if this patch is the selected one
        if patch_counter == patch_final
            % Find the maximum intensity value and its position in the patch
            [patch_max_intensity, linear_idx] = max(patch(:));
            [patch_max_row, patch_max_col] = ind2sub(size(patch), linear_idx);
            
            % Calculate the position in the original image
            original_max_row = row + patch_max_row - 1;
            original_max_col = col + patch_max_col - 1;
            
            % Update the maximum intensity and position
            max_intensity = patch_max_intensity;
            max_position = [original_max_row, original_max_col];
        end
        
        % Increment the patch counter
        patch_counter = patch_counter + 1;
    end
end

end
