function scaled_point = scale_point_to_original(point, target_size, original_size)
    scaled_point = zeros(1, 2);
    scaled_point(1) = floor(((point(2) / target_size(2)) * original_size(2)));
    scaled_point(2) = floor(((point(1) / target_size(1)) * original_size(1)));
end