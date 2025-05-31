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
