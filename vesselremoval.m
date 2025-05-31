function Clean_Image = vesselremoval(Test_Image)


Resized_Image = imresize(Test_Image, [512 512]);

Converted_Image = im2double(Resized_Image);
Lab_Image = rgb2lab(Converted_Image);
fill = cat(3, 0,1,0);
Filled_Image = bsxfun(@times, fill, Lab_Image);
Reshaped_Lab_Image = reshape(Filled_Image, [], 3);
[~, S] = pca(Reshaped_Lab_Image);
S = reshape(S, size(Lab_Image));
S = S(:, :, 1);
Gray_Image = (S-min(S(:)))./(max(S(:))-min(S(:)));
Enhanced_Image = adapthisteq(Gray_Image, 'numTiles', [8 8], 'nBins', 128);
Avg_Filter = fspecial('average', [10 10]);
Filtered_Image = imfilter(Enhanced_Image, Avg_Filter);
Subtracted_Image = imsubtract(Filtered_Image,Enhanced_Image);




Binary_Image = ~imbinarize(Subtracted_Image,"adaptive");


Clean_Image = bwareaopen(Binary_Image,80);

% 
% LO(:,:,1)=double(Resized_Image(:,:,1)).*double(~Clean_Image);
% LO(:,:,2)=double(Resized_Image(:,:,2)).*double(~Clean_Image);
% LO(:,:,3)=double(Resized_Image(:,:,3)).*double(~Clean_Image);
% L=LO;
% 
% 
% 
% KK=imresize(gambarclose,[size(Test_Image,1),size(Test_Image,2)]);

% 
% Kr=uint8(KK);






 








