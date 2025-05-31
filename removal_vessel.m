
function inpainted_image = removal_vessel(I)
BW2=vessel_seg(I);
%% Remove small pixels

%% Overlay
BW2 = imcomplement(BW2);
% Create an artificial mask for missing regions
mask=imresize(BW2,[512,512]);
I=imresize((I),[512,512]);
damaged_img = I;
% ---------- Image initialization --------------------------------
% mask=double(vesselremoval(I));
im=double(damaged_img);
inpainted_image(:,:,1) = PDE_inpainting(im(:,:,1),mask);
inpainted_image(:,:,2) = PDE_inpainting(im(:,:,2),mask);
inpainted_image(:,:,3) = PDE_inpainting(im(:,:,3),mask);
inpainted_image=uint8(inpainted_image);
end





