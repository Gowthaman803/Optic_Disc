function image_out=vessel_seg(I)
mask = imbinarize(I,40/255);
se = strel('diamond',20);              
erodedmask = im2uint8(imerode(mask,se));   
lab = rgb2lab(im2double(I));
lab(:,:,2)=0;lab(:,:,3)=0;
wlab = reshape(lab,[],3);
[C,S] = pca(wlab); 
S = reshape(S,size(lab));
S = S(:,:,1);
gray = (S-min(S(:)))./(max(S(:))-min(S(:)));
J = adapthisteq(gray,'numTiles',[8 8],'nBins',256); 
h = fspecial('average', [11 11]);
JF = imfilter(J, h);
Z = imsubtract(JF, J);
level = graythresh(Z);
BW = imbinarize(Z, level-0.008);
BW2 = bwareaopen(BW, 50);
image_out=BW2.*(erodedmask==255);
image_out=im2gray(image_out);
end