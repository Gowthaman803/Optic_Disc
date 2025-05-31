function I_3=preprocess(I_4)
I = imresize(I_4, [600 600]);
BlueChannel=double(I(:,:,3));
Var_Blue=var(BlueChannel,1,"all");
L=rgb2lab(I);
LChannel=L(:,:,1);
aChannel=L(:,:,2);
bChannel=L(:,:,3);
Variance_threshold=1500;
if Var_Blue <= Variance_threshold
a_Cl=adapthisteq(aChannel,'NumTiles',[10 10],'ClipLimit',0.02);
[r,c]=size(LChannel);
Img=zeros(r,c,3);
Img(:,:,1)=LChannel;
Img(:,:,2)=a_Cl;
Img(:,:,3)=bChannel;
else
b_Cl=adapthisteq(bChannel,'NumTiles',[10 10],'ClipLimit',0.02);
[r,c]=size(LChannel);
Img=zeros(r,c,3);
Img(:,:,1)=LChannel;
Img(:,:,2)=aChannel;
Img(:,:,3)=b_Cl;
end
I_1=lab2rgb(Img);

Green_channel_Updated=I_1(:,:,2);
G_Cl=adapthisteq(Green_channel_Updated,'NumTiles',[12 12],'ClipLimit',0.02);
I_1(:,:,2)=G_Cl;
I_2 = imbilatfilt(I_1);
I_2gray=im2gray((I_2));
mini=min(min(I_2gray));
maxi=max(I_2gray,[],"all");
alpha=255/(maxi-mini);
beta=-mini*alpha;

I_3=uint8(alpha*I_2+beta);
end
