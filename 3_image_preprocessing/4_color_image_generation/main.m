clear
clc
X = imread('n07745940_12933.JPEG');
X_ori = im2double(X);
% imshow(X_ori)

%%
% X_1 = imscramble_amplitude(X_ori, 'cutoff');
% imshow(X_1)

Xscrambled = imscramble(X_ori,0.8,'cutoff');
imshow(Xscrambled)

imwrite(Xscrambled,'result.jpg')

