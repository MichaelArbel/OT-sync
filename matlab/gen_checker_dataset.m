close all;

for i = 1:5
    imageFileName = sprintf('image%d.tif', i);
    imageFileNames{i} = fullfile(matlabroot,'toolbox','vision',...
        'visiondata','calibration','webcam',imageFileName);
end

[imagePoints,boardSize,imagesUsed] = detectCheckerboardPoints(imageFileNames);

imageFileNames = imageFileNames(imagesUsed);
numImages = numel(imageFileNames);
for i = 1:numImages
    I = imread(imageFileNames{i});
    subplot(2, 2, i);
    imshow(I);
    hold on;
    plot(imagePoints(:,1,i),imagePoints(:,2,i),'ro');
end

for i = 1:numel(imageFileNames)
    I = imread(imageFileNames{i});
    subplot(2, 2, i);
    imshow(I);
    hold on;
    plot(imagePoints(:,1,i),imagePoints(:,2,i),'ro');
end

for i=1:numImages
    for j=1:numImages
        if (i==j)
            continue;
        end
        
        imagePoints1 = imagePoints(:, :, i);
        imagePoints2 = imagePoints(:, :, j);
        
        I1 = imread(imageFileNames{i});
        I2 = imread(imageFileNames{j});
        
        imagePoints1 = detectSURFFeatures(I1);
        imagePoints2 = detectSURFFeatures(I2);
        features1 = extractFeatures(I1,imagePoints1,'Upright',false);
        features2 = extractFeatures(I2,imagePoints2,'Upright',false);
    
        indexPairs = matchFeatures(features1,features2);
        matchedPoints1 = imagePoints1(indexPairs(:,1),:);
        matchedPoints2 = imagePoints2(indexPairs(:,2),:);
        
        [relativeOrient, relativeLoc, inlierIdx] = helperEstimateRelativePose(...
        matchedPoints1, matchedPoints2, intrinsics);
        
        % Visualize correspondences
        figure
        showMatchedFeatures(I1, I2, matchedPoints1, matchedPoints2);
        title('Tracked Features');
    end
end



% Detect feature points
imagePoints1 = detectMinEigenFeatures(rgb2gray(I1), 'MinQuality', 0.1);

% Visualize detected points
figure
imshow(I1, 'InitialMagnification', 50);
title('150 Strongest Corners from the First Image');
hold on
plot(selectStrongest(imagePoints1, 150));



