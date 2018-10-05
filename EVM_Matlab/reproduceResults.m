% Generates all the results for the SIGGRAPH paper at:
% http://people.csail.mit.edu/mrub/vidmag
%
% Copyright (c) 2011-2012 Massachusetts Institute of Technology, 
% Quanta Research Cambridge, Inc.
%
% Authors: Hao-yu Wu, Michael Rubinstein, Eugene Shih
% License: Please refer to the LICENCE file
% Date: June 2012
%

clear;

dataDir = './data';
resultsDir = 'ResultsSIGGRAPH2012';

mkdir(resultsDir);


%% baby
inFile = fullfile(dataDir,'shake.mp4');
fprintf('Processing %s\n', inFile);
amplify_spatial_lpyr_temporal_iir(inFile, resultsDir, 10, 16, 3.6,6.2, 0.1);

% Alternative processing using butterworth filter
% amplify_spatial_lpyr_temporal_butter(inFile, resultsDir, 30, 16, 0.4, 3, 30, 0.1);

%% baby2
inFile = fullfile(dataDir,'shake.mp4');
fprintf('Processing %s\n', inFile);
amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,60, 90, 3.6, 6.2, 30, 0.3);

%% camera
inFile = fullfile(dataDir,'iron.mp4');
fprintf('Processing %s\n', inFile);
amplify_spatial_lpyr_temporal_butter(inFile, resultsDir,  50, 10, 5, 6, 100, 0.3);
%% subway
inFile = fullfile(dataDir,'shake.mp4');
fprintf('Processing %s\n', inFile);
amplify_spatial_lpyr_temporal_butter(inFile, resultsDir, 60, 90, 3.6, 6.2, 30, 0.3);

%% wrist
%% No mask is used here to generate the output video.
inFile = fullfile(dataDir,'iron.mp4');
fprintf('Processing %s\n', inFile);
amplify_spatial_lpyr_temporal_iir(inFile, resultsDir, 50, 16, 5, 6, 0.1);

% Alternative processing using butterworth filter
% amplify_spatial_lpyr_temporal_butter(inFile, resultsDir, 30, 16, 0.4, 3, 30, 0.1);


%% shadow
inFile = fullfile(dataDir,'iron.mp4');
fprintf('Processing %s\n', inFile);
amplify_spatial_lpyr_temporal_butter(inFile, resultsDir, 50, 10, 5, 60, 100, 1);

%% guitar
inFile = fullfile(dataDir,'iron.mp4');
fprintf('Processing %s\n', inFile);
% amplify E
amplify_spatial_lpyr_temporal_ideal(inFile, resultsDir, 50, 10, 5, 7, 100, 0);
% amplify A
amplify_spatial_lpyr_temporal_ideal(inFile, resultsDir, 100, 10, 100, 120, 600, 0);


%% face
inFile = fullfile(dataDir,'angry2.mp4');
fprintf('Processing %s\n', inFile);
amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,60,6, ...
                     50/60,60/60,30, 1);


%% face2
inFile = fullfile(dataDir,'hand_outdoor.mp4');
fprintf('Processing %s\n', inFile);

% Motion
amplify_spatial_lpyr_temporal_butter(inFile,resultsDir,20,80, ...
                                     0.5,10,30, 0);
% Color
amplify_spatial_Gdown_temporal_ideal(inFile,resultsDir,50,6, ...
                                     50/60,60/60,30, 1);
