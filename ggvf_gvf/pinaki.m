clc;
clear all;
close all;

%% Get all images inside sample_images
root_dir = "sample_images";
output_dir = "ggvf_output";
image_list = dir(fullfile(root_dir, "*.jpg"));
for i = 1:numel(image_list)
	F = fullfile(root_dir, image_list(i).name);
	display(F);
	image = imread(F);
	I = rgb2gray(image);
	CE = edge(I, 'canny');

	%% GGVF implementation from paper
	time = 20;
	[u, v, original_image] = GGVF(CE, time);

	%% GVF implementation from paper
	% time = 20;
	% mu = 0.06;
	% [u, v] = GVF(CE, mu, time);

	clear F image I CE time original_image;

	save(fullfile(output_dir, image_list(i).name));
	display(size(u));
end
