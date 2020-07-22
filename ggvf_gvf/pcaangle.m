function angle = pcaangle(x, y, plotoption)
%PCAANGLE - Estimate main axis angle of a point cloud
%  
%   ANGLE = PCAANGLE(X, Y) estimates the angle of the main axis of
%   variation from the points (X,Y). If (X,Y) defines an oval point
%   cloud, the major axis of the ellipse is found.
%
%   ANGLE = PCAANGLE(..., 'visualize') plots the points and the direction
%   of the main axis.
%
%   PCAANGLE with no arguments starts a demo of the function.
%
%   If pcaangle is called with no arguments, the function enters
%   demonstration mode. 

%% AUTHOR    : Jøger Hansegård 
%% $DATE     : 06-Dec-2004 10:13:10 $ 
%% DEVELOPED : 7.0.1.24704 (R14) Service Pack 1 
%% FILENAME  : pcaangle.m 

%If no input is specified, let the user create some
if nargin==0
    [x, y] = createtestdata;
    plotoption = 'visualize';
end
if isempty(x)
   disp('No input specified');
    return
end
if numel(x)<2
    error('pcaangle needs at least two points to estimate the rotational angle');
end

%Remove the mean from the data
x = x(:)-mean(x(:));
y = y(:)-mean(y(:));

%Apply PCA to retreive major axis
c = cov(x, y);
[a, ev] = eig(c);
[ev,ind] = sort(diag(ev));

%Extract the angle of the major axis
[xa, ya] = deal(a(1,ind(end)), a(2,ind(end)));
angle = cart2pol(xa, ya);
if nargin == 3||nargin == 0
    if isequal(lower(plotoption), 'visualize')
        scatter(x,y);
        axis equal;
        hold on
        quiver(0,0, xa*norm(axis)/2, ya*norm(axis)/2, 'k', 'linewidth', 3);
        drawnow
    end
end

%If no input were specified, give graphical response
if nargin==0
    mh = msgbox(sprintf('Estimated angle: %f', angle));
    waitfor(mh);
    return;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   Create testdata. Only required for demo mode %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [x,y] = createtestdata
x = 4*randn(1000,1);
y = randn(1000,1);

fh = figure;
scatter(x,y);
title('Some testdata');
axis equal
response = inputdlg(sprintf('Please specify a rotation angle in radians.\nExample: ''pi/3'''),...
    'User input', 1, {'pi/3'});
if isempty(response)
    he = errordlg('Operation cancelled by user', 'Cancel');
    waitfor(he);
    close(fh);
    error('Operation cancelled by user');
end
if isempty(response{1})
    he = errordlg('No input specified. Aborting.', 'No input error');
    waitfor(he);
    close(fh);
    error('No input specified. Aborting.');
end
try
    alpha = eval(response{1});
catch
    he = errordlg(sprintf('Unable to interpret expression\n %s', lasterr), 'Invalid input');
    waitfor (he)
    close(fh);
    error('Unable to interpret expression\n %s', lasterr);
end
%Rotate the data
rotmat = [cos(-alpha), sin(-alpha); -sin(-alpha), cos(-alpha)];
rotated = rotmat*[x y]';
[x, y] = deal(rotated(1,:)', rotated(2,:)');
clf, scatter(x,y);
title(sprintf('Rotated testdata by angle %f', alpha));
axis equal

% Created with NEWFCN.m by Jøger Hansegård  
% Contact...: jogerh@ifi.uio.no  
% $Log: pcaangle.m,v $
% 
% ===== EOF ====== [pcaangle.m] ======  
