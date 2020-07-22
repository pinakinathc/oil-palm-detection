% clear workspace
clc;
clear all;
close all;

% get input image
input_image = (imread('palm-1.jpg'));
I = rgb2gray(input_image);
CE = edge(I,'canny');

%% GGVF implementation from paper

% time = 20;
% [u,v,original_image] = GGVF(CE, time);
%
% % It must be normalized
% mag = sqrt(u.*u+v.*v);
% px = u./(mag+1e-10); py = v./(mag+1e-10);
%
% figure(1);
% colormap(gray);imagesc(original_image)
% hold on
% quiver(px, py)
% axis image
% title('GGVF Vectors & Contour Lines')

%% GVF implementation

time = 20;
mu = 0.06;

[u,v] = GVF(CE, mu, time);

% It must be normalized
mag = sqrt(u.*u+v.*v);
px = u./(mag+1e-10); py = v./(mag+1e-10);

%% dominance matrix extraction

[row, col] = size(CE);
zerosMatrix = zeros(row, col);

for i = 1:row-2
    for j = 1:col-2
        if CE(i+1, j+1) == 0
            a11 = atan2(px(i, j), py(i, j)) * (180 / pi);
            a12 = atan2(px(i, j+1), py(i, j+1)) * (180 / pi);
            a13 = atan2(px(i, j+2), py(i, j+2)) * (180 / pi);
            a21 = atan2(px(i+1, j), py(i+1, j)) * (180 / pi);
            a23 = atan2(px(i+1, j+1), py(i+1, j+1)) * (180 / pi);
            a31 = atan2(px(i+2,j), py(i+2, j)) * (180 / pi);
            a32 = atan2(px(i+2, j+1), py(i+2, j+1)) * (180 / pi);
            a33 = atan2(px(i+2, j+2), py(i+2, j+2)) * (180 / pi);
            
            if (abs(a11 - a33) >= 175) && (abs(a11 - a33) <= 185)
                zerosMatrix(i+1, j+1) = 1;
            elseif (abs(a12 - a32) >= 175) && (abs(a12 - a32) <= 185)
                zerosMatrix(i+1, j+1) = 1;
            elseif (abs(a13 - a31) >= 175) && (abs(a13 - a31) <= 185)
                zerosMatrix(i+1, j+1) = 1;
            elseif (abs(a21 - a23) >= 175) && (abs(a21 - a23) <= 185)
                zerosMatrix(i+1,j+1) = 1;
            end
        end
    end
end

BW = zerosMatrix;
% imshow(BW)


%% % remove edge components with aspect ratio (width / length) between 0.7 to 1.3

[row, col] = size(CE);
zerosMatrix = zeros(row, col);

b = [];
% remove pixels group less than 10 pixels
BW = bwareaopen(BW, 10);

% compute connected components label
CC = bwconncomp(BW);
st = regionprops(CC, 'BoundingBox', 'PixelList');

% draw bounding boxes
for k = 1 : length(st)
    BB = st(k).BoundingBox;
    xy = st(k).PixelList;
    %     xy = cat(1,st(k).PixelList);
    
    ratio = BB(3) / BB(4);
        b = [b; ratio]
    if ratio <= 0.7 || ratio >= 1.3
        for list = 1:length(xy)
            x = xy(list,2);
            y = xy(list,1);
            zerosMatrix(x, y) = 1;
        end
    end
end

BW = zerosMatrix;
% imshow(BW)

%% compare the centroid for the edge component lies on any edge componets pixel

% [row, col] = size(CE);
% zerosMatrix = zeros(row, col);
% 
% % remove pixels group less than 10 pixels
% BW = bwareaopen(BW, 10);
% % BW4 = BW3;
% % imshow(input_image)
% % imshow(BW4)
% % hold on;
% 
% % compute connected components label
% CC = bwconncomp(BW);
% st = regionprops(CC, 'Centroid', 'PixelList');
% 
% % check condition if centroid edge component match with white pixel of the
% % edge component
% for k=1:length(st) % k = field column
%     excentroid = round(st(k).Centroid(1));
%     eycentroid = round(st(k).Centroid(2));
%     xy = st(k).PixelList;
%     
%     %     xy = cat(1,st(k).PixelList);
%     
%     if BW(eycentroid, excentroid) == 1
%         for list = 1:length(xy) % list = number of pixels inside one field
%             if list < length(xy)
%                 x = xy(list,2);
%                 y = xy(list,1);
%                 zerosMatrix(x, y) = 1;
%             end
%         end
%     end
% end
% 
% BW = zerosMatrix;
% imshow(BW)

% hold off


%% compute pca angle and show only diagonal components - remove unrelated pixels

[row, col] = size(CE);
zerosMatrix = zeros(row, col);

% compute connected components label
CC = bwconncomp(BW);
st = regionprops(CC, 'Centroid', 'PixelList');

% compute pca angle & select diagonal components
for k=1:length(st)
    xy = st(k).PixelList;
    
        xa = xy(1:length(xy),2);
        ya = xy(1:length(xy),1);
    theta = (pcaangle(xa, ya)) * 180 / pi;
    
    if theta < 0
        theta = theta + 180;
    end
    
    if theta >= 30 && theta <= 75
        for list = 1:length(xy)
                x1 = xy(list,2);
                y1 = xy(list,1);
                zerosMatrix(x1, y1) = 1;
        end
        
    elseif theta >= 105 && theta <= 145
        for list = 1:length(xy)
                x1 = xy(list,2);
                y1 = xy(list,1);
                zerosMatrix(x1, y1) = 1;
        end
    end
end

BW = zerosMatrix;

%% compute intersection points for each edge components ( note: x & y flipping)

% imshow(input_image)
imshow(BW)
hold on

CC2 = bwconncomp(BW);
st = regionprops( CC2, 'Centroid', 'PixelList');

% compute image center coordinate
[row, col] = size(CE);
cex = length(BW(1:row/2));
cey = length(BW(1:col/2));

radiusOtr = 0.8*cex;
radiusInr = 20;

L = 20;

for k=1:length(st)-1
    
    xy = st(k).PixelList;
    
    xa = xy(1:length(xy),1);
    ya = xy(1:length(xy),2);
    
    theta1 = (pcaangle(xa, ya)) * 180 / pi;
    
    if theta1 < 0
        theta1 = theta1 + 180;
    end
    
    % compute first pair points from end of edge components
    x1 = xy(length(xy),1);
    y1 = xy(length(xy),2);

    x2 = x1 + (L * cosd(theta1));
    y2 = y1 + (L * sind(theta1));
    
    for l=k+1:length(st)
        xy2 = st(l).PixelList;
        
        xb = xy2(1:length(xy2),1);
        yb = xy2(1:length(xy2),2);
        
        theta2 = (pcaangle(xb, yb)) * 180 / pi;
        
        if theta2 < 0
            theta2 = theta2 + 180;
        end
        
        % compute next pair points from end of edge components
        x3 = xy2(length(xy2),1);
        y3 = xy2(length(xy2),2);
        
        x4 = x3 + (L * cosd(theta2)); 
        y4 = y3 + (L * sind(theta2)); 
        
        if st(k).Centroid ~= st(l).Centroid

            % compute intersection point
            x = [x1 x3; x2 x4];
            y = [y1 y3; y2 y4];
            
            % Take the differences down each column
            dx = diff(x);
            dy = diff(y);
            
            % Precompute the denominator
            den = dx(1)*dy(2) - dy(1)*dx(2);
            
            ua = (dx(2)*(y(1) - y(3)) - dy(2)*(x(1) - x(3))) / den;
            ub = (dx(1)*(y(1) - y(3)) - dy(1)*(x(1) - x(3))) / den;
            
            xi = x(1) + ua*dx(1);
            yi = y(1) + ua*dy(1);
            
            % image center coordinate
            cdx = [xi, cex];
            cdy = [yi, cey];
            
            % n edge component coordinate
            cfx = [x1, cex];
            cfy = [y1, cey];
            
            % n+1 edge component coordinate
            cgx = [x3, cex];
            cgy = [y3, cey];
            
            % distance from intersection point to image center
            distInr = sum(sqrt(diff(cdx).^2+diff(cdy).^2));
            
            % distance from 'n edge component' to image center
            distOtr1 = sum(sqrt(diff(cfx).^2+diff(cfy).^2));
            
            % distance from 'n+1 edge component' to image center
            distOtr2 = sum(sqrt(diff(cgx).^2+diff(cgy).^2));
     
            if distInr <= radiusInr && distOtr1 <= radiusOtr && distOtr2 <= radiusOtr
                
                % draw line
                line([x1,xi], [y1,yi], 'LineStyle', ':', 'Color', 'g', 'LineWidth', 1);
                line([x3,xi], [y3,yi], 'LineStyle', ':', 'Color', 'g', 'LineWidth', 1);
                
                % draw intersection point
                plot(xi,yi,'o', 'Color', 'm', 'LineWidth', 1);
                
            end
        end
    end
end

% draw image center circle inner & outer
theta = 0:pi/50:2*pi;
ccx = radiusInr * cos(theta) + cex;
ccy = radiusInr * sin(theta) + cey;

cbx = radiusOtr * cos(theta) + cex;
cby = radiusOtr * sin(theta) + cey;

plot(ccx, ccy, 'LineStyle', ':', 'Color', 'c', 'LineWidth',1);
plot(cbx, cby, 'LineStyle', ':', 'Color', 'y', 'LineWidth',1);

hold off
