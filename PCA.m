% size
row=112; col=92; P = 48;  % 112x92 image x 48
% timer start
tic;
% read Training image
imgbuf_Tr = zeros(row,col,P);
sum_Tr = zeros(row,col,'double');
for i = 1:1:6  % P개의 training vector를 읽음
    for j = 1:1:8
        img_path_Tr = "./Training/set_" + i + "/" + j + ".bmp";
        img_tmp = imread(img_path_Tr);
        imgbuf_Tr(:,:,8*(i-1)+j) = img_tmp(:,:,1);  % imgbuf_Tr = 모든 data 읽음 set1 부터 set6까지
        sum_Tr = sum_Tr + cast(img_tmp(:,:,1), 'double');       % sum_Tr = 합
    end
end
Average_Tr = sum_Tr / P;      % training vector의 평균을 구함
% normalize, x - m = x': 112x92x48 normalized vector를 구함
normalized_Tr = zeros(row,col,P);
for i = 1:1:P
    normalized_Tr(:,:,i) = imgbuf_Tr(:,:,i) - Average_Tr;
end
% 2D->1D flatten NxP, P = 48, N(row*col) = 10304
np_Tr = zeros(row*col,P); % 10304x48 size
for i = 1:1:P
    np_Tr(:,i) = reshape(normalized_Tr(:,:,i), [row*col,1]);
end
% covariance matrix : x*xt
cov_Tr = (np_Tr * np_Tr.')/(row*col);  % 10304x10304
% compute eigen value - 48개중 eigenvalue 크기 상위 10개와 그 eigenvector
[e_vector, e_value] = eigs(cov_Tr, 10);  % 10304x10, 10x10 diagonal
% projection - representative vector
pj_Tr = zeros(10,P);
for i = 1:1:P
    pj_Tr(:,i) = e_vector.' * np_Tr(:,i);  % 10x48
end
% Read Test Image
imgbuf_Te = zeros(row,col,12);
for i = 1:1:12
    img_path_Te = "./Test/" + i + ".bmp";
    img_tmp = imread(img_path_Te);
    imgbuf_Te(:,:,i) = img_tmp(:,:,1);
end
% normalize Test Image
normalized_Te = zeros(row,col,12);
for i = 1:1:12
    normalized_Te(:,:,i) = imgbuf_Te(:,:,i) - Average_Tr;
end
% flatten Test Image
np_Te = zeros(row*col,12);
for i = 1:1:12
    np_Te(:,i) = reshape(normalized_Te(:,:,i), [row*col,1]);
end
% mapping
pj_Te = zeros(10,12);
for i = 1:1:12
    pj_Te(:,i) = e_vector.' * np_Te(:,i);  % 10x12
end

% Clustering
% cl_Tr = zeros(10, 6);   %10x6
% for i = 1:6
%     k = (i-1)*6 + 1;
%     cl_Tr(:,i) = pj_Tr(:,k) + pj_Tr(:,k+1) + pj_Tr(:,k+2) + pj_Tr(:,k+3) + pj_Tr(:,k+4) + pj_Tr(:,k+5);
%     cl_Tr(:,i) = cl_Tr(:,i)/6;
% end

% Euclidean Distance
ed = zeros(10, 48);
for i = 1:1:12
    for j = 1:1:48
        dist = pj_Te(:,i) - pj_Tr(:,j);
        ed(i, j) = sum(dist.*dist, 'all');  % element-wise square
    end
end
% classification
[result, index] = min(ed, [], 2);
% 정답은 2,2,1,1,3,3,4,4,5,5,6,6이 나와야 함
% print result
answer = fix((index - 0.1)/8) + 1;
% stop timer
toc;
disp("test data 1~12 label result: ");
disp(answer);


