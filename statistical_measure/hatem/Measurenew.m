%THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
%AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
%IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
%DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
%FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
%DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
%SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
%CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
%OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
%OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

% Nil Goyette
% University of Sherbrooke
% Sherbrooke, Quebec, Canada. April 2012

function [Mean_Eva , Eva, TP, FP, FN, TN, stats]  = Measurenew(GTPath, ResultsFolder)     
    groundtruthFolder = fullfile(GTPath);
    binaryFolder = fullfile(ResultsFolder);
    [Mean_Eva , Eva, TP, FP, FN, TN, stats] = compareImageFiles(groundtruthFolder, binaryFolder, 1, 100);
%     f = fopen(['2016_384_results.txt'], 'wt');
    fprintf(' Acc: %u DSC: %u JI: %u Sen: %u Spe: %u', stats(1), stats(2), stats(3), stats(4));
%     fclose(f);
end

function [Mean_Eva, Eva, TP, FP, FN, TN, stats] = compareImageFiles(gtFolder, binaryFolder, idxFrom, ~)
    % Compare the binary files with the groundtruth files.   
    extension = '.jpg'; % TODO Change extension if required
    threshold = strcmp(extension, '.jpg') == 1 || strcmp(extension, '.jpeg') == 1;
    listofGenerated = dir([binaryFolder '/*.jpg']);
    listofGT = dir([gtFolder '/*.png']);
    imBinary = imread(fullfile(binaryFolder, listofGenerated(1).name));
    int8trap = isa(imBinary, 'uint8') & min(min(imBinary)) == 0 & max(max(imBinary)) == 1;
    
    confusionMatrix = [0 0 0 0]; % TP FP FN TN 
    idxTo = length(listofGenerated);
    Eva = [];
    for idx = idxFrom:idxTo
%         fileName = num2str(idx, '%.3d');
        imBinary = imread(fullfile(binaryFolder, listofGenerated(idx).name));
        if size(imBinary, 3) > 1
            imBinary = rgb2gray(imBinary);
        end
        if islogical(imBinary) | int8trap
            imBinary = uint8(imBinary)*255;
        end
        if threshold
            imBinary = im2bw(imBinary, 0.5);
            imBinary = im2uint8(imBinary);
        end
        imGT = 255*(imread(fullfile(gtFolder, listofGT(idx).name)));
        [h,w,~] = size(imGT);
        imBinary = imfill(imBinary, 'holes');
%         se1 = strel('disk',2);
%         se2 = strel('disk',1);
%         imBinary = imerode(imBinary, se1);
%         imBinary = imdilate(imBinary, se2);
        imBinary = imresize(imBinary, [h,w],'bicubic');
        confusionMatrix = compare(imBinary, imGT);
        [TP, FP, FN, TN, stats] = confusionMatrixToVar(confusionMatrix);
        Eva = [Eva;stats];
    end
    %[TP, FP, FN, TN, stats] = confusionMatrixToVar(confusionMatrix);
 Mean_Eva = mean(Eva);
end

function confusionMatrix = compare(imBinary, imGT)
    % Compares a binary frames with the groundtruth frame
    TP = sum(sum(imGT==255&imBinary==255));		% True Positive 
    TN = sum(sum(imGT<=20&imBinary==0));		% True Negative
    FP = sum(sum((imGT<=20)&imBinary==255));	% False Positive
    FN = sum(sum(imGT==255&imBinary==0));		% False Negative
    confusionMatrix = [TP FP FN TN];
    
end

function [TP, FP, FN, TN, stats] = confusionMatrixToVar(confusionMatrix)
    TP = confusionMatrix(1);
    FP = confusionMatrix(2);
    FN = confusionMatrix(3);
    TN = confusionMatrix(4);

  
    %%%% the benchmark evaluation metrics from ISIB 2016 and 2017 challanges
    Accuracy= (TP+TN)/(TP+TN+FP+FN);
%     Accuracy_mean = mean(Accuracy);
    DSC= 2*TP./(TP+TP+FN+FP);
%     DSC = mean(DSC);
    JI= TP./(TP+FP+FN);
    JI = mean(JI);
    Sensitivity= TP./(TP+FN);
%     Sensitiv
ity = Sensitivity;
    Specificity= TN./(TN+FP);
%     Specificity = Specificity;
%     precision= TP./(TP+FP+0.00001);
%     FalsePositiveRate = 1-Specificity;
    stats = [Accuracy, DSC, JI, Sensitivity, Specificity];
    
   
end

