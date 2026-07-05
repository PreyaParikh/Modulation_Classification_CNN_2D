function varargout = onnxSplit13(X, ONNXaxis, splits, numSplits, numDimsX)
% Implements the ONNX Split operator

% Copyright 2021-2024 The MathWorks, Inc.

% ONNXaxis is origin 0. splits is a vector of the lengths of each segment.
% If splits is empty, instead split into segments of equal length.
if ONNXaxis<0
    ONNXaxis = ONNXaxis + numDimsX;
end
DLTAxis = numDimsX - ONNXaxis;
if isempty(splits)
    C       = size(X, DLTAxis);
    sz      = floor(C/numSplits);
    splits	= repmat(sz, 1, numSplits);
else
    splits = extractdata(splits);
end
fullIndices = repmat({':'}, 1, numDimsX);     % Important to use numDimsX. ndims(X) may be too small.
splitIndices = [0 cumsum(splits(:)')];
numY = numel(splitIndices)-1;
for i = 1:numY
    from            = splitIndices(i) + 1;
    to              = splitIndices(i+1);
    fullIndices{DLTAxis}	= from:to;
    % The first numY outputs are the Y's. The second numY outputs are their
    % numDims. We assume all the outputs of Split have the same numDims as
    % the input.
    varargout{i}        = X(fullIndices{:});
    varargout{i + numY} = numDimsX;  
end
end
