function [Y, numDimsY] = onnxPad(X, pads, value, mode, ONNXAxis, numDimsX)
% Implements the ONNX Pad operator

% ONNX 'pads' is a vector: [x1_begin, x2_begin...x1_end, x2_end,...], with
% x1,x2, listed in FORWARD ONNX dimension ordering, because it is data
% within a dimension and so is not flipped. xi_begin is the number of
% pixels added at the beginning of axis `i` and xi_end, the number of
% pixels added at the end of axis `i`.  pads can be negative, in which case
% that number of pixels is removed. 

% Copyright 2020-2024 The MathWorks, Inc.

pads = pads(:)';
numDimsY = numDimsX;
if ONNXAxis < 0
    ONNXAxis = ONNXAxis + numDimsX;
end
% Fill in pads to length 2*numDimsX if size(ONNXAxis,1) < numDimsX
if size(ONNXAxis,1) < numDimsX
    helpPads = dlarray(zeros(1,2*numDimsX));
    helpPads([ONNXAxis+1,ONNXAxis+numDimsX+1]) = pads;
    pads = helpPads;
end

if numDimsX==1
    % X is Nx1. Temporarily make it reverse-ONNX 2D (1xN), then transpose
    % the result back to 1D at the end.
    X = X';
    numDimsX = 2;
    pads = [pads(1) 0 pads(2) 0];  % Don't pad the dummy dimension
    numDimsY = 1;
end
sizeX  = size(X, 1:numDimsX);
fwdPadMat = reshape(extractdata(pads), [], 2)';  % row1 = begins, row2 = ends
% Columns of padmat are in reverse ONNX ordering. Still the case that row1
% = begins, row2 = ends:
padmat = fliplr(fwdPadMat);
sizeY  = sum([sizeX; padmat]);
% Create output tensor of the right size
Y = value*ones(sizeY, 'like', X);
% Construct  indices for inserting (and cropping) the original
for i=1:numel(sizeX)
    Ysubs{i} = max(1,1+padmat(1,i)) : min(sizeY(i), sizeY(i)-padmat(2,i));
    Xsubs{i} = max(1,1-padmat(1,i)) : min(sizeX(i), sizeX(i)+padmat(2,i));
end
% Insert/crop the original into the result
Y(Ysubs{:}) = X(Xsubs{:});
% Handle 'reflect', 'edge' and 'wrap' modes, but don't do it if X was 1D, 0x1.
if ismember(mode, ["edge", "reflect", "wrap"]) && ~(numDimsY==1 && sizeX(2)==0)
    for dim = 1:numDimsX
        if any(padmat(:,dim)>0)
            prepad  = padmat(1,dim);
            postpad = padmat(2,dim);
            if prepad > 0
                [Ysubs, Xsubs] = prepadIndices(sizeX, prepad, dim, mode);
                Y(Ysubs{:}) = Y(Xsubs{:});
            end
            if postpad > 0
                [Ysubs, Xsubs] = postpadIndices(sizeX, sizeY, prepad, postpad, dim, mode);
                Y(Ysubs{:}) = Y(Xsubs{:});
            end
        end
    end
end
% Transpose the result back to 1D if the input was 1D
if numDimsY==1
    Y = Y';
end

% Subfunctions in onnxPad:
    function [Ysub, Xsub] = prepadIndices(sizeX, prepad, dim, mode)
        Ysub	= repmat({':'}, [1 numel(sizeX)]);
        Xsub   	= Ysub;
        % Write into the first 'prepad' elements of Y.dim.
        Ysub{dim} = 1:prepad;
        switch mode
            case 'reflect'
                % Create indices 2:prepad+1 of X.dim, in the reverse order, with
                % wraparound. Then add prepad to convert them to Y indices.
                Xsub{dim} = wrapIndices(prepad+1 : -1 : 2, sizeX(dim)) + prepad;
            case 'edge'
                % Create replicated indices 1 of X.dim. Then add prepad to
                % convert them to Y indices.
                Xsub{dim} = ones(1, prepad) + prepad;
            case 'wrap'
                Xsub{dim} = flip(wrapIndices(sizeX(dim): -1 : sizeX(dim)-prepad+1, sizeX(dim))) + prepad;
            otherwise
                assert(false);
        end
    end

    function [Ysub, Xsub] = postpadIndices(sizeX, sizeY, prepad, postpad, dim, mode)
        Ysub	= repmat({':'}, [1 numel(sizeX)]);
        Xsub   	= Ysub;
        % Write into the last 'postpad' elements of Y.dim.
        Ysub{dim} = sizeY(dim)-postpad+1 : sizeY(dim);
        switch mode
            case 'reflect'
                % Create indices in the reverse order, with wraparound. Then add
                % prepad to convert them to Y indices.
                Xsub{dim} = wrapIndices(sizeX(dim)-1 : -1 : sizeX(dim)-postpad, sizeX(dim)) + prepad;
            case 'edge'
                % Create replicated end indices . Then add prepad to convert them
                % to Y indices.
                Xsub{dim} = repmat(sizeX(dim), [1 postpad]) + prepad;
            case 'wrap'
                Xsub{dim} = wrapIndices(1 : 1 : postpad, sizeX(dim)) + prepad;
            otherwise
                assert(false);
        end
    end

    function j = wrapIndices(i, maxIdx)
        % i can be positive, negative or zero. Legal output indices are in the
        % range 1:maxIdx.
        j = mod(i-1, maxIdx) + 1;
    end
end

