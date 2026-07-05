function [Y_, numDimsY] = onnxPad(X_, pads_, value_, mode_, ONNXAxis_, numDimsX_)
% Implements the ONNX Pad operator

% ONNX 'pads' is a vector: [x1_begin, x2_begin...x1_end, x2_end,...], with
% x1,x2, listed in FORWARD ONNX dimension ordering, because it is data
% within a dimension and so is not flipped. xi_begin is the number of
% pixels added at the beginning of axis `i` and xi_end, the number of
% pixels added at the end of axis `i`.  pads can be negative, in which case
% that number of pixels is removed.
%#codegen

% Copyright 2024 The MathWorks, Inc.

X__   = modclass_vbx.coder.ops.extractIfDlarray(X_);
pads__   = modclass_vbx.coder.ops.extractIfDlarray(pads_);
value   = modclass_vbx.coder.ops.extractIfDlarray(value_);
mode   = modclass_vbx.coder.ops.extractIfDlarray(mode_);
ONNXAxis   = modclass_vbx.coder.ops.extractIfDlarray(ONNXAxis_);
numDimsX   = modclass_vbx.coder.ops.extractIfDlarray(numDimsX_);

pads___ = pads__(:)';
numDimsY = numDimsX;
if ONNXAxis < 0
    ONNXAxis = ONNXAxis + numDimsX;
end

% Fill in pads to length 2*numDimsX if size(ONNXAxis,1) < numDimsX
if size(ONNXAxis,1) < numDimsX
    helpPads = zeros(1,2*numDimsX);
    helpPads([ONNXAxis+1,ONNXAxis+numDimsX+1]) = pads___;
    pads____ = helpPads;
else
    pads____ = pads___;
end

if numDimsX==1
    % X is Nx1. Temporarily make it reverse-ONNX 2D (1xN), then transpose
    % the result back to 1D at the end.
    X = X__';
    numDimsX = 2;
    pads = [pads____(1) 0 pads____(2) 0];  % Don't pad the dummy dimension
    numDimsY = 1;
else
    X = X__;
    pads = pads____;
end

sizeX  = size(X, 1:numDimsX);
fwdPadMat = reshape(pads, [], 2)';  % row1 = begins, row2 = ends
% Columns of padmat are in reverse ONNX ordering. Still the case that row1
% = begins, row2 = ends:
padmat = fliplr(fwdPadMat);
sizeY  = sum([sizeX; padmat]);
% Create output tensor of the right size
Y = value*ones(sizeY, 'like', X);
% Construct  indices for inserting (and cropping) the original
Ysubs = cell(1, numel(sizeX));
Xsubs = cell(1, numel(sizeX));
coder.unroll();
for i=1:numel(sizeX)
    Ysubs{i} = max(1,1+padmat(1,i)) : min(sizeY(i), sizeY(i)-padmat(2,i));
    Xsubs{i} = max(1,1-padmat(1,i)) : min(sizeX(i), sizeX(i)+padmat(2,i));
end
% Insert/crop the original into the result
Y(Ysubs{:}) = X(Xsubs{:});

% Handle 'reflect', 'edge' and 'wrap' modes, but don't do it if X was 1D, 0x1.
if (strcmp(mode, 'edge') || strcmp(mode,'reflect') || strcmp(mode, 'wrap')) && ~(numDimsY==1 && sizeX(2)==0)
    coder.unroll();
    for dim = 1:numDimsX
        if any(padmat(:,dim)>0)
            prepad  = padmat(1,dim);
            postpad = padmat(2,dim);
            if prepad > 0
                [YsubsPre, XsubsPre] = prepadIndices(sizeX, sizeY, prepad, dim, mode);
                Y(YsubsPre{:}) = Y(XsubsPre{:});
            end
            if postpad > 0
                [YsubsPost, XsubsPost] = postpadIndices(sizeX, sizeY, prepad, postpad, dim, mode);
                Y(YsubsPost{:}) = Y(XsubsPost{:});
            end
        end
    end
end

% Transpose the result back to 1D if the input was 1D
if numDimsY==1
    Y_ = Y';
else
    Y_ = Y;
end

% Subfunctions in onnxPad:
    function [Ysub, Xsub] = prepadIndices(sizeX, sizeY, prepad, dim, mode)
        Xsub = cell(1, numel(sizeX));
        Ysub = cell(1, numel(sizeX));
        coder.unroll();
        for j = 1:numel(sizeX)
            if j==dim
                switch mode
                    case 'reflect'
                        % Create indices 2:prepad+1 of X.dim, in the reverse order, with
                        % wraparound. Then add prepad to convert them to Y indices.
                        Xsub{j} = wrapIndices(prepad+1 : -1 : 2, sizeX(j)) + prepad;
                    case 'edge'
                        % Create replicated indices 1 of X.dim. Then add prepad to
                        % convert them to Y indices.
                        Xsub{j} = ones(1, prepad) + prepad;
                    case 'wrap'
                        Xsub{j} = flip(wrapIndices(sizeX(j): -1 : sizeX(j)-prepad+1, sizeX(j))) + prepad;
                end
                % Write into the first 'prepad' elements of Y.dim.
                Ysub{j} = 1:prepad;
            else
                Xsub{j}	    = 1 : sizeY(j);
                Ysub{j}	    = 1 : sizeY(j);
            end
        end   
    end

    function [Ysub, Xsub] = postpadIndices(sizeX, sizeY, prepad, postpad, dim, mode)
        Xsub = cell(1, numel(sizeX));
        Ysub = cell(1, numel(sizeX));        
        coder.unroll();
        for j = 1:numel(sizeX)
			if j ~= dim
                Xsub{j}	= 1 : sizeY(j);
				Ysub{j}	= 1 : sizeY(j);
            else
                switch mode
                    case 'reflect'
                        % Create indices in the reverse order, with wraparound. Then add
                        % prepad to convert them to Y indices.
                        Xsub{j} = wrapIndices(sizeX(j)-1 : -1 : sizeX(j)-postpad, sizeX(j)) + prepad;
                    case 'edge'
                        % Create replicated end indices . Then add prepad to convert them
                        % to Y indices.
                        Xsub{j} = repmat(sizeX(j), [1 postpad]) + prepad;
                    case 'wrap'
                        Xsub{j} = wrapIndices(1 : 1 : postpad, sizeX(j)) + prepad;
                end
                % Write into the last 'postpad' elements of Y.dim.
                Ysub{j} = sizeY(j)-postpad+1 : sizeY(j);
			end
        end 
    end

    function j = wrapIndices(idx, maxIdx)
        % idx can be positive, negative or zero. Legal output indices are in the
        % range 1:maxIdx.
        j = mod(idx-1, maxIdx) + 1;
    end
end