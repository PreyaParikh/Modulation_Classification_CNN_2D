classdef Shape_To_PadLayer1003 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        CNN4_Split_split
        CNN4_Pad_value
    end

    properties (State)
    end

    properties
        Vars
        NumDims
    end

    methods(Static, Hidden)
        % Specify the properties of the class that will not be modified
        % after the first assignment.
        function p = matlabCodegenNontunableProperties(~)
            p = {
                % Constants, i.e., Vars, NumDims and all learnables and states
                'Vars'
                'NumDims'
                'CNN4_Split_split'
                'CNN4_Pad_value'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = modclass_vbx.coder.Shape_To_PadLayer1003(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = modclass_vbx.Shape_To_PadLayer1003(cgInstance.Name);
            if isstruct(cgInstance.Vars)
                names = fieldnames(cgInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this_ml.Vars.(fieldname) = dlarray(cgInstance.Vars.(fieldname));
                end
            else
                this_ml.Vars = [];
            end
            this_ml.NumDims = cgInstance.NumDims;
            this_ml.CNN4_Split_split = cgInstance.CNN4_Split_split;
            this_ml.CNN4_Pad_value = cgInstance.CNN4_Pad_value;
        end
    end

    methods
        function this = Shape_To_PadLayer1003(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'CNN4_Pad'};
            if isstruct(mlInstance.Vars)
                names = fieldnames(mlInstance.Vars);
                for i=1:numel(names)
                    fieldname = names{i};
                    this.Vars.(fieldname) = modclass_vbx.coder.ops.extractIfDlarray(mlInstance.Vars.(fieldname));
                end
            else
                this.Vars = [];
            end

            this.NumDims = mlInstance.NumDims;
            this.CNN4_Split_split = mlInstance.CNN4_Split_split;
            this.CNN4_Pad_value = mlInstance.CNN4_Pad_value;
        end

        function [CNN4_Pad] = predict(this, MaxPool3__)
            if isdlarray(MaxPool3__)
                MaxPool3_ = stripdims(MaxPool3__);
            else
                MaxPool3_ = MaxPool3__;
            end
            MaxPool3NumDims = 4;
            MaxPool3 = modclass_vbx.coder.ops.permuteInputVar(MaxPool3_, [4 3 1 2], 4);

            [CNN4_Pad__, CNN4_PadNumDims__] = Shape_To_PadGraph1006(this, MaxPool3, MaxPool3NumDims, false);
            CNN4_Pad_ = modclass_vbx.coder.ops.permuteOutputVar(CNN4_Pad__, [3 4 2 1], 4);

            CNN4_Pad = dlarray(single(CNN4_Pad_), 'SSCB');
        end

        function [CNN4_Pad, CNN4_PadNumDims1007] = Shape_To_PadGraph1006(this, MaxPool3, MaxPool3NumDims, Training)

            % Execute the operators:
            % Shape:
            [CNN4_InputShapeInt, CNN4_InputShapeIntNumDims] = modclass_vbx.coder.ops.onnxShape(MaxPool3, MaxPool3NumDims, 0, MaxPool3NumDims+1);

            % Cast:
            CNN4_InputShape = single(CNN4_InputShapeInt);
            CNN4_InputShapeNumDims = CNN4_InputShapeIntNumDims;

            % Div:
            CNN4_Div = CNN4_InputShape ./ this.Vars.CNN4_Stride;
            CNN4_DivNumDims = max(CNN4_InputShapeNumDims, this.NumDims.CNN4_Stride);

            % Ceil:
            CNN4_Ceil = ceil(CNN4_Div);
            CNN4_CeilNumDims = CNN4_DivNumDims;

            % Sub:
            CNN4_Sub = CNN4_Ceil - this.Vars.CNN4_One;
            CNN4_SubNumDims = max(CNN4_CeilNumDims, this.NumDims.CNN4_One);

            % Mul:
            CNN4_Mul = CNN4_Sub .* this.Vars.CNN4_Stride;
            CNN4_MulNumDims = max(CNN4_SubNumDims, this.NumDims.CNN4_Stride);

            % Add:
            CNN4_Add = CNN4_Mul + this.Vars.CNN4_Filter;
            CNN4_AddNumDims = max(CNN4_MulNumDims, this.NumDims.CNN4_Filter);

            % Sub:
            CNN4_Sub1 = CNN4_Add - CNN4_InputShape;
            CNN4_Sub1NumDims = max(CNN4_AddNumDims, CNN4_InputShapeNumDims);

            % Relu:
            X1006 = dlarray(modclass_vbx.coder.ops.extractIfDlarray(CNN4_Sub1));
            Y1007 = relu(X1006);
            CNN4_ReluInt = modclass_vbx.coder.ops.extractIfDlarray(Y1007);
            CNN4_ReluIntNumDims = CNN4_Sub1NumDims;

            % Cast:
            CNN4_Relu = single(CNN4_ReluInt);
            CNN4_ReluNumDims = CNN4_ReluIntNumDims;

            % Split:
            [CNN4_BCPadding, CNN4_totalPaddingNee, CNN4_totalPaddingN_5, CNN4_BCPaddingNumDims, CNN4_totalPaddingNeeNumDims, CNN4_totalPaddingN_5NumDims] = modclass_vbx.coder.ops.onnxSplit13(CNN4_Relu, 0, this.CNN4_Split_split, 3, CNN4_ReluNumDims);

            % Div:
            CNN4_totalPaddingN_2 = CNN4_totalPaddingNee ./ this.Vars.CNN4_totalPaddingN_4;
            CNN4_totalPaddingN_2NumDims = max(CNN4_totalPaddingNeeNumDims, this.NumDims.CNN4_totalPaddingN_4);

            % Floor:
            CNN4_totalPaddingN_3 = floor(CNN4_totalPaddingN_2);
            CNN4_totalPaddingN_3NumDims = CNN4_totalPaddingN_2NumDims;

            % Ceil:
            CNN4_totalPaddingN_1 = ceil(CNN4_totalPaddingN_2);
            CNN4_totalPaddingN_1NumDims = CNN4_totalPaddingN_2NumDims;

            % Div:
            CNN4_totalPaddingN_7 = CNN4_totalPaddingN_5 ./ this.Vars.CNN4_totalPaddingN_9;
            CNN4_totalPaddingN_7NumDims = max(CNN4_totalPaddingN_5NumDims, this.NumDims.CNN4_totalPaddingN_9);

            % Floor:
            CNN4_totalPaddingN_8 = floor(CNN4_totalPaddingN_7);
            CNN4_totalPaddingN_8NumDims = CNN4_totalPaddingN_7NumDims;

            % Ceil:
            CNN4_totalPaddingN_6 = ceil(CNN4_totalPaddingN_7);
            CNN4_totalPaddingN_6NumDims = CNN4_totalPaddingN_7NumDims;

            % Concat:
            [CNN4_paddingSize, CNN4_paddingSizeNumDims] = modclass_vbx.coder.ops.onnxConcat(0, {this.Vars.CNN4_Zero, this.Vars.CNN4_Zero, CNN4_totalPaddingN_3, CNN4_totalPaddingN_8, this.Vars.CNN4_Zero, this.Vars.CNN4_Zero, CNN4_totalPaddingN_1, CNN4_totalPaddingN_6}, [this.NumDims.CNN4_Zero, this.NumDims.CNN4_Zero, CNN4_totalPaddingN_3NumDims, CNN4_totalPaddingN_8NumDims, this.NumDims.CNN4_Zero, this.NumDims.CNN4_Zero, CNN4_totalPaddingN_1NumDims, CNN4_totalPaddingN_6NumDims]);

            % Cast:
            CNN4_paddingSizeInt = cast(int64(modclass_vbx.coder.ops.extractIfDlarray(CNN4_paddingSize)), 'like', CNN4_paddingSize);
            CNN4_paddingSizeIntNumDims = CNN4_paddingSizeNumDims;

            % Pad:
            [CNN4_Pad, CNN4_PadNumDims] = modclass_vbx.coder.ops.onnxPad(MaxPool3, CNN4_paddingSizeInt, this.CNN4_Pad_value, 'constant', [0:MaxPool3NumDims]', MaxPool3NumDims);

            % Set graph output arguments
            CNN4_PadNumDims1007 = CNN4_PadNumDims;

        end

    end

end