classdef Shape_To_PadLayer1001 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        CNN2_Split_split
        CNN2_Pad_value
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
                'CNN2_Split_split'
                'CNN2_Pad_value'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = modclass_vbx.coder.Shape_To_PadLayer1001(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = modclass_vbx.Shape_To_PadLayer1001(cgInstance.Name);
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
            this_ml.CNN2_Split_split = cgInstance.CNN2_Split_split;
            this_ml.CNN2_Pad_value = cgInstance.CNN2_Pad_value;
        end
    end

    methods
        function this = Shape_To_PadLayer1001(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'CNN2_Pad'};
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
            this.CNN2_Split_split = mlInstance.CNN2_Split_split;
            this.CNN2_Pad_value = mlInstance.CNN2_Pad_value;
        end

        function [CNN2_Pad] = predict(this, MaxPool1__)
            if isdlarray(MaxPool1__)
                MaxPool1_ = stripdims(MaxPool1__);
            else
                MaxPool1_ = MaxPool1__;
            end
            MaxPool1NumDims = 4;
            MaxPool1 = modclass_vbx.coder.ops.permuteInputVar(MaxPool1_, [4 3 1 2], 4);

            [CNN2_Pad__, CNN2_PadNumDims__] = Shape_To_PadGraph1002(this, MaxPool1, MaxPool1NumDims, false);
            CNN2_Pad_ = modclass_vbx.coder.ops.permuteOutputVar(CNN2_Pad__, [3 4 2 1], 4);

            CNN2_Pad = dlarray(single(CNN2_Pad_), 'SSCB');
        end

        function [CNN2_Pad, CNN2_PadNumDims1003] = Shape_To_PadGraph1002(this, MaxPool1, MaxPool1NumDims, Training)

            % Execute the operators:
            % Shape:
            [CNN2_InputShapeInt, CNN2_InputShapeIntNumDims] = modclass_vbx.coder.ops.onnxShape(MaxPool1, MaxPool1NumDims, 0, MaxPool1NumDims+1);

            % Cast:
            CNN2_InputShape = single(CNN2_InputShapeInt);
            CNN2_InputShapeNumDims = CNN2_InputShapeIntNumDims;

            % Div:
            CNN2_Div = CNN2_InputShape ./ this.Vars.CNN2_Stride;
            CNN2_DivNumDims = max(CNN2_InputShapeNumDims, this.NumDims.CNN2_Stride);

            % Ceil:
            CNN2_Ceil = ceil(CNN2_Div);
            CNN2_CeilNumDims = CNN2_DivNumDims;

            % Sub:
            CNN2_Sub = CNN2_Ceil - this.Vars.CNN2_One;
            CNN2_SubNumDims = max(CNN2_CeilNumDims, this.NumDims.CNN2_One);

            % Mul:
            CNN2_Mul = CNN2_Sub .* this.Vars.CNN2_Stride;
            CNN2_MulNumDims = max(CNN2_SubNumDims, this.NumDims.CNN2_Stride);

            % Add:
            CNN2_Add = CNN2_Mul + this.Vars.CNN2_Filter;
            CNN2_AddNumDims = max(CNN2_MulNumDims, this.NumDims.CNN2_Filter);

            % Sub:
            CNN2_Sub1 = CNN2_Add - CNN2_InputShape;
            CNN2_Sub1NumDims = max(CNN2_AddNumDims, CNN2_InputShapeNumDims);

            % Relu:
            X1002 = dlarray(modclass_vbx.coder.ops.extractIfDlarray(CNN2_Sub1));
            Y1003 = relu(X1002);
            CNN2_ReluInt = modclass_vbx.coder.ops.extractIfDlarray(Y1003);
            CNN2_ReluIntNumDims = CNN2_Sub1NumDims;

            % Cast:
            CNN2_Relu = single(CNN2_ReluInt);
            CNN2_ReluNumDims = CNN2_ReluIntNumDims;

            % Split:
            [CNN2_BCPadding, CNN2_totalPaddingNee, CNN2_totalPaddingN_5, CNN2_BCPaddingNumDims, CNN2_totalPaddingNeeNumDims, CNN2_totalPaddingN_5NumDims] = modclass_vbx.coder.ops.onnxSplit13(CNN2_Relu, 0, this.CNN2_Split_split, 3, CNN2_ReluNumDims);

            % Div:
            CNN2_totalPaddingN_2 = CNN2_totalPaddingNee ./ this.Vars.CNN2_totalPaddingN_4;
            CNN2_totalPaddingN_2NumDims = max(CNN2_totalPaddingNeeNumDims, this.NumDims.CNN2_totalPaddingN_4);

            % Floor:
            CNN2_totalPaddingN_3 = floor(CNN2_totalPaddingN_2);
            CNN2_totalPaddingN_3NumDims = CNN2_totalPaddingN_2NumDims;

            % Ceil:
            CNN2_totalPaddingN_1 = ceil(CNN2_totalPaddingN_2);
            CNN2_totalPaddingN_1NumDims = CNN2_totalPaddingN_2NumDims;

            % Div:
            CNN2_totalPaddingN_7 = CNN2_totalPaddingN_5 ./ this.Vars.CNN2_totalPaddingN_9;
            CNN2_totalPaddingN_7NumDims = max(CNN2_totalPaddingN_5NumDims, this.NumDims.CNN2_totalPaddingN_9);

            % Floor:
            CNN2_totalPaddingN_8 = floor(CNN2_totalPaddingN_7);
            CNN2_totalPaddingN_8NumDims = CNN2_totalPaddingN_7NumDims;

            % Ceil:
            CNN2_totalPaddingN_6 = ceil(CNN2_totalPaddingN_7);
            CNN2_totalPaddingN_6NumDims = CNN2_totalPaddingN_7NumDims;

            % Concat:
            [CNN2_paddingSize, CNN2_paddingSizeNumDims] = modclass_vbx.coder.ops.onnxConcat(0, {this.Vars.CNN2_Zero, this.Vars.CNN2_Zero, CNN2_totalPaddingN_3, CNN2_totalPaddingN_8, this.Vars.CNN2_Zero, this.Vars.CNN2_Zero, CNN2_totalPaddingN_1, CNN2_totalPaddingN_6}, [this.NumDims.CNN2_Zero, this.NumDims.CNN2_Zero, CNN2_totalPaddingN_3NumDims, CNN2_totalPaddingN_8NumDims, this.NumDims.CNN2_Zero, this.NumDims.CNN2_Zero, CNN2_totalPaddingN_1NumDims, CNN2_totalPaddingN_6NumDims]);

            % Cast:
            CNN2_paddingSizeInt = cast(int64(modclass_vbx.coder.ops.extractIfDlarray(CNN2_paddingSize)), 'like', CNN2_paddingSize);
            CNN2_paddingSizeIntNumDims = CNN2_paddingSizeNumDims;

            % Pad:
            [CNN2_Pad, CNN2_PadNumDims] = modclass_vbx.coder.ops.onnxPad(MaxPool1, CNN2_paddingSizeInt, this.CNN2_Pad_value, 'constant', [0:MaxPool1NumDims]', MaxPool1NumDims);

            % Set graph output arguments
            CNN2_PadNumDims1003 = CNN2_PadNumDims;

        end

    end

end