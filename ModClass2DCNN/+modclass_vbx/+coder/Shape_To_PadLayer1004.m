classdef Shape_To_PadLayer1004 < nnet.layer.Layer & nnet.layer.Formattable
    % A custom layer auto-generated while importing an ONNX network.
    %#codegen

    %#ok<*PROPLC>
    %#ok<*NBRAK>
    %#ok<*INUSL>
    %#ok<*VARARG>
    properties (Learnable)
        CNN5_Split_split
        CNN5_Pad_value
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
                'CNN5_Split_split'
                'CNN5_Pad_value'
                };
        end
    end


    methods(Static, Hidden)
        % Instantiate a codegenable layer instance from a MATLAB layer instance
        function this_cg = matlabCodegenToRedirected(mlInstance)
            this_cg = modclass_vbx.coder.Shape_To_PadLayer1004(mlInstance);
        end
        function this_ml = matlabCodegenFromRedirected(cgInstance)
            this_ml = modclass_vbx.Shape_To_PadLayer1004(cgInstance.Name);
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
            this_ml.CNN5_Split_split = cgInstance.CNN5_Split_split;
            this_ml.CNN5_Pad_value = cgInstance.CNN5_Pad_value;
        end
    end

    methods
        function this = Shape_To_PadLayer1004(mlInstance)
            this.Name = mlInstance.Name;
            this.OutputNames = {'CNN5_Pad'};
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
            this.CNN5_Split_split = mlInstance.CNN5_Split_split;
            this.CNN5_Pad_value = mlInstance.CNN5_Pad_value;
        end

        function [CNN5_Pad] = predict(this, MaxPool4__)
            if isdlarray(MaxPool4__)
                MaxPool4_ = stripdims(MaxPool4__);
            else
                MaxPool4_ = MaxPool4__;
            end
            MaxPool4NumDims = 4;
            MaxPool4 = modclass_vbx.coder.ops.permuteInputVar(MaxPool4_, [4 3 1 2], 4);

            [CNN5_Pad__, CNN5_PadNumDims__] = Shape_To_PadGraph1008(this, MaxPool4, MaxPool4NumDims, false);
            CNN5_Pad_ = modclass_vbx.coder.ops.permuteOutputVar(CNN5_Pad__, [3 4 2 1], 4);

            CNN5_Pad = dlarray(single(CNN5_Pad_), 'SSCB');
        end

        function [CNN5_Pad, CNN5_PadNumDims1009] = Shape_To_PadGraph1008(this, MaxPool4, MaxPool4NumDims, Training)

            % Execute the operators:
            % Shape:
            [CNN5_InputShapeInt, CNN5_InputShapeIntNumDims] = modclass_vbx.coder.ops.onnxShape(MaxPool4, MaxPool4NumDims, 0, MaxPool4NumDims+1);

            % Cast:
            CNN5_InputShape = single(CNN5_InputShapeInt);
            CNN5_InputShapeNumDims = CNN5_InputShapeIntNumDims;

            % Div:
            CNN5_Div = CNN5_InputShape ./ this.Vars.CNN5_Stride;
            CNN5_DivNumDims = max(CNN5_InputShapeNumDims, this.NumDims.CNN5_Stride);

            % Ceil:
            CNN5_Ceil = ceil(CNN5_Div);
            CNN5_CeilNumDims = CNN5_DivNumDims;

            % Sub:
            CNN5_Sub = CNN5_Ceil - this.Vars.CNN5_One;
            CNN5_SubNumDims = max(CNN5_CeilNumDims, this.NumDims.CNN5_One);

            % Mul:
            CNN5_Mul = CNN5_Sub .* this.Vars.CNN5_Stride;
            CNN5_MulNumDims = max(CNN5_SubNumDims, this.NumDims.CNN5_Stride);

            % Add:
            CNN5_Add = CNN5_Mul + this.Vars.CNN5_Filter;
            CNN5_AddNumDims = max(CNN5_MulNumDims, this.NumDims.CNN5_Filter);

            % Sub:
            CNN5_Sub1 = CNN5_Add - CNN5_InputShape;
            CNN5_Sub1NumDims = max(CNN5_AddNumDims, CNN5_InputShapeNumDims);

            % Relu:
            X1008 = dlarray(modclass_vbx.coder.ops.extractIfDlarray(CNN5_Sub1));
            Y1009 = relu(X1008);
            CNN5_ReluInt = modclass_vbx.coder.ops.extractIfDlarray(Y1009);
            CNN5_ReluIntNumDims = CNN5_Sub1NumDims;

            % Cast:
            CNN5_Relu = single(CNN5_ReluInt);
            CNN5_ReluNumDims = CNN5_ReluIntNumDims;

            % Split:
            [CNN5_BCPadding, CNN5_totalPaddingNee, CNN5_totalPaddingN_5, CNN5_BCPaddingNumDims, CNN5_totalPaddingNeeNumDims, CNN5_totalPaddingN_5NumDims] = modclass_vbx.coder.ops.onnxSplit13(CNN5_Relu, 0, this.CNN5_Split_split, 3, CNN5_ReluNumDims);

            % Div:
            CNN5_totalPaddingN_2 = CNN5_totalPaddingNee ./ this.Vars.CNN5_totalPaddingN_4;
            CNN5_totalPaddingN_2NumDims = max(CNN5_totalPaddingNeeNumDims, this.NumDims.CNN5_totalPaddingN_4);

            % Floor:
            CNN5_totalPaddingN_3 = floor(CNN5_totalPaddingN_2);
            CNN5_totalPaddingN_3NumDims = CNN5_totalPaddingN_2NumDims;

            % Ceil:
            CNN5_totalPaddingN_1 = ceil(CNN5_totalPaddingN_2);
            CNN5_totalPaddingN_1NumDims = CNN5_totalPaddingN_2NumDims;

            % Div:
            CNN5_totalPaddingN_7 = CNN5_totalPaddingN_5 ./ this.Vars.CNN5_totalPaddingN_9;
            CNN5_totalPaddingN_7NumDims = max(CNN5_totalPaddingN_5NumDims, this.NumDims.CNN5_totalPaddingN_9);

            % Floor:
            CNN5_totalPaddingN_8 = floor(CNN5_totalPaddingN_7);
            CNN5_totalPaddingN_8NumDims = CNN5_totalPaddingN_7NumDims;

            % Ceil:
            CNN5_totalPaddingN_6 = ceil(CNN5_totalPaddingN_7);
            CNN5_totalPaddingN_6NumDims = CNN5_totalPaddingN_7NumDims;

            % Concat:
            [CNN5_paddingSize, CNN5_paddingSizeNumDims] = modclass_vbx.coder.ops.onnxConcat(0, {this.Vars.CNN5_Zero, this.Vars.CNN5_Zero, CNN5_totalPaddingN_3, CNN5_totalPaddingN_8, this.Vars.CNN5_Zero, this.Vars.CNN5_Zero, CNN5_totalPaddingN_1, CNN5_totalPaddingN_6}, [this.NumDims.CNN5_Zero, this.NumDims.CNN5_Zero, CNN5_totalPaddingN_3NumDims, CNN5_totalPaddingN_8NumDims, this.NumDims.CNN5_Zero, this.NumDims.CNN5_Zero, CNN5_totalPaddingN_1NumDims, CNN5_totalPaddingN_6NumDims]);

            % Cast:
            CNN5_paddingSizeInt = cast(int64(modclass_vbx.coder.ops.extractIfDlarray(CNN5_paddingSize)), 'like', CNN5_paddingSize);
            CNN5_paddingSizeIntNumDims = CNN5_paddingSizeNumDims;

            % Pad:
            [CNN5_Pad, CNN5_PadNumDims] = modclass_vbx.coder.ops.onnxPad(MaxPool4, CNN5_paddingSizeInt, this.CNN5_Pad_value, 'constant', [0:MaxPool4NumDims]', MaxPool4NumDims);

            % Set graph output arguments
            CNN5_PadNumDims1009 = CNN5_PadNumDims;

        end

    end

end